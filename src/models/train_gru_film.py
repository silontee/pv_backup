"""GRU + FiLM — Track A 한 단계 더.

vs MLP + FiLM 차이:
  - 입력: 단일 시점 → 24h 시퀀스 (시간 의존성 학습)
  - Backbone: MLP → GRU (recurrent)
  - 나머지 동일: FiLM modulation, Gaussian NLL, 같은 features

Architecture:
  Main seq (B, 24, 5):  dsr, zenith, ta, hm, ws — 24h history
    → GRU(hidden=32) → last hidden h
  Cond now (B, n_cond): cloud, hour_sin/cos, month_sin/cos, site
    → MLP_cond → γ, β
  FiLM:  h' = γ * h + β
  Output: mu, sigma (Normal)

데이터 부족 risk → 작은 모델 (hidden=32, 1 layer)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# Config (MLP+FiLM과 동일 기본)
MAIN_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws"]
COND_NUM_FEATURES = ["dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos"]
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

SEQ_LEN = 24       # 24h history
HIDDEN = 32        # 작게 시작
N_LAYERS = 1
LR = 1e-3
BATCH_SIZE = 2048
MAX_EPOCHS = 60
PATIENCE = 6


# ========== Data ==========

def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)
    # 1. Daytime mark BEFORE fillna (원본 NaN 기반)
    df["is_daytime"] = (df["dsr_mean"].notna() & df["cf"].notna()).astype(np.int32)
    # 2. 시퀀스 연속성을 위해 야간 fillna (학습 input 용)
    for col in ["ta", "hm", "ws", "dc10Tca", "dsr_mean", "zenith_center"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    df["cf"] = df["cf"].fillna(0)   # 야간 cf=0 (loss mask로 무시할 거)
    return df


def build_sequences(df, all_sites, mean_main, std_main):
    """각 daytime 시점에 대해 24h history 시퀀스 + 현재 cond 빌드.

    Returns:
      main_seq: (N, 24, 5)
      cond_now: (N, n_cond)
      y: (N,)
      meta: DataFrame with datetime, site, capacity for evaluation
    """
    sequences = []
    cond_arr = []
    targets = []
    meta_rows = []

    site_idx = {s: i for i, s in enumerate(all_sites)}
    n_sites = len(all_sites)

    main_cols = MAIN_FEATURES
    for site, sub in df.groupby("site"):
        sub = sub.sort_values("datetime_kst").reset_index(drop=True)
        # main features array
        m = sub[main_cols].values.astype(np.float32)
        # standardize
        m = (m - mean_main) / std_main
        # cond
        cn = sub[COND_NUM_FEATURES].values.astype(np.float32)
        cn[:, 0] = cn[:, 0] / 10.0   # cloud → [0,1]
        # site one-hot
        soh = np.zeros((len(sub), n_sites), dtype=np.float32)
        soh[:, site_idx[site]] = 1.0
        c = np.concatenate([cn, soh], axis=1)
        # daytime mask — 원본 NaN 기반 (loss masking)
        is_day = sub["is_daytime"].values
        for t in range(SEQ_LEN, len(sub)):
            # NGBoost와 동일: daytime target만 학습/평가
            if not is_day[t]:
                continue
            sequences.append(m[t - SEQ_LEN:t])  # 과거 24h (야간 포함, fillna된 값)
            cond_arr.append(c[t])
            targets.append(sub["cf"].iloc[t])
            meta_rows.append({
                "datetime_kst": sub["datetime_kst"].iloc[t],
                "site": site,
                "site_capacity_kw": sub["site_capacity_kw"].iloc[t],
            })

    main_seq = np.stack(sequences)   # (N, 24, 5)
    cond_now = np.stack(cond_arr)
    y = np.array(targets, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return main_seq, cond_now, y, meta


# ========== Model ==========

class GRUFiLM(nn.Module):
    def __init__(self, n_main, n_cond, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.gru = nn.GRU(n_main, hidden, n_layers, batch_first=True)
        self.cond_net = nn.Sequential(
            nn.Linear(n_cond, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * hidden),
        )
        self.mu_head = nn.Linear(hidden, 1)
        self.log_sig_head = nn.Linear(hidden, 1)

    def forward(self, x_main_seq, x_cond_now):
        out, _ = self.gru(x_main_seq)
        h_last = out[:, -1]
        γβ = self.cond_net(x_cond_now)
        γ, β = γβ.chunk(2, dim=-1)
        h = γ * h_last + β
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h)).squeeze(-1) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


# ========== Train ==========

def train_loop(model, train_loader, val_loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        tr_loss = 0
        n = 0
        for ms, cn, y in train_loader:
            ms, cn, y = ms.to(device), cn.to(device), y.to(device)
            mu, sig = model(ms, cn)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(y)
            n += len(y)
        tr_loss /= n
        scheduler.step()

        model.eval()
        v_loss = 0
        v_mae = 0
        nv = 0
        with torch.no_grad():
            for ms, cn, y in val_loader:
                ms, cn, y = ms.to(device), cn.to(device), y.to(device)
                mu, sig = model(ms, cn)
                v_loss += gaussian_nll(y, mu, sig).mean().item() * len(y)
                v_mae += (y - mu).abs().sum().item()
                nv += len(y)
        v_loss /= nv
        v_mae /= nv

        marker = ""
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            marker = "★"
        else:
            bad += 1

        print(f"  ep {ep:>3}: tr_loss {tr_loss:.4f}  val_loss {v_loss:.4f}  val_mae {v_mae:.4f} {marker}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True)
            break

    model.load_state_dict(best_state)
    return model


def evaluate(mu_arr, sig_arr, y, meta):
    cap = meta["site_capacity_kw"].values
    err = np.abs(y - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu_arr - 1.28 * sig_arr) & (y <= mu_arr + 1.28 * sig_arr)).mean() * 100
    cov95 = ((y >= mu_arr - 1.96 * sig_arr) & (y <= mu_arr + 1.96 * sig_arr)).mean() * 100
    print(f"\n=== TEST 평가 ===")
    print(f"  rows: {len(y):,}")
    print(f"  MAE (cf): {err.mean():.4f}")
    print(f"  NMAE (capacity): {nmae:.2f}%")
    print(f"  80% coverage: {cov80:.1f}%")
    print(f"  95% coverage: {cov95:.1f}%")

    df_eval = meta.copy()
    df_eval["cf"] = y
    df_eval["pred_cf"] = mu_arr
    df_eval["pred_std"] = sig_arr
    df_eval["err"] = err
    print(f"\n  사이트별 NMAE:")
    print(f"  {'site':<14} {'cap MW':>7} {'NMAE %':>8}")
    for s in sorted(df_eval["site"].unique()):
        sub = df_eval[df_eval["site"] == s]
        c = sub["site_capacity_kw"].iloc[0] / 1000
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        print(f"  {s:<14} {c:>7.1f} {n:>7.2f}%")

    # Portfolio
    df_eval["pred_kwh"] = mu_arr * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = y * df_eval["site_capacity_kw"]
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"),
    )
    port_nmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    port_crash = ((port["actual"] < port["pred"] * 0.5)).sum()
    print(f"\n  포트폴리오 NMAE: {port_nmae:.2f}% / crash {port_crash}")

    return df_eval, nmae, cov80, port_nmae


def main():
    print("=" * 70)
    print("GRU + FiLM (Track A — sequential backbone)")
    print(f"  SEQ_LEN={SEQ_LEN}, HIDDEN={HIDDEN}, N_LAYERS={N_LAYERS}")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/4] 데이터 로딩 + 시퀀스 빌드...")
    df = load_data()
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]
    all_sites = sorted(df["site"].unique())

    # standardize main from train
    mean_main = train_df[MAIN_FEATURES].mean().values.astype(np.float32)
    std_main = (train_df[MAIN_FEATURES].std().values + 1e-6).astype(np.float32)

    print("  building sequences...", flush=True)
    main_tr, cond_tr, y_tr, meta_tr = build_sequences(train_df, all_sites, mean_main, std_main)
    main_v, cond_v, y_v, meta_v = build_sequences(val_df, all_sites, mean_main, std_main)
    main_te, cond_te, y_te, meta_te = build_sequences(test_df, all_sites, mean_main, std_main)
    print(f"  train seq: {main_tr.shape}, val: {main_v.shape}, test: {main_te.shape}", flush=True)

    n_main = main_tr.shape[2]
    n_cond = cond_tr.shape[1]
    print(f"  n_main={n_main}, n_cond={n_cond}")

    print(f"\n[2/4] GRU+FiLM 모델 빌드...")
    model = GRUFiLM(n_main, n_cond).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e3:.1f}k")

    train_ds = TensorDataset(torch.from_numpy(main_tr), torch.from_numpy(cond_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(main_v), torch.from_numpy(cond_v), torch.from_numpy(y_v))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("\n[3/4] 학습...")
    model = train_loop(model, train_loader, val_loader, device)

    print("\n[4/4] Test 추론...")
    test_ds = TensorDataset(torch.from_numpy(main_te), torch.from_numpy(cond_te), torch.from_numpy(y_te))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    model.eval()
    mus = []
    sigs = []
    with torch.no_grad():
        for ms, cn, _ in test_loader:
            ms, cn = ms.to(device), cn.to(device)
            mu, sig = model(ms, cn)
            mus.append(mu.cpu().numpy())
            sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)

    df_eval, nmae, cov80, port_nmae = evaluate(mu_arr, sig_arr, y_te, meta_te)

    out_dir = ROOT / "pv/experiments/gru_film"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_eval.rename(columns={"pred_std": "pred_std_cf", "pred_cf": "pred_cf"}).to_parquet(
        out_dir / "test_predictions.parquet", index=False
    )
    torch.save({
        "state_dict": model.state_dict(),
        "mean_main": mean_main, "std_main": std_main,
        "all_sites": all_sites,
        "main_features": MAIN_FEATURES,
        "cond_num_features": COND_NUM_FEATURES,
    }, out_dir / "model.pt")
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}")

    print("\n" + "=" * 70)
    print("비교 정리")
    print("=" * 70)
    print(f"  Strong NGBoost: 6.07% NMAE / 5.04% portfolio / cov80 81.7%")
    print(f"  MLP + FiLM:     5.88% NMAE / 4.88% portfolio / cov80 84.3%")
    print(f"  GRU + FiLM:     {nmae:.2f}% NMAE / {port_nmae:.2f}% portfolio / cov80 {cov80:.1f}%")


if __name__ == "__main__":
    main()
