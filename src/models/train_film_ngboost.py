"""FiLM-conditional MLP — Track A 보완 (NGBoost capacity 한계 돌파).

Architecture:
  Main features (continuous physics):
    dsr_mean, zenith_center, ta, hm, ws → MLP encoder → h
  Conditioning (regime):
    cloud (dc10Tca), hour_sin/cos, month_sin/cos, site_one_hot → MLP_cond → γ, β
  FiLM modulation:
    h' = γ * h + β
  Output:
    mu = Linear(h')
    sigma = softplus(Linear(h'))

Loss: Gaussian NLL (NGBoost와 동일)
   → 시나리오/reweight 파이프라인 그대로 호환

Train: 2022-2023, Val: 2024, Test: 2025 (NGBoost와 동일)
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

# Config
MAIN_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws"]
COND_NUM_FEATURES = ["dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos"]
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

HIDDEN = 64
LR = 1e-3
BATCH_SIZE = 4096
MAX_EPOCHS = 100
PATIENCE = 8


# ========== Data ==========

def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    return df


def build_features(df, all_sites=None):
    if all_sites is None:
        all_sites = sorted(df["site"].unique())
    main = df[MAIN_FEATURES].values.astype(np.float32)
    cond_num = df[COND_NUM_FEATURES].values.astype(np.float32)
    site_oh = np.zeros((len(df), len(all_sites)), dtype=np.float32)
    for i, s in enumerate(all_sites):
        site_oh[:, i] = (df["site"] == s).astype(np.float32)
    cond = np.concatenate([cond_num, site_oh], axis=1)
    y = df[TARGET].values.astype(np.float32)
    return main, cond, y, all_sites


# ========== Model ==========

class FiLMRegressor(nn.Module):
    def __init__(self, n_main, n_cond, hidden=HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_main, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        # Conditioning → FiLM (γ, β)
        self.cond_net = nn.Sequential(
            nn.Linear(n_cond, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * hidden),
        )
        # Heads
        self.mu_head = nn.Linear(hidden, 1)
        self.log_sig_head = nn.Linear(hidden, 1)

    def forward(self, x_main, x_cond):
        h = self.encoder(x_main)
        γβ = self.cond_net(x_cond)
        γ, β = γβ.chunk(2, dim=-1)
        h = γ * h + β
        mu = self.mu_head(h).squeeze(-1)
        log_sig = self.log_sig_head(h).squeeze(-1)
        sigma = F.softplus(log_sig) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


# ========== Train ==========

def train_model(train, val, device):
    print(f"\n  [Train] n_train={len(train[2]):,}, n_val={len(val[2]):,}")
    main_tr, cond_tr, y_tr = train
    main_v, cond_v, y_v = val

    # Standardize main features (cond features mostly already scaled)
    mean_main = main_tr.mean(axis=0)
    std_main = main_tr.std(axis=0) + 1e-6
    main_tr_s = (main_tr - mean_main) / std_main
    main_v_s = (main_v - mean_main) / std_main

    # Cond: dc10Tca scale to [0,1]
    cond_tr_s = cond_tr.copy()
    cond_v_s = cond_v.copy()
    cond_tr_s[:, 0] = cond_tr_s[:, 0] / 10.0
    cond_v_s[:, 0] = cond_v_s[:, 0] / 10.0

    n_main = main_tr.shape[1]
    n_cond = cond_tr.shape[1]
    print(f"  n_main={n_main}, n_cond={n_cond}, hidden={HIDDEN}")

    model = FiLMRegressor(n_main, n_cond).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    train_ds = TensorDataset(
        torch.from_numpy(main_tr_s),
        torch.from_numpy(cond_tr_s),
        torch.from_numpy(y_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(main_v_s),
        torch.from_numpy(cond_v_s),
        torch.from_numpy(y_v),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        tr_loss = 0
        n = 0
        for xm, xc, y in train_loader:
            xm, xc, y = xm.to(device), xc.to(device), y.to(device)
            mu, sig = model(xm, xc)
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
            for xm, xc, y in val_loader:
                xm, xc, y = xm.to(device), xc.to(device), y.to(device)
                mu, sig = model(xm, xc)
                v_loss += gaussian_nll(y, mu, sig).mean().item() * len(y)
                v_mae += (y - mu).abs().sum().item()
                nv += len(y)
        v_loss /= nv
        v_mae /= nv

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
            marker = "★"
        else:
            bad += 1
            marker = ""

        print(f"  ep {ep:>3}: tr_loss {tr_loss:.4f}  val_loss {v_loss:.4f}  val_mae {v_mae:.4f} {marker}")
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}")
            break

    model.load_state_dict(best_state)
    return model, mean_main, std_main


# ========== Evaluate ==========

def evaluate(model, df_test, all_sites, mean_main, std_main, device):
    main, cond, y, _ = build_features(df_test, all_sites)
    main_s = (main - mean_main) / std_main
    cond_s = cond.copy()
    cond_s[:, 0] = cond_s[:, 0] / 10.0

    model.eval()
    mus = []
    sigs = []
    with torch.no_grad():
        bs = 8192
        for i in range(0, len(y), bs):
            xm = torch.from_numpy(main_s[i:i + bs]).to(device)
            xc = torch.from_numpy(cond_s[i:i + bs]).to(device)
            mu, sig = model(xm, xc)
            mus.append(mu.cpu().numpy())
            sigs.append(sig.cpu().numpy())
    mu_all = np.clip(np.concatenate(mus), 0, None)
    sig_all = np.concatenate(sigs)

    cap = df_test["site_capacity_kw"].values
    err = np.abs(y - mu_all)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu_all - 1.28 * sig_all) & (y <= mu_all + 1.28 * sig_all)).mean() * 100
    cov95 = ((y >= mu_all - 1.96 * sig_all) & (y <= mu_all + 1.96 * sig_all)).mean() * 100
    print(f"\n=== TEST 평가 ===")
    print(f"  rows: {len(y):,}")
    print(f"  MAE (cf):       {err.mean():.4f}")
    print(f"  NMAE (capacity): {nmae:.2f}%")
    print(f"  80% coverage:   {cov80:.1f}%")
    print(f"  95% coverage:   {cov95:.1f}%")

    # Per-site
    df_eval = df_test.copy()
    df_eval["mu_pred"] = mu_all
    df_eval["sig_pred"] = sig_all
    df_eval["err"] = err
    print(f"\n  사이트별 NMAE:")
    print(f"  {'site':<14} {'cap MW':>7} {'NMAE %':>8}")
    for s in sorted(df_eval["site"].unique()):
        sub = df_eval[df_eval["site"] == s]
        c = sub["site_capacity_kw"].iloc[0] / 1000
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        print(f"  {s:<14} {c:>7.1f} {n:>7.2f}%")

    # Cloud bin별
    df_eval["cloud_bin"] = pd.cut(df_eval["dc10Tca"], bins=[-1, 3, 5, 7, 9, 11],
                                   labels=["0-3", "3-5", "5-7", "7-9", "9-10"])
    print(f"\n  Cloud bin별 NMAE:")
    print(f"  {'cloud':<10} {'n':>7} {'NMAE %':>8}")
    for b in ["0-3", "3-5", "5-7", "7-9", "9-10"]:
        sub = df_eval[df_eval["cloud_bin"] == b]
        if len(sub) == 0:
            continue
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        print(f"  {b:<10} {len(sub):>7,} {n:>7.2f}%")

    # Portfolio
    df_eval["pred_kwh"] = mu_all * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = y * df_eval["site_capacity_kw"]
    df_eval["var_kwh2"] = (sig_all * df_eval["site_capacity_kw"]) ** 2
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        var_sum=("var_kwh2", "sum"), cap=("site_capacity_kw", "sum"),
    )
    port["err"] = (port["pred"] - port["actual"]).abs()
    port_nmae = port["err"].sum() / port["cap"].sum() * 100
    port_crash = ((port["actual"] < port["pred"] * 0.5)).sum()
    print(f"\n  포트폴리오 NMAE: {port_nmae:.2f}%  / crash {port_crash}")

    return mu_all, sig_all, nmae, port_nmae, port_crash, cov80


def main():
    print("=" * 70)
    print("FiLM-NGBoost — Track A 보완 (cloud-conditional MLP)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n[1/4] 데이터 로딩...")
    df = load_data()
    train_df = df[df.datetime_kst < TRAIN_END].copy()
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)].copy()
    test_df = df[df.datetime_kst >= VAL_END].copy()
    print(f"  train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")

    print("\n[2/4] Feature build...")
    all_sites = sorted(df["site"].unique())
    main_tr, cond_tr, y_tr, _ = build_features(train_df, all_sites)
    main_v, cond_v, y_v, _ = build_features(val_df, all_sites)
    print(f"  main features ({len(MAIN_FEATURES)}): {MAIN_FEATURES}")
    print(f"  cond features ({cond_tr.shape[1]}): {COND_NUM_FEATURES} + {len(all_sites)} site-onehot")

    print("\n[3/4] FiLM 학습...")
    model, mean_main, std_main = train_model(
        (main_tr, cond_tr, y_tr),
        (main_v, cond_v, y_v),
        device,
    )

    print("\n[4/4] Test 평가...")
    mu, sig, nmae, port_nmae, port_crash, cov80 = evaluate(
        model, test_df, all_sites, mean_main, std_main, device
    )

    # Save
    out_dir = ROOT / "pv/experiments/film_ngboost"
    out_dir.mkdir(parents=True, exist_ok=True)
    test_out = test_df[["datetime_kst", "cf", "site_capacity_kw", "site"]].copy()
    test_out["pred_cf"] = mu
    test_out["pred_std_cf"] = sig
    test_out.to_parquet(out_dir / "test_predictions.parquet", index=False)
    torch.save({
        "state_dict": model.state_dict(),
        "mean_main": mean_main,
        "std_main": std_main,
        "all_sites": all_sites,
        "main_features": MAIN_FEATURES,
        "cond_num_features": COND_NUM_FEATURES,
    }, out_dir / "model.pt")
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}")
    print(f"저장: {out_dir / 'model.pt'}")

    print("\n" + "=" * 70)
    print("비교")
    print("=" * 70)
    print(f"  Strong NGBoost:  6.07% NMAE / 5.04% portfolio / cov80 81.7%")
    print(f"  FiLM:            {nmae:.2f}% NMAE / {port_nmae:.2f}% portfolio / cov80 {cov80:.1f}%")


if __name__ == "__main__":
    main()
