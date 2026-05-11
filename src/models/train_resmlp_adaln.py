"""ResMLP + AdaLN — Track A 정교화.

vs MLP+FiLM 차이:
  - Backbone: shallow MLP → ResMLP (residual block × N)
  - Modulation: post-FiLM → AdaLN (Adaptive Layer Norm — conditioning이 LN scale/shift 직접 modulate)
  - 깊이/안정성 ↑

Architecture:
  Input:  main features (5) + cond features → joint 입력
  Linear → x_0 (dim=64)
  for i in range(N):
    norm = LN(x_i)
    γ, β = MLP_cond(cond)  # AdaLN params
    h = γ * norm + β
    x_{i+1} = x_i + MLP(h)  # residual
  Output: μ, σ = head(x_N)
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

MAIN_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws"]
COND_NUM_FEATURES = ["dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos"]
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

DIM = 64
N_BLOCKS = 4
LR = 1e-3
BATCH_SIZE = 4096
MAX_EPOCHS = 100
PATIENCE = 10


def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    return df


def build_features(df, all_sites):
    main = df[MAIN_FEATURES].values.astype(np.float32)
    cond_num = df[COND_NUM_FEATURES].values.astype(np.float32)
    site_oh = np.zeros((len(df), len(all_sites)), dtype=np.float32)
    for i, s in enumerate(all_sites):
        site_oh[:, i] = (df["site"] == s).astype(np.float32)
    cond = np.concatenate([cond_num, site_oh], axis=1)
    y = df[TARGET].values.astype(np.float32)
    return main, cond, y


class AdaLNBlock(nn.Module):
    def __init__(self, dim, n_cond, mlp_ratio=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_to_ss = nn.Linear(n_cond, 2 * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.cond_to_ss(cond).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = self.mlp(h)
        return x + h


class ResMLPAdaLN(nn.Module):
    def __init__(self, n_main, n_cond, dim=DIM, n_blocks=N_BLOCKS):
        super().__init__()
        self.input_proj = nn.Linear(n_main, dim)
        self.blocks = nn.ModuleList([AdaLNBlock(dim, n_cond) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_final = nn.Linear(n_cond, 2 * dim)
        self.mu_head = nn.Linear(dim, 1)
        self.log_sig_head = nn.Linear(dim, 1)

    def forward(self, x_main, x_cond):
        x = self.input_proj(x_main)
        for blk in self.blocks:
            x = blk(x, x_cond)
        h = self.final_norm(x)
        scale, shift = self.cond_final(x_cond).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h)).squeeze(-1) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


def train_loop(model, train_loader, val_loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best_val = float("inf")
    best_state = None
    bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        tr_loss = 0; n = 0
        for xm, xc, y in train_loader:
            xm, xc, y = xm.to(device), xc.to(device), y.to(device)
            mu, sig = model(xm, xc)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(y); n += len(y)
        tr_loss /= n; scheduler.step()
        model.eval()
        v_loss = 0; v_mae = 0; nv = 0
        with torch.no_grad():
            for xm, xc, y in val_loader:
                xm, xc, y = xm.to(device), xc.to(device), y.to(device)
                mu, sig = model(xm, xc)
                v_loss += gaussian_nll(y, mu, sig).mean().item() * len(y)
                v_mae += (y - mu).abs().sum().item(); nv += len(y)
        v_loss /= nv; v_mae /= nv
        marker = ""
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0; marker = "★"
        else:
            bad += 1
        print(f"  ep {ep:>3}: tr {tr_loss:.4f}  val {v_loss:.4f}  mae {v_mae:.4f} {marker}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


def main():
    print("=" * 70)
    print("ResMLP + AdaLN")
    print(f"  DIM={DIM}, N_BLOCKS={N_BLOCKS}")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_data()
    train_df = df[df.datetime_kst < TRAIN_END].copy()
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)].copy()
    test_df = df[df.datetime_kst >= VAL_END].copy()
    all_sites = sorted(df["site"].unique())

    main_tr, cond_tr, y_tr = build_features(train_df, all_sites)
    main_v, cond_v, y_v = build_features(val_df, all_sites)
    main_te, cond_te, y_te = build_features(test_df, all_sites)

    mean_main = main_tr.mean(axis=0); std_main = main_tr.std(axis=0) + 1e-6
    main_tr = (main_tr - mean_main) / std_main
    main_v = (main_v - mean_main) / std_main
    main_te = (main_te - mean_main) / std_main
    cond_tr[:, 0] /= 10.0; cond_v[:, 0] /= 10.0; cond_te[:, 0] /= 10.0

    n_main = main_tr.shape[1]; n_cond = cond_tr.shape[1]
    print(f"\n[Data] train {len(y_tr):,} / val {len(y_v):,} / test {len(y_te):,}")
    print(f"[Data] n_main={n_main}, n_cond={n_cond}")

    model = ResMLPAdaLN(n_main, n_cond).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] params: {n_params/1e3:.1f}k")

    train_ds = TensorDataset(torch.from_numpy(main_tr), torch.from_numpy(cond_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(main_v), torch.from_numpy(cond_v), torch.from_numpy(y_v))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("\n[Train]")
    model = train_loop(model, train_loader, val_loader, device)

    # Test
    model.eval()
    test_ds = TensorDataset(torch.from_numpy(main_te), torch.from_numpy(cond_te), torch.from_numpy(y_te))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    mus = []; sigs = []
    with torch.no_grad():
        for xm, xc, _ in test_loader:
            mu, sig = model(xm.to(device), xc.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)

    cap = test_df["site_capacity_kw"].values
    err = np.abs(y_te - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y_te >= mu_arr - 1.28 * sig_arr) & (y_te <= mu_arr + 1.28 * sig_arr)).mean() * 100
    cov95 = ((y_te >= mu_arr - 1.96 * sig_arr) & (y_te <= mu_arr + 1.96 * sig_arr)).mean() * 100
    print(f"\n=== TEST ===")
    print(f"  NMAE: {nmae:.2f}% / cov80: {cov80:.1f}% / cov95: {cov95:.1f}%")

    # site/portfolio
    df_eval = test_df[["datetime_kst", "site", "site_capacity_kw"]].copy()
    df_eval["cf"] = y_te; df_eval["pred_cf"] = mu_arr; df_eval["pred_std_cf"] = sig_arr
    df_eval["err"] = err
    print(f"  사이트별 NMAE:")
    for s in sorted(df_eval["site"].unique()):
        sub = df_eval[df_eval["site"] == s]
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        print(f"    {s:<14} {n:>6.2f}%")

    df_eval["pred_kwh"] = mu_arr * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = y_te * df_eval["site_capacity_kw"]
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"))
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    print(f"  포트폴리오 NMAE: {pnmae:.2f}%")

    out_dir = ROOT / "pv/experiments/resmlp_adaln"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_eval.to_parquet(out_dir / "test_predictions.parquet", index=False)
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}")

    print("\n" + "=" * 70)
    print(f"  NGBoost strong: 6.07% / 5.04%")
    print(f"  MLP+FiLM:       5.88% / 4.88%")
    print(f"  GRU+FiLM:       6.05% / 4.98%")
    print(f"  ResMLP+AdaLN:   {nmae:.2f}% / {pnmae:.2f}%")


if __name__ == "__main__":
    main()
