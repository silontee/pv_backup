"""ResMLP+AdaLN + shallow TCN hybrid.

Architecture:
  Branch 1 (Static): ResMLP+AdaLN backbone (현 winner와 동일)
    Input: x_main (5), x_cond (13)
    → h_main (B, dim=64)

  Branch 2 (Temporal): shallow 1D TCN
    Input: 12h window of weather features (centered: t-6 ~ t+5)
            *perfect-foresight라 ±side both 사용*
    Conv1d × 2 (dilation 1, 2) + AdaptiveAvgPool
    → h_temp (B, 32)

  Fusion: concat([h_main, h_temp]) → MLP(80→64→64) → mu_head, sigma_head

목표: ResMLP+AdaLN baseline (5.12% / 4.13% / 82.6%) 대비 개선 측정
"""
import os
import random
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

# Static features (current step) — same as ResMLP+AdaLN
MAIN_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws"]
COND_NUM_FEATURES = ["dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos"]

# Temporal features for TCN (sequence)
TEMPORAL_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws", "dc10Tca"]
WINDOW = 12              # 12h centered window (t-6 to t+5)
WINDOW_OFFSET_LEFT = 6   # 6 hours past
# perfect-foresight: future side OK

TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

DIM = 64
N_BLOCKS = 4
DIM_TEMPORAL = 32
TCN_KERNEL = 3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4096
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 42


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== Model ==========

class AdaLNBlock(nn.Module):
    def __init__(self, dim, n_cond):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_to_ss = nn.Linear(n_cond, 2 * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.cond_to_ss(cond).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        return x + self.mlp(h)


class ResMLP_AdaLN_TCN(nn.Module):
    def __init__(self, n_main, n_cond, n_temporal_feat,
                 dim=DIM, n_blocks=N_BLOCKS, dim_temporal=DIM_TEMPORAL):
        super().__init__()
        # ===== Static branch (ResMLP + AdaLN) =====
        self.input_proj = nn.Linear(n_main, dim)
        self.blocks = nn.ModuleList([AdaLNBlock(dim, n_cond) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_final = nn.Linear(n_cond, 2 * dim)

        # ===== Temporal branch (shallow TCN) =====
        # Conv1d with increasing dilation (1, 2) for receptive field 5
        self.tcn = nn.Sequential(
            nn.Conv1d(n_temporal_feat, dim_temporal, kernel_size=TCN_KERNEL, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(dim_temporal, dim_temporal, kernel_size=TCN_KERNEL, padding=2, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ===== Fusion =====
        self.fusion = nn.Sequential(
            nn.Linear(dim + dim_temporal, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )

        # ===== Heads =====
        self.mu_head = nn.Linear(dim, 1)
        self.log_sig_head = nn.Linear(dim, 1)

    def forward(self, x_main, x_cond, x_temporal):
        # Static branch
        x = self.input_proj(x_main)
        for blk in self.blocks:
            x = blk(x, x_cond)
        h = self.final_norm(x)
        scale, shift = self.cond_final(x_cond).chunk(2, dim=-1)
        h_static = h * (1 + scale) + shift

        # Temporal branch — (B, T, F) → (B, F, T) for Conv1d
        h_temp = self.tcn(x_temporal.transpose(1, 2)).squeeze(-1)

        # Fusion
        h_fused = self.fusion(torch.cat([h_static, h_temp], dim=-1))

        mu = self.mu_head(h_fused).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h_fused)).squeeze(-1) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


# ========== Data ==========

def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)
    df["is_daytime"] = (df["dsr_mean"].notna() & df["cf"].notna()).astype(np.int32)
    for col in ["ta", "hm", "ws", "dc10Tca", "dsr_mean", "zenith_center"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["cf"] = df["cf"].fillna(0)
    return df


def build(full_df, all_sites, target_start, target_end, scalers=None):
    """Build all 3 inputs:
       x_main (current step), x_cond (current step), x_temporal (centered window)

    Use full_df for temporal history/future, filter target time only.
    """
    site_idx = {s: i for i, s in enumerate(all_sites)}
    n_sites = len(all_sites)
    if scalers is not None:
        mean_m, std_m, mean_t, std_t = scalers
    else:
        mean_m = std_m = mean_t = std_t = None

    main_list, cond_list, temporal_list, y_list, meta = [], [], [], [], []
    for site, sub in full_df.groupby("site"):
        sub = sub.sort_values("datetime_kst").reset_index(drop=True)
        m_arr = sub[MAIN_FEATURES].values.astype(np.float32)
        cn = sub[COND_NUM_FEATURES].values.astype(np.float32)
        cn[:, 0] /= 10.0
        soh = np.zeros((len(sub), n_sites), dtype=np.float32)
        soh[:, site_idx[site]] = 1.0
        cond = np.concatenate([cn, soh], axis=1)
        t_arr = sub[TEMPORAL_FEATURES].values.astype(np.float32)
        is_day = sub["is_daytime"].values

        right_offset = WINDOW - WINDOW_OFFSET_LEFT  # 6 future hours

        for t in range(WINDOW_OFFSET_LEFT, len(sub) - right_offset):
            if not is_day[t]:
                continue
            target_dt = sub["datetime_kst"].iloc[t]
            if target_dt < target_start or target_dt >= target_end:
                continue
            main_list.append(m_arr[t])
            cond_list.append(cond[t])
            # centered window: [t-6, t-5, ..., t-1, t, t+1, ..., t+5]
            temporal_list.append(t_arr[t - WINDOW_OFFSET_LEFT:t + right_offset])
            y_list.append(sub["cf"].iloc[t])
            meta.append({"datetime_kst": target_dt, "site": site,
                         "site_capacity_kw": float(sub["site_capacity_kw"].iloc[t])})

    X_main = np.stack(main_list)
    X_cond = np.stack(cond_list)
    X_temp = np.stack(temporal_list)   # (N, 12, n_temporal_feat)
    Y = np.array(y_list, dtype=np.float32)
    M = pd.DataFrame(meta)

    if mean_m is None:
        mean_m = X_main.mean(axis=0)
        std_m = X_main.std(axis=0) + 1e-6
        mean_t = X_temp.reshape(-1, X_temp.shape[-1]).mean(axis=0)
        std_t = X_temp.reshape(-1, X_temp.shape[-1]).std(axis=0) + 1e-6
    X_main = (X_main - mean_m) / std_m
    X_temp = (X_temp - mean_t) / std_t
    return X_main, X_cond, X_temp, Y, M, (mean_m, std_m, mean_t, std_t)


# ========== Train ==========

def train_loop(model, train_loader, val_loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        tr = 0; n = 0
        for xm, xc, xt, y in train_loader:
            xm, xc, xt, y = xm.to(device), xc.to(device), xt.to(device), y.to(device)
            mu, sig = model(xm, xc, xt)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(y); n += len(y)
        tr /= n; sch.step()

        model.eval()
        v = 0; vmae = 0; nv = 0
        with torch.no_grad():
            for xm, xc, xt, y in val_loader:
                xm, xc, xt, y = xm.to(device), xc.to(device), xt.to(device), y.to(device)
                mu, sig = model(xm, xc, xt)
                v += gaussian_nll(y, mu, sig).mean().item() * len(y)
                vmae += (y - mu).abs().sum().item(); nv += len(y)
        v /= nv; vmae /= nv

        m = ""
        if v < best:
            best = v
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            bad = 0; m = "★"
        else:
            bad += 1
        if ep % 5 == 0 or m:
            print(f"  ep {ep:>3}: tr {tr:.4f} val {v:.4f} mae {vmae:.4f} {m}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


def evaluate(model, X_main, X_cond, X_temp, Y, meta, device, label):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_main), torch.from_numpy(X_cond),
                       torch.from_numpy(X_temp), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2)
    mus, sigs = [], []
    with torch.no_grad():
        for xm, xc, xt, _ in loader:
            mu, sig = model(xm.to(device), xc.to(device), xt.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)
    cap = meta["site_capacity_kw"].values
    err = np.abs(Y - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((Y >= mu_arr - 1.282 * sig_arr) & (Y <= mu_arr + 1.282 * sig_arr)).mean() * 100
    cov95 = ((Y >= mu_arr - 1.96 * sig_arr) & (Y <= mu_arr + 1.96 * sig_arr)).mean() * 100
    nll = (0.5 * np.log(2 * np.pi * sig_arr ** 2) + (Y - mu_arr) ** 2 / (2 * sig_arr ** 2)).mean()

    df_eval = meta.copy(); df_eval["mu"] = mu_arr; df_eval["sig"] = sig_arr; df_eval["err"] = err
    df_eval["pred_kwh"] = mu_arr * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = Y * df_eval["site_capacity_kw"]
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"))
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100

    print(f"\n=== {label} ===", flush=True)
    print(f"  rows {len(Y):,}  NMAE {nmae:.3f}%  Port {pnmae:.3f}%  Cov80 {cov80:.1f}%  Cov95 {cov95:.1f}%  NLL {nll:.4f}", flush=True)
    return df_eval, {"nmae": nmae, "pnmae": pnmae, "cov80": cov80, "cov95": cov95, "nll": float(nll)}


def main():
    print("=" * 70, flush=True)
    print("ResMLP+AdaLN + Shallow TCN Hybrid", flush=True)
    print("=" * 70, flush=True)
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = load_data()
    all_sites = sorted(df["site"].unique())

    print(f"\n[Build] sequences (window={WINDOW} centered, t-{WINDOW_OFFSET_LEFT}~t+{WINDOW-WINDOW_OFFSET_LEFT-1})...", flush=True)
    X_m_tr, X_c_tr, X_t_tr, Y_tr, M_tr, scalers = build(
        df, all_sites, df.datetime_kst.min(), TRAIN_END)
    X_m_v, X_c_v, X_t_v, Y_v, M_v, _ = build(df, all_sites, TRAIN_END, VAL_END, scalers=scalers)
    X_m_te, X_c_te, X_t_te, Y_te, M_te, _ = build(
        df, all_sites, VAL_END, df.datetime_kst.max() + pd.Timedelta(hours=1), scalers=scalers)
    print(f"  train {len(Y_tr):,}, val {len(Y_v):,}, test {len(Y_te):,}", flush=True)
    print(f"  static n_main={X_m_tr.shape[1]}, n_cond={X_c_tr.shape[1]}", flush=True)
    print(f"  temporal shape: {X_t_tr.shape} (n_features={X_t_tr.shape[2]})", flush=True)

    n_main = X_m_tr.shape[1]; n_cond = X_c_tr.shape[1]; n_temp = X_t_tr.shape[2]
    model = ResMLP_AdaLN_TCN(n_main, n_cond, n_temp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] params: {n_params/1e3:.1f}k", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_m_tr), torch.from_numpy(X_c_tr),
                             torch.from_numpy(X_t_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_m_v), torch.from_numpy(X_c_v),
                           torch.from_numpy(X_t_v), torch.from_numpy(Y_v))
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("\n[Train]", flush=True)
    model = train_loop(model, train_loader, val_loader, device)

    df_te, m = evaluate(model, X_m_te, X_c_te, X_t_te, Y_te, M_te, device, "TEST")

    out_dir = ROOT / "pv/experiments/resmlp_tcn"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_te.rename(columns={"mu": "pred_cf", "sig": "pred_std_cf"})[
        ["datetime_kst", "site", "site_capacity_kw"]
    ].assign(cf=Y_te, pred_cf=df_te["mu"].values, pred_std_cf=df_te["sig"].values).to_parquet(
        out_dir / "test_predictions.parquet", index=False)
    torch.save({"state_dict": model.state_dict(), "scalers": scalers, "metrics": m},
               out_dir / "model.pt")

    print("\n" + "=" * 70, flush=True)
    print("ResMLP+AdaLN baseline vs ResMLP+AdaLN+TCN hybrid", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Model':<28} {'NMAE %':>8} {'Port %':>8} {'Cov80':>7} {'Cov95':>7} {'NLL':>8}", flush=True)
    print("-" * 75, flush=True)
    print(f"  {'ResMLP+AdaLN baseline':<28} {5.12:>7.2f}% {4.13:>7.2f}% {82.6:>6.1f}% {93.2:>6.1f}% {' ?':>8}", flush=True)
    print(f"  {'ResMLP+AdaLN + TCN':<28} {m['nmae']:>7.2f}% {m['pnmae']:>7.2f}% {m['cov80']:>6.1f}% {m['cov95']:>6.1f}% {m['nll']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
