"""Two-mode single family PV model.

Architecture:
  Shared trunk (ResMLP + AdaLN)
    └─ D-1 head     (forecast-only inputs → 24h prediction)
    └─ Intraday branch (last 6h cf history → CNN → fusion → 1h-ahead)

Training:
  Stage 1: trunk + D-1 head (forecast-only)
  Stage 2: freeze trunk, train local CNN + intraday head only

Both modes output (μ, σ) Gaussian.
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

# ===== 재현성 설정 =====
SEED = 42

def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds()

# Config
MAIN_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws"]
COND_NUM_FEATURES = ["dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos"]
LOCAL_FEATURES = ["cf", "dsr_mean", "delta_cf", "delta_dsr"]
LOCAL_HOURS = 6
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

DIM = 64
N_BLOCKS = 4
DIM_LOCAL = 16
LR = 7e-4              # batch 32 sweet spot
BATCH_SIZE = 32        # 16보다 빠르면서 SGD noise 유지
MAX_EPOCHS_S1 = 50
MAX_EPOCHS_S2 = 30
PATIENCE = 10


# ========== Shared trunk (ResMLP + AdaLN) ==========

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


class SharedTrunk(nn.Module):
    """ResMLP + AdaLN — produces h_global of shape (B, dim)."""
    def __init__(self, n_main, n_cond, dim=DIM, n_blocks=N_BLOCKS):
        super().__init__()
        self.input_proj = nn.Linear(n_main, dim)
        self.blocks = nn.ModuleList([AdaLNBlock(dim, n_cond) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.final_cond = nn.Linear(n_cond, 2 * dim)

    def forward(self, x_main, x_cond):
        x = self.input_proj(x_main)
        for blk in self.blocks:
            x = blk(x, x_cond)
        h = self.final_norm(x)
        scale, shift = self.final_cond(x_cond).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        return h


# ========== Heads ==========

class GaussianHead(nn.Module):
    """μ, σ output."""
    def __init__(self, dim_in):
        super().__init__()
        self.mu_head = nn.Linear(dim_in, 1)
        self.log_sig_head = nn.Linear(dim_in, 1)

    def forward(self, h):
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h)).squeeze(-1) + 1e-3
        return mu, sigma


class LocalCNNEncoder(nn.Module):
    """Conv1D × 2 + AdaptiveAvgPool → (B, dim_local)."""
    def __init__(self, n_local_feat, dim_local=DIM_LOCAL):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_local_feat, dim_local, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim_local, dim_local, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x_local):
        # x_local: (B, T, F) → (B, F, T) for conv1d
        return self.conv(x_local.transpose(1, 2)).squeeze(-1)


class IntradayHead(nn.Module):
    """Fusion (h_global + h_local) → μ, σ."""
    def __init__(self, dim_global, dim_local):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim_global + dim_local, dim_global), nn.GELU(),
            nn.Linear(dim_global, dim_global), nn.GELU(),
        )
        self.gaussian = GaussianHead(dim_global)

    def forward(self, h_global, h_local):
        h = torch.cat([h_global, h_local], dim=-1)
        h = self.fusion(h)
        return self.gaussian(h)


# ========== Two-mode model ==========

class TwoModePV(nn.Module):
    def __init__(self, n_main, n_cond, n_local_feat,
                 dim=DIM, n_blocks=N_BLOCKS, dim_local=DIM_LOCAL):
        super().__init__()
        self.trunk = SharedTrunk(n_main, n_cond, dim, n_blocks)
        self.d1_head = GaussianHead(dim)
        self.local_encoder = LocalCNNEncoder(n_local_feat, dim_local)
        self.intraday_head = IntradayHead(dim, dim_local)

    def forward(self, x_main, x_cond, mode, x_local=None):
        h_global = self.trunk(x_main, x_cond)
        if mode == "d1":
            return self.d1_head(h_global)
        elif mode == "intraday":
            assert x_local is not None
            h_local = self.local_encoder(x_local)
            return self.intraday_head(h_global, h_local)
        raise ValueError(mode)


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


# ========== Data ==========

def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)
    df["is_daytime"] = (df["dsr_mean"].notna() & df["cf"].notna()).astype(np.int32)
    for col in ["ta", "hm", "ws", "dc10Tca", "dsr_mean", "zenith_center"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    df["cf"] = df["cf"].fillna(0)
    df["delta_cf"] = df.groupby("site")["cf"].diff().fillna(0)
    df["delta_dsr"] = df.groupby("site")["dsr_mean"].diff().fillna(0)
    return df


def build_d1(df, all_sites, mean_g=None, std_g=None):
    """D-1 mode: daytime targets만, no local history."""
    site_idx = {s: i for i, s in enumerate(all_sites)}
    n_sites = len(all_sites)
    main = df[MAIN_FEATURES].values.astype(np.float32)
    cn = df[COND_NUM_FEATURES].values.astype(np.float32)
    cn[:, 0] /= 10.0
    soh = np.zeros((len(df), n_sites), dtype=np.float32)
    for i, s in enumerate(all_sites):
        soh[:, i] = (df["site"] == s).astype(np.float32)
    cond = np.concatenate([cn, soh], axis=1)
    y = df[TARGET].values.astype(np.float32)

    # daytime filter
    mask = df["is_daytime"].values.astype(bool)
    main = main[mask]; cond = cond[mask]; y = y[mask]
    df_meta = df[mask].reset_index(drop=True)

    if mean_g is None:
        mean_g = main.mean(axis=0); std_g = main.std(axis=0) + 1e-6
    main = (main - mean_g) / std_g
    return main, cond, y, df_meta, (mean_g, std_g)


def build_intraday(df, all_sites, scalers):
    """Intraday mode: build sequences with last 6h history, daytime targets."""
    mean_g, std_g, mean_l, std_l = scalers if len(scalers) == 4 else (*scalers, None, None)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    n_sites = len(all_sites)

    g_list, c_list, l_list, y_list, meta = [], [], [], [], []
    for site, sub in df.groupby("site"):
        sub = sub.sort_values("datetime_kst").reset_index(drop=True)
        gm = sub[MAIN_FEATURES].values.astype(np.float32)
        cn = sub[COND_NUM_FEATURES].values.astype(np.float32)
        cn[:, 0] /= 10.0
        soh = np.zeros((len(sub), n_sites), dtype=np.float32)
        soh[:, site_idx[site]] = 1.0
        cond = np.concatenate([cn, soh], axis=1)
        loc = sub[LOCAL_FEATURES].values.astype(np.float32)
        is_day = sub["is_daytime"].values

        for t in range(LOCAL_HOURS, len(sub)):
            if not is_day[t]:
                continue
            g_list.append(gm[t])
            c_list.append(cond[t])
            l_list.append(loc[t - LOCAL_HOURS:t])
            y_list.append(sub["cf"].iloc[t])
            meta.append({"datetime_kst": sub["datetime_kst"].iloc[t],
                         "site": site,
                         "site_capacity_kw": sub["site_capacity_kw"].iloc[t]})

    g = np.stack(g_list); c = np.stack(c_list); l = np.stack(l_list)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta)

    g = (g - mean_g) / std_g
    if mean_l is None:
        mean_l = l.reshape(-1, l.shape[-1]).mean(axis=0)
        std_l = l.reshape(-1, l.shape[-1]).std(axis=0) + 1e-6
    l = (l - mean_l) / std_l

    return g, c, l, y, meta_df, (mean_g, std_g, mean_l, std_l)


# ========== Stage 1: D-1 ==========

def stage1_train(model, train_loader, val_loader, device):
    print("\n[Stage 1] D-1 mode 학습 (trunk + d1_head)")
    # Optimize trunk + d1_head only
    params = list(model.trunk.parameters()) + list(model.d1_head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS_S1)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS_S1):
        model.train()
        tr = 0; n = 0
        for xm, xc, y in train_loader:
            xm, xc, y = xm.to(device), xc.to(device), y.to(device)
            mu, sig = model(xm, xc, mode="d1")
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr += loss.item() * len(y); n += len(y)
        tr /= n; scheduler.step()
        model.eval()
        v = 0; vmae = 0; nv = 0
        with torch.no_grad():
            for xm, xc, y in val_loader:
                xm, xc, y = xm.to(device), xc.to(device), y.to(device)
                mu, sig = model(xm, xc, mode="d1")
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
        if ep % 10 == 0:
            print(f"  ep {ep:>3}: tr {tr:.4f} val {v:.4f} mae {vmae:.4f} {m}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


# ========== Stage 2: intraday ==========

def stage2_train(model, train_loader, val_loader, device):
    print("\n[Stage 2] Intraday mode 학습 (local_encoder + intraday_head, trunk frozen)")
    # Freeze trunk + d1 head
    for p in model.trunk.parameters():
        p.requires_grad = False
    for p in model.d1_head.parameters():
        p.requires_grad = False
    # Train only local + intraday head
    params = list(model.local_encoder.parameters()) + list(model.intraday_head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS_S2)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS_S2):
        model.train()
        # Trunk in eval mode (frozen, no dropout)
        model.trunk.eval()
        model.d1_head.eval()
        tr = 0; n = 0
        for xm, xc, xl, y in train_loader:
            xm, xc, xl, y = xm.to(device), xc.to(device), xl.to(device), y.to(device)
            mu, sig = model(xm, xc, mode="intraday", x_local=xl)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr += loss.item() * len(y); n += len(y)
        tr /= n; scheduler.step()
        model.eval()
        v = 0; vmae = 0; nv = 0
        with torch.no_grad():
            for xm, xc, xl, y in val_loader:
                xm, xc, xl, y = xm.to(device), xc.to(device), xl.to(device), y.to(device)
                mu, sig = model(xm, xc, mode="intraday", x_local=xl)
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
        if ep % 10 == 0:
            print(f"  ep {ep:>3}: tr {tr:.4f} val {v:.4f} mae {vmae:.4f} {m}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


# ========== Eval ==========

def evaluate_d1(model, main, cond, y, meta, device, label):
    model.eval()
    test_ds = TensorDataset(torch.from_numpy(main), torch.from_numpy(cond), torch.from_numpy(y))
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    mus, sigs = [], []
    with torch.no_grad():
        for xm, xc, _ in loader:
            mu, sig = model(xm.to(device), xc.to(device), mode="d1")
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)
    cap = meta["site_capacity_kw"].values
    err = np.abs(y - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu_arr - 1.282 * sig_arr) & (y <= mu_arr + 1.282 * sig_arr)).mean() * 100
    print(f"\n=== {label} (D-1 mode) ===")
    print(f"  rows: {len(y):,}, NMAE: {nmae:.2f}%, cov80: {cov80:.1f}%")
    # Per site
    df_eval = meta.copy(); df_eval["cf"] = y; df_eval["mu"] = mu_arr; df_eval["sig"] = sig_arr; df_eval["err"] = err
    print(f"  사이트별 NMAE:")
    for s in sorted(df_eval["site"].unique()):
        sub = df_eval[df_eval["site"] == s]
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        print(f"    {s:<14} {n:>6.2f}%")
    df_eval["pred_kwh"] = mu_arr * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = y * df_eval["site_capacity_kw"]
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"))
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    print(f"  포트폴리오 NMAE: {pnmae:.2f}%")
    return df_eval, nmae, pnmae, cov80


def evaluate_intraday(model, g, c, l, y, meta, device, label):
    model.eval()
    test_ds = TensorDataset(torch.from_numpy(g), torch.from_numpy(c), torch.from_numpy(l), torch.from_numpy(y))
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    mus, sigs = [], []
    with torch.no_grad():
        for xm, xc, xl, _ in loader:
            mu, sig = model(xm.to(device), xc.to(device), mode="intraday", x_local=xl.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)
    cap = meta["site_capacity_kw"].values
    err = np.abs(y - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu_arr - 1.282 * sig_arr) & (y <= mu_arr + 1.282 * sig_arr)).mean() * 100
    print(f"\n=== {label} (Intraday mode) ===")
    print(f"  rows: {len(y):,}, NMAE: {nmae:.2f}%, cov80: {cov80:.1f}%")
    print(f"  사이트별 NMAE:")
    df_eval = meta.copy(); df_eval["cf"] = y; df_eval["mu"] = mu_arr; df_eval["sig"] = sig_arr; df_eval["err"] = err
    for s in sorted(df_eval["site"].unique()):
        sub = df_eval[df_eval["site"] == s]
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        print(f"    {s:<14} {n:>6.2f}%")
    df_eval["pred_kwh"] = mu_arr * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = y * df_eval["site_capacity_kw"]
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"))
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    print(f"  포트폴리오 NMAE: {pnmae:.2f}%")
    return df_eval, nmae, pnmae, cov80


def main():
    print("=" * 70)
    print("Two-mode Single Family PV Model")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_data()
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]
    all_sites = sorted(df["site"].unique())
    n_sites = len(all_sites)
    print(f"\n[Data] train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,}")

    # Build D-1 datasets
    print("\n[Build] D-1 mode datasets...")
    main_tr, cond_tr, y_tr, meta_tr, scalers_d1 = build_d1(train_df, all_sites)
    main_v, cond_v, y_v, meta_v, _ = build_d1(val_df, all_sites, *scalers_d1)
    main_te, cond_te, y_te, meta_te, _ = build_d1(test_df, all_sites, *scalers_d1)
    print(f"  D-1 train {len(y_tr):,}, val {len(y_v):,}, test {len(y_te):,}")

    n_main = main_tr.shape[1]; n_cond = cond_tr.shape[1]
    print(f"  n_main={n_main}, n_cond={n_cond}")

    # Build intraday datasets (use stage 1 scalers for global, stage 2 fits local)
    print("\n[Build] Intraday datasets (later for stage 2)...")
    g_tr, c_tr, l_tr, y_i_tr, meta_i_tr, scalers_intra = build_intraday(
        train_df, all_sites, scalers_d1)
    g_v, c_v, l_v, y_i_v, meta_i_v, _ = build_intraday(val_df, all_sites, scalers_intra)
    g_te, c_te, l_te, y_i_te, meta_i_te, _ = build_intraday(test_df, all_sites, scalers_intra)
    print(f"  intraday train {len(y_i_tr):,}, val {len(y_i_v):,}, test {len(y_i_te):,}")

    n_local = l_tr.shape[2]

    # ===== Build model =====
    model = TwoModePV(n_main, n_cond, n_local).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] Total params: {n_params/1e3:.1f}k")
    print(f"  trunk: {sum(p.numel() for p in model.trunk.parameters())/1e3:.1f}k")
    print(f"  d1_head: {sum(p.numel() for p in model.d1_head.parameters())/1e3:.1f}k")
    print(f"  local_encoder: {sum(p.numel() for p in model.local_encoder.parameters())/1e3:.1f}k")
    print(f"  intraday_head: {sum(p.numel() for p in model.intraday_head.parameters())/1e3:.1f}k")

    # Stage 1: D-1
    train_d1_ds = TensorDataset(torch.from_numpy(main_tr), torch.from_numpy(cond_tr), torch.from_numpy(y_tr))
    val_d1_ds = TensorDataset(torch.from_numpy(main_v), torch.from_numpy(cond_v), torch.from_numpy(y_v))
    g_d1 = torch.Generator(); g_d1.manual_seed(SEED)
    train_d1_loader = DataLoader(train_d1_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_d1)
    val_d1_loader = DataLoader(val_d1_ds, batch_size=BATCH_SIZE)
    model = stage1_train(model, train_d1_loader, val_d1_loader, device)
    df_d1, nmae_d1, pnmae_d1, cov_d1 = evaluate_d1(
        model, main_te, cond_te, y_te, meta_te, device, "Stage 1 — Test")

    # Stage 2: Intraday
    train_i_ds = TensorDataset(torch.from_numpy(g_tr), torch.from_numpy(c_tr),
                               torch.from_numpy(l_tr), torch.from_numpy(y_i_tr))
    val_i_ds = TensorDataset(torch.from_numpy(g_v), torch.from_numpy(c_v),
                             torch.from_numpy(l_v), torch.from_numpy(y_i_v))
    g_i = torch.Generator(); g_i.manual_seed(SEED + 1)
    train_i_loader = DataLoader(train_i_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_i)
    val_i_loader = DataLoader(val_i_ds, batch_size=BATCH_SIZE)
    model = stage2_train(model, train_i_loader, val_i_loader, device)
    df_i, nmae_i, pnmae_i, cov_i = evaluate_intraday(
        model, g_te, c_te, l_te, y_i_te, meta_i_te, device, "Stage 2 — Test")

    # Save
    out_dir = ROOT / "pv/experiments/two_mode_pv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_d1.rename(columns={"mu": "pred_cf", "sig": "pred_std_cf"})[
        ["datetime_kst", "site", "site_capacity_kw", "cf", "pred_cf", "pred_std_cf"]
    ].to_parquet(out_dir / "test_predictions_d1.parquet", index=False)
    df_i.rename(columns={"mu": "pred_cf", "sig": "pred_std_cf"})[
        ["datetime_kst", "site", "site_capacity_kw", "cf", "pred_cf", "pred_std_cf"]
    ].to_parquet(out_dir / "test_predictions_intraday.parquet", index=False)
    torch.save({
        "state_dict": model.state_dict(),
        "scalers_d1": scalers_d1,
        "scalers_intra": scalers_intra,
        "all_sites": all_sites,
    }, out_dir / "model.pt")
    print(f"\n저장: {out_dir}")

    # Final comparison vs separate models
    print("\n" + "=" * 70)
    print("최종 비교 — Unified Two-mode vs Separate models")
    print("=" * 70)
    print(f"  {'Mode':<25} {'Site NMAE':>10} {'Port NMAE':>10} {'Cov80':>7}")
    print("-" * 60)
    print(f"  {'Separate ResMLP+AdaLN':<25} {'5.12%':>10} {'4.13%':>10} {'82.6%':>7}")
    print(f"  {'Separate Hybrid':<25} {'4.27%':>10} {'3.74%':>10} {'79.6%':>7}")
    print("-" * 60)
    print(f"  {'Two-mode D-1 (Stage 1)':<25} {f'{nmae_d1:.2f}%':>10} {f'{pnmae_d1:.2f}%':>10} {f'{cov_d1:.1f}%':>7}")
    print(f"  {'Two-mode Intraday (S2)':<25} {f'{nmae_i:.2f}%':>10} {f'{pnmae_i:.2f}%':>10} {f'{cov_i:.1f}%':>7}")
    print("-" * 60)


if __name__ == "__main__":
    main()
