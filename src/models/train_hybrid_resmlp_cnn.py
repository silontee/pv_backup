"""Hybrid: ResMLP+AdaLN (global) + 1D-CNN (local 3~6h history) + fusion.

목표: GRU의 long-range memory 부담 없이, *짧은 시간 의존성*만 추가 학습.

Architecture:
  Global branch (현재 시점 + 조건):
    main features (5)        → Linear → x_g
    cond features (n_cond)   → AdaLN modulation
    ResMLP block × 4         → h_g (dim=64)

  Local branch (최근 6h history):
    [cf, dsr_mean, delta_cf, delta_dsr] × 6시간   ← 짧은 momentum
    Conv1D (k=3) × 2 + pooling                    → h_l (16)

  Fusion:
    concat([h_g, h_l]) → MLP(dim) → output heads

  Head: μ, σ (Gaussian NLL)

vs ResMLP+AdaLN 차이: local CNN branch 추가
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

# Global features (current step)
MAIN_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws"]
COND_NUM_FEATURES = ["dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos"]

# Local features (history)
LOCAL_FEATURES = ["cf", "dsr_mean", "delta_cf", "delta_dsr"]
LOCAL_HOURS = 6   # 최근 6h history

TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

DIM_GLOBAL = 64
N_BLOCKS = 4
DIM_LOCAL = 16
LR = 1e-3
BATCH_SIZE = 4096
MAX_EPOCHS = 100
PATIENCE = 10


def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)
    # is_daytime mark BEFORE fillna
    df["is_daytime"] = (df["dsr_mean"].notna() & df["cf"].notna()).astype(np.int32)
    # fillna for input
    for col in ["ta", "hm", "ws", "dc10Tca", "dsr_mean", "zenith_center"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    df["cf"] = df["cf"].fillna(0)
    # delta features
    df["delta_cf"] = df.groupby("site")["cf"].diff().fillna(0)
    df["delta_dsr"] = df.groupby("site")["dsr_mean"].diff().fillna(0)
    return df


def build(df, all_sites, mean_g=None, std_g=None, mean_l=None, std_l=None):
    """Build global / cond / local sequences + targets."""
    site_idx = {s: i for i, s in enumerate(all_sites)}
    n_sites = len(all_sites)

    globals_list = []
    conds_list = []
    locals_list = []
    targets = []
    meta_rows = []

    for site, sub in df.groupby("site"):
        sub = sub.sort_values("datetime_kst").reset_index(drop=True)
        gm = sub[MAIN_FEATURES].values.astype(np.float32)
        cn = sub[COND_NUM_FEATURES].values.astype(np.float32)
        cn[:, 0] = cn[:, 0] / 10.0  # cloud → [0,1]
        soh = np.zeros((len(sub), n_sites), dtype=np.float32)
        soh[:, site_idx[site]] = 1.0
        cond = np.concatenate([cn, soh], axis=1)
        loc = sub[LOCAL_FEATURES].values.astype(np.float32)
        is_day = sub["is_daytime"].values

        for t in range(LOCAL_HOURS, len(sub)):
            if not is_day[t]:
                continue
            globals_list.append(gm[t])
            conds_list.append(cond[t])
            locals_list.append(loc[t - LOCAL_HOURS:t])
            targets.append(sub["cf"].iloc[t])
            meta_rows.append({
                "datetime_kst": sub["datetime_kst"].iloc[t],
                "site": site,
                "site_capacity_kw": sub["site_capacity_kw"].iloc[t],
            })

    g = np.stack(globals_list)
    c = np.stack(conds_list)
    l = np.stack(locals_list)  # (N, LOCAL_HOURS, n_local_features)
    y = np.array(targets, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)

    # standardize (computed on train only)
    if mean_g is None:
        mean_g = g.mean(axis=0); std_g = g.std(axis=0) + 1e-6
        mean_l = l.reshape(-1, l.shape[-1]).mean(axis=0)
        std_l = l.reshape(-1, l.shape[-1]).std(axis=0) + 1e-6
    g = (g - mean_g) / std_g
    l = (l - mean_l) / std_l

    return g, c, l, y, meta, (mean_g, std_g, mean_l, std_l)


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
        h = self.mlp(h)
        return x + h


class HybridResMLPCNN(nn.Module):
    def __init__(self, n_main, n_cond, n_local_feat,
                 dim_g=DIM_GLOBAL, n_blocks=N_BLOCKS, dim_l=DIM_LOCAL):
        super().__init__()
        # Global branch
        self.global_input = nn.Linear(n_main, dim_g)
        self.global_blocks = nn.ModuleList([AdaLNBlock(dim_g, n_cond) for _ in range(n_blocks)])
        self.global_final_norm = nn.LayerNorm(dim_g, elementwise_affine=False)
        self.global_final_cond = nn.Linear(n_cond, 2 * dim_g)
        # Local branch — 1D CNN
        self.local_conv = nn.Sequential(
            nn.Conv1d(n_local_feat, dim_l, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim_l, dim_l, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim_g + dim_l, dim_g), nn.GELU(),
            nn.Linear(dim_g, dim_g), nn.GELU(),
        )
        # Heads
        self.mu_head = nn.Linear(dim_g, 1)
        self.log_sig_head = nn.Linear(dim_g, 1)

    def forward(self, x_global, x_cond, x_local):
        h_g = self.global_input(x_global)
        for blk in self.global_blocks:
            h_g = blk(h_g, x_cond)
        h_g = self.global_final_norm(h_g)
        scale, shift = self.global_final_cond(x_cond).chunk(2, dim=-1)
        h_g = h_g * (1 + scale) + shift

        # x_local: (B, T, F) → (B, F, T) for conv1d
        h_l = self.local_conv(x_local.transpose(1, 2)).squeeze(-1)

        h = torch.cat([h_g, h_l], dim=-1)
        h = self.fusion(h)
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h)).squeeze(-1) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


def train_loop(model, train_loader, val_loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train(); tr = 0; n = 0
        for xg, xc, xl, y in train_loader:
            xg, xc, xl, y = xg.to(device), xc.to(device), xl.to(device), y.to(device)
            mu, sig = model(xg, xc, xl)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(y); n += len(y)
        tr /= n; scheduler.step()
        model.eval(); v = 0; vm = 0; nv = 0
        with torch.no_grad():
            for xg, xc, xl, y in val_loader:
                xg, xc, xl, y = xg.to(device), xc.to(device), xl.to(device), y.to(device)
                mu, sig = model(xg, xc, xl)
                v += gaussian_nll(y, mu, sig).mean().item() * len(y)
                vm += (y - mu).abs().sum().item(); nv += len(y)
        v /= nv; vm /= nv
        m = ""
        if v < best_val:
            best_val = v
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            bad = 0; m = "★"
        else:
            bad += 1
        print(f"  ep {ep:>3}: tr {tr:.4f}  val {v:.4f}  mae {vm:.4f} {m}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


def main():
    print("=" * 70)
    print(f"Hybrid: ResMLP+AdaLN (global) + 1D-CNN (local {LOCAL_HOURS}h)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_data()
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]
    all_sites = sorted(df["site"].unique())

    print("\n[Build] features...")
    g_tr, c_tr, l_tr, y_tr, meta_tr, scalers = build(train_df, all_sites)
    g_v, c_v, l_v, y_v, meta_v, _ = build(val_df, all_sites, *scalers)
    g_te, c_te, l_te, y_te, meta_te, _ = build(test_df, all_sites, *scalers)
    print(f"  train {len(y_tr):,} / val {len(y_v):,} / test {len(y_te):,}")
    print(f"  global ({g_tr.shape[1]}): {MAIN_FEATURES}")
    print(f"  cond ({c_tr.shape[1]}): {COND_NUM_FEATURES} + 8 site")
    print(f"  local ({LOCAL_HOURS}h × {l_tr.shape[2]}): {LOCAL_FEATURES}")

    n_main = g_tr.shape[1]; n_cond = c_tr.shape[1]; n_local = l_tr.shape[2]
    model = HybridResMLPCNN(n_main, n_cond, n_local).to(device)
    print(f"\n[Model] params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")

    train_ds = TensorDataset(torch.from_numpy(g_tr), torch.from_numpy(c_tr), torch.from_numpy(l_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(g_v), torch.from_numpy(c_v), torch.from_numpy(l_v), torch.from_numpy(y_v))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("\n[Train]")
    model = train_loop(model, train_loader, val_loader, device)

    # Test
    model.eval()
    test_ds = TensorDataset(torch.from_numpy(g_te), torch.from_numpy(c_te), torch.from_numpy(l_te), torch.from_numpy(y_te))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    mus = []; sigs = []
    with torch.no_grad():
        for xg, xc, xl, _ in test_loader:
            mu, sig = model(xg.to(device), xc.to(device), xl.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)

    cap = meta_te["site_capacity_kw"].values
    err = np.abs(y_te - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y_te >= mu_arr - 1.28 * sig_arr) & (y_te <= mu_arr + 1.28 * sig_arr)).mean() * 100
    cov95 = ((y_te >= mu_arr - 1.96 * sig_arr) & (y_te <= mu_arr + 1.96 * sig_arr)).mean() * 100
    print(f"\n=== TEST ===")
    print(f"  NMAE: {nmae:.2f}% / cov80: {cov80:.1f}% / cov95: {cov95:.1f}%")

    df_eval = meta_te.copy()
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

    out_dir = ROOT / "pv/experiments/hybrid_resmlp_cnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_eval.to_parquet(out_dir / "test_predictions.parquet", index=False)
    torch.save({
        "state_dict": model.state_dict(),
        "scalers": scalers,
        "all_sites": all_sites,
    }, out_dir / "model.pt")
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}")

    print("\n" + "=" * 70)
    print("최종 비교")
    print("=" * 70)
    print(f"  NGBoost strong:    6.07% / 5.04%")
    print(f"  MLP+FiLM:          5.88% / 4.88%")
    print(f"  GRU+FiLM:          6.05% / 4.98%")
    print(f"  FT-Transformer:    5.49% / 4.48%")
    print(f"  ResMLP+AdaLN:      5.12% / 4.13%")
    print(f"  Hybrid:            {nmae:.2f}% / {pnmae:.2f}%")


if __name__ == "__main__":
    main()
