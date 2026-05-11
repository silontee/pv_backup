"""FT-Transformer + site conditioning — Track A 정교화.

각 feature를 *token*으로 처리, self-attention으로 feature interaction 학습.
+ site embedding으로 conditioning.

Architecture:
  per-feature linear projection: each scalar → d_model vector
  feature embedding (learnable, like positional)
  CLS token + site embedding
  → Transformer encoder × N
  → CLS output → (mu, sigma) heads

vs MLP+FiLM:
  - feature interaction을 *attention*으로 (MLP는 dense linear)
  - cloud, dsr, hm 등 cross interaction 더 유연
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

# 모든 feature를 attention의 token으로 — main + cond_num 합침 (site는 별도 emb)
ALL_NUM_FEATURES = [
    "dsr_mean", "zenith_center", "ta", "hm", "ws",
    "dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos",
]
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

D_MODEL = 32
N_HEADS = 4
N_LAYERS = 3
LR = 1e-3
BATCH_SIZE = 4096
MAX_EPOCHS = 80
PATIENCE = 8


def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    return df


def build_features(df, all_sites):
    feats = df[ALL_NUM_FEATURES].values.astype(np.float32)
    site_idx_map = {s: i for i, s in enumerate(all_sites)}
    site_idx = df["site"].map(site_idx_map).values.astype(np.int64)
    y = df[TARGET].values.astype(np.float32)
    return feats, site_idx, y


class FTTransformer(nn.Module):
    def __init__(self, n_features, n_sites, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        # per-feature scalar → d_model
        self.feature_proj = nn.Linear(1, d_model)
        # learnable feature embedding (서로 다른 feature 구분)
        self.feature_emb = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Site embedding (added to CLS)
        self.site_emb = nn.Embedding(n_sites, d_model)
        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        # Output heads
        self.norm = nn.LayerNorm(d_model)
        self.mu_head = nn.Linear(d_model, 1)
        self.log_sig_head = nn.Linear(d_model, 1)

    def forward(self, x_features, site_idx):
        B, F = x_features.shape
        # Project: (B, F, 1) → (B, F, d_model)
        h = self.feature_proj(x_features.unsqueeze(-1))
        h = h + self.feature_emb.unsqueeze(0)  # add feature emb
        # CLS with site
        cls = self.cls_token.expand(B, -1, -1) + self.site_emb(site_idx).unsqueeze(1)
        # Concat
        h = torch.cat([cls, h], dim=1)  # (B, F+1, d_model)
        # Transformer
        h = self.transformer(h)
        # CLS for prediction
        cls_out = self.norm(h[:, 0])
        mu = self.mu_head(cls_out).squeeze(-1)
        sigma = nn.functional.softplus(self.log_sig_head(cls_out)).squeeze(-1) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


def train_loop(model, train_loader, val_loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train(); tr = 0; n = 0
        for xf, si, y in train_loader:
            xf, si, y = xf.to(device), si.to(device), y.to(device)
            mu, sig = model(xf, si)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(y); n += len(y)
        tr /= n; scheduler.step()
        model.eval(); v = 0; vm = 0; nv = 0
        with torch.no_grad():
            for xf, si, y in val_loader:
                xf, si, y = xf.to(device), si.to(device), y.to(device)
                mu, sig = model(xf, si)
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
    print(f"FT-Transformer (d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS})")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_data()
    train_df = df[df.datetime_kst < TRAIN_END].copy()
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)].copy()
    test_df = df[df.datetime_kst >= VAL_END].copy()
    all_sites = sorted(df["site"].unique())

    f_tr, s_tr, y_tr = build_features(train_df, all_sites)
    f_v, s_v, y_v = build_features(val_df, all_sites)
    f_te, s_te, y_te = build_features(test_df, all_sites)

    mean_f = f_tr.mean(axis=0); std_f = f_tr.std(axis=0) + 1e-6
    f_tr = (f_tr - mean_f) / std_f
    f_v = (f_v - mean_f) / std_f
    f_te = (f_te - mean_f) / std_f

    n_features = f_tr.shape[1]
    print(f"[Data] train {len(y_tr):,} / val {len(y_v):,} / test {len(y_te):,}")
    print(f"[Data] features ({n_features}): {ALL_NUM_FEATURES} + site_emb")

    model = FTTransformer(n_features, len(all_sites)).to(device)
    print(f"[Model] params: {sum(p.numel() for p in model.parameters())/1e3:.1f}k")

    train_ds = TensorDataset(torch.from_numpy(f_tr), torch.from_numpy(s_tr), torch.from_numpy(y_tr))
    val_ds = TensorDataset(torch.from_numpy(f_v), torch.from_numpy(s_v), torch.from_numpy(y_v))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("\n[Train]")
    model = train_loop(model, train_loader, val_loader, device)

    # Test
    model.eval()
    test_ds = TensorDataset(torch.from_numpy(f_te), torch.from_numpy(s_te), torch.from_numpy(y_te))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    mus = []; sigs = []
    with torch.no_grad():
        for xf, si, _ in test_loader:
            mu, sig = model(xf.to(device), si.to(device))
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

    out_dir = ROOT / "pv/experiments/ft_transformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_eval.to_parquet(out_dir / "test_predictions.parquet", index=False)
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}")

    print("\n" + "=" * 70)
    print(f"  NGBoost strong:  6.07% / 5.04%")
    print(f"  MLP+FiLM:        5.88% / 4.88%")
    print(f"  GRU+FiLM:        6.05% / 4.98%")
    print(f"  FT-Transformer:  {nmae:.2f}% / {pnmae:.2f}%")


if __name__ == "__main__":
    main()
