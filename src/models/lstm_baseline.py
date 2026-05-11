"""D-1 baseline: Group-conditioned shared/private LSTM + fusion gate + Gaussian head.

Architecture:
  Input: (B, T=24, F) sequence + group_id (3 groups: 수상/해안/내륙)

  ├─ Group Embedding (3 → 16)
  │
  ├─ Shared LSTM   (input: F → hidden_shared=64, 1 layer)
  │   ↓ last hidden h_shared (B, 64)
  │
  ├─ Private LSTM  (input: F+16 group_emb → hidden_private=32, 1 layer)
  │   ↓ last hidden h_priv (B, 32)
  │   → Linear projection to (B, 64) for fusion
  │
  ├─ Fusion Gate (attention-style):
  │   gate = σ(Linear([h_shared; h_priv_proj]))
  │   h_fused = gate * h_shared + (1-gate) * h_priv_proj
  │
  └─ Heads (dropout 0.1):
      μ_head, log_σ_head → softplus → σ

Loss: Gaussian NLL.

Config:
  window=24, batch=64, hidden_shared=64, hidden_private=32, 1-layer,
  head dropout=0.1, AdamW(lr=3e-4, wd=1e-4), max_epochs=50, patience=8.

Group definition (from data_strategy.md):
  수상  : 고흥만수상              → group 0
  해안  : 광양항세방, 삼천포, 영흥 → group 1
  내륙  : 경상대, 창원, 구미, 예천 → group 2
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

# Config
SEQ_FEATURES = [
    "dsr_mean", "zenith_center",
    "ta", "hm", "ws", "dc10Tca",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

WINDOW = 24
BATCH_SIZE = 64
HIDDEN_SHARED = 64
HIDDEN_PRIVATE = 32
N_LAYERS = 1
GROUP_EMB_DIM = 16
HEAD_DROPOUT = 0.1
LR = 3e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8

SITE_GROUPS = {
    "고흥만수상": 0,    # 수상
    "광양항세방": 1, "삼천포": 1, "영흥": 1,   # 해안
    "경상대": 2, "창원": 2, "구미": 2, "예천": 2,   # 내륙
}
N_GROUPS = 3
GROUP_NAMES = {0: "수상", 1: "해안", 2: "내륙"}


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== Model ==========

class PVBaselineLSTM(nn.Module):
    def __init__(self, n_features: int, n_groups: int = N_GROUPS,
                 group_emb_dim: int = GROUP_EMB_DIM,
                 hidden_shared: int = HIDDEN_SHARED,
                 hidden_private: int = HIDDEN_PRIVATE,
                 n_layers: int = N_LAYERS,
                 head_dropout: float = HEAD_DROPOUT):
        super().__init__()
        # Group embedding
        self.group_emb = nn.Embedding(n_groups, group_emb_dim)

        # Shared LSTM
        self.lstm_shared = nn.LSTM(n_features, hidden_shared,
                                    num_layers=n_layers, batch_first=True)

        # Private LSTM (input augmented with group embedding)
        self.lstm_private = nn.LSTM(n_features + group_emb_dim, hidden_private,
                                     num_layers=n_layers, batch_first=True)

        # Project private → shared dim for fusion
        self.private_proj = nn.Linear(hidden_private, hidden_shared)

        # Attention-style fusion gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_shared * 2, hidden_shared),
            nn.Sigmoid(),
        )

        # Output heads
        self.head_dropout = nn.Dropout(head_dropout)
        self.mu_head = nn.Linear(hidden_shared, 1)
        self.log_sig_head = nn.Linear(hidden_shared, 1)

    def forward(self, x_seq: torch.Tensor, group_id: torch.Tensor):
        """
        x_seq: (B, T, F)
        group_id: (B,) long
        Returns: (mu, sigma) each shape (B,)
        """
        B, T, _ = x_seq.shape
        g_e = self.group_emb(group_id)   # (B, group_emb_dim)

        # Shared
        shared_out, _ = self.lstm_shared(x_seq)   # (B, T, H_s)
        h_shared = shared_out[:, -1]   # (B, H_s)

        # Private (group-conditioned via input concat)
        g_e_t = g_e.unsqueeze(1).expand(-1, T, -1)
        x_priv = torch.cat([x_seq, g_e_t], dim=-1)
        priv_out, _ = self.lstm_private(x_priv)
        h_priv = priv_out[:, -1]   # (B, H_p)
        h_priv_proj = self.private_proj(h_priv)   # (B, H_s)

        # Fusion gate
        gate_input = torch.cat([h_shared, h_priv_proj], dim=-1)
        gate = self.gate(gate_input)
        h_fused = gate * h_shared + (1 - gate) * h_priv_proj

        # Heads
        h_fused = self.head_dropout(h_fused)
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
    df["group_id"] = df["site"].map(SITE_GROUPS).astype(np.int64)
    return df


def build_sequences(full_df, target_start, target_end, scalers=None):
    """24h sliding window, target만 [target_start, target_end) 필터.

    History는 full_df에서 자유롭게 가져옴 (boundary cleanly).
    """
    if scalers is not None:
        mean_f, std_f = scalers
    else:
        mean_f = std_f = None

    seqs, groups, targets, meta = [], [], [], []
    for site, sub in full_df.groupby("site"):
        sub = sub.sort_values("datetime_kst").reset_index(drop=True)
        feats = sub[SEQ_FEATURES].values.astype(np.float32)
        cf = sub["cf"].values.astype(np.float32)
        is_day = sub["is_daytime"].values
        gid = int(sub["group_id"].iloc[0])

        for t in range(WINDOW, len(sub)):
            if not is_day[t]:
                continue
            target_dt = sub["datetime_kst"].iloc[t]
            if target_dt < target_start or target_dt >= target_end:
                continue
            seqs.append(feats[t - WINDOW:t])    # 24h history
            groups.append(gid)
            targets.append(cf[t])
            meta.append({"datetime_kst": target_dt, "site": site,
                         "site_capacity_kw": float(sub["site_capacity_kw"].iloc[t])})

    X = np.stack(seqs)             # (N, T, F)
    G = np.array(groups, dtype=np.int64)
    Y = np.array(targets, dtype=np.float32)
    M = pd.DataFrame(meta)

    # Standardize features (fit on train only, reused on val/test)
    if mean_f is None:
        mean_f = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std_f = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    X = (X - mean_f) / std_f
    return X, G, Y, M, (mean_f, std_f)


# ========== Train ==========

def train_loop(model, train_loader, val_loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        tr = 0; n = 0
        for x, g, y in train_loader:
            x, g, y = x.to(device), g.to(device), y.to(device)
            mu, sig = model(x, g)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * len(y); n += len(y)
        tr /= n; sch.step()

        model.eval()
        v = 0; vmae = 0; nv = 0
        with torch.no_grad():
            for x, g, y in val_loader:
                x, g, y = x.to(device), g.to(device), y.to(device)
                mu, sig = model(x, g)
                v += gaussian_nll(y, mu, sig).mean().item() * len(y)
                vmae += (y - mu).abs().sum().item(); nv += len(y)
        v /= nv; vmae /= nv

        if v < best:
            best = v
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            bad = 0; mark = "★"
        else:
            bad += 1; mark = ""
        if ep % 5 == 0 or mark:
            print(f"  ep {ep:>3}: tr {tr:.4f} val {v:.4f} mae {vmae:.4f} {mark}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


def evaluate(model, X, G, Y, meta, device, label):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(G), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 4)
    mus, sigs = [], []
    with torch.no_grad():
        for x, g, _ in loader:
            mu, sig = model(x.to(device), g.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)

    cap = meta["site_capacity_kw"].values
    err = np.abs(Y - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((Y >= mu_arr - 1.282 * sig_arr) & (Y <= mu_arr + 1.282 * sig_arr)).mean() * 100
    cov95 = ((Y >= mu_arr - 1.96 * sig_arr) & (Y <= mu_arr + 1.96 * sig_arr)).mean() * 100
    print(f"\n=== {label} ===", flush=True)
    print(f"  rows: {len(Y):,}, NMAE: {nmae:.2f}%, cov80: {cov80:.1f}%, cov95: {cov95:.1f}%", flush=True)

    df_eval = meta.copy(); df_eval["cf"] = Y; df_eval["mu"] = mu_arr; df_eval["sig"] = sig_arr; df_eval["err"] = err
    print(f"  사이트별 NMAE:", flush=True)
    for s in sorted(df_eval["site"].unique()):
        sub = df_eval[df_eval["site"] == s]
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        gname = GROUP_NAMES[SITE_GROUPS[s]]
        print(f"    {s:<14} (group={gname:<3}) {n:>6.2f}%", flush=True)

    df_eval["pred_kwh"] = mu_arr * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = Y * df_eval["site_capacity_kw"]
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"))
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    print(f"  포트폴리오 NMAE: {pnmae:.2f}%", flush=True)
    return df_eval, nmae, pnmae, cov80


def main():
    print("=" * 70, flush=True)
    print("LSTM Baseline (D-1) — group-conditioned shared/private + fusion gate", flush=True)
    print("=" * 70, flush=True)
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = load_data()
    print(f"\n[Data] total rows {len(df):,}, daytime {df['is_daytime'].sum():,}", flush=True)
    print(f"  Site → Group: {SITE_GROUPS}", flush=True)

    print("\n[Build] sequences (window=24, full_df + target filter)...", flush=True)
    X_tr, G_tr, Y_tr, M_tr, scalers = build_sequences(df, df.datetime_kst.min(), TRAIN_END)
    X_v, G_v, Y_v, M_v, _ = build_sequences(df, TRAIN_END, VAL_END, scalers)
    X_te, G_te, Y_te, M_te, _ = build_sequences(df, VAL_END, df.datetime_kst.max() + pd.Timedelta(hours=1), scalers)
    print(f"  train {len(Y_tr):,}, val {len(Y_v):,}, test {len(Y_te):,}", flush=True)
    print(f"  feature dim: {X_tr.shape[2]}, window: {X_tr.shape[1]}", flush=True)

    n_features = X_tr.shape[2]
    model = PVBaselineLSTM(n_features=n_features).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] params: {n_params/1e3:.1f}k", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(G_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(G_v), torch.from_numpy(Y_v))
    g_loader = torch.Generator(); g_loader.manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_loader)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("\n[Train]", flush=True)
    model = train_loop(model, train_loader, val_loader, device)

    df_te, nmae, pnmae, cov80 = evaluate(model, X_te, G_te, Y_te, M_te, device, "TEST")

    out_dir = ROOT / "pv/experiments/lstm_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_te.rename(columns={"mu": "pred_cf", "sig": "pred_std_cf"})[
        ["datetime_kst", "site", "site_capacity_kw", "cf", "pred_cf", "pred_std_cf"]
    ].to_parquet(out_dir / "test_predictions.parquet", index=False)
    torch.save({
        "state_dict": model.state_dict(),
        "scalers": scalers,
        "site_groups": SITE_GROUPS,
        "config": {
            "window": WINDOW, "hidden_shared": HIDDEN_SHARED, "hidden_private": HIDDEN_PRIVATE,
            "group_emb_dim": GROUP_EMB_DIM, "head_dropout": HEAD_DROPOUT,
            "n_features": n_features,
        },
    }, out_dir / "model.pt")
    print(f"\n저장: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
