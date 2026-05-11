"""LSTM Baseline v2 — point accuracy 우선 튜닝.

v1 → v2 변경:
  - window: 24 → 12 (야간 노이즈 ↓)
  - hidden_shared: 64 → 128
  - hidden_private: 32 → 64
  - num_layers: 1 → 2
  - batch: 64 → 32 (SGD noise ↑)
  - lr: 3e-4 → 5e-4
  - dropout: 0.1 → 0.10/0.15 (sweep)
  - Group emb 단독 → Group + Site embedding 둘 다
  - Loss: NLL only → MAE + α·NLL (α sweep, point accuracy 우선)

3 configs (NLL weight α 와 dropout 변화):
  Config A: α=0.30, dropout=0.10  (probabilistic-leaning)
  Config B: α=0.25, dropout=0.15  (balanced + 약한 reg)
  Config C: α=0.20, dropout=0.10  (point-leaning)
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

SEQ_FEATURES = [
    "dsr_mean", "zenith_center",
    "ta", "hm", "ws", "dc10Tca",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
]
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

# v2 config (defaults)
WINDOW = 12
BATCH_SIZE = 32
HIDDEN_SHARED = 128
HIDDEN_PRIVATE = 64
N_LAYERS = 2
GROUP_EMB_DIM = 16
SITE_EMB_DIM = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8

SITE_GROUPS = {
    "고흥만수상": 0,
    "광양항세방": 1, "삼천포": 1, "영흥": 1,
    "경상대": 2, "창원": 2, "구미": 2, "예천": 2,
}
N_GROUPS = 3
GROUP_NAMES = {0: "수상", 1: "해안", 2: "내륙"}

CONFIGS = {
    "A_alpha030_drop010": {"alpha": 0.30, "dropout": 0.10},
    "B_alpha025_drop015": {"alpha": 0.25, "dropout": 0.15},
    "C_alpha020_drop010": {"alpha": 0.20, "dropout": 0.10},
}
SEED = 42


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== Model v2 ==========

class PVBaselineLSTMv2(nn.Module):
    def __init__(self, n_features: int, n_sites: int, n_groups: int = N_GROUPS,
                 group_emb_dim: int = GROUP_EMB_DIM, site_emb_dim: int = SITE_EMB_DIM,
                 hidden_shared: int = HIDDEN_SHARED, hidden_private: int = HIDDEN_PRIVATE,
                 n_layers: int = N_LAYERS, dropout: float = 0.10):
        super().__init__()
        self.group_emb = nn.Embedding(n_groups, group_emb_dim)
        self.site_emb = nn.Embedding(n_sites, site_emb_dim)

        # Shared LSTM — features only (truly shared)
        self.lstm_shared = nn.LSTM(
            n_features, hidden_shared,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Private LSTM — features + group + site (full conditioning)
        self.lstm_private = nn.LSTM(
            n_features + group_emb_dim + site_emb_dim, hidden_private,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Project private → shared dim for fusion
        self.private_proj = nn.Linear(hidden_private, hidden_shared)

        # Attention-style fusion gate
        self.gate = nn.Sequential(
            nn.Linear(hidden_shared * 2, hidden_shared),
            nn.Sigmoid(),
        )

        self.head_dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(hidden_shared, 1)
        self.log_sig_head = nn.Linear(hidden_shared, 1)

    def forward(self, x_seq, group_id, site_id):
        B, T, _ = x_seq.shape
        g_e = self.group_emb(group_id)   # (B, gE)
        s_e = self.site_emb(site_id)     # (B, sE)

        shared_out, _ = self.lstm_shared(x_seq)
        h_shared = shared_out[:, -1]

        cond_t = torch.cat([g_e, s_e], dim=-1).unsqueeze(1).expand(-1, T, -1)
        x_priv = torch.cat([x_seq, cond_t], dim=-1)
        priv_out, _ = self.lstm_private(x_priv)
        h_priv = priv_out[:, -1]
        h_priv_proj = self.private_proj(h_priv)

        gate_input = torch.cat([h_shared, h_priv_proj], dim=-1)
        gate = self.gate(gate_input)
        h_fused = gate * h_shared + (1 - gate) * h_priv_proj

        h_fused = self.head_dropout(h_fused)
        mu = self.mu_head(h_fused).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h_fused)).squeeze(-1) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


def loss_combined(y, mu, sigma, alpha):
    """MAE + α·NLL — point accuracy 우선."""
    mae = (y - mu).abs().mean()
    nll = gaussian_nll(y, mu, sigma).mean()
    return mae + alpha * nll, mae.item(), nll.item()


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


def build_sequences(full_df, all_sites, target_start, target_end, window=WINDOW, scalers=None):
    site_idx = {s: i for i, s in enumerate(all_sites)}
    if scalers is not None:
        mean_f, std_f = scalers
    else:
        mean_f = std_f = None

    seqs, groups, sites, targets, meta = [], [], [], [], []
    for site, sub in full_df.groupby("site"):
        sub = sub.sort_values("datetime_kst").reset_index(drop=True)
        feats = sub[SEQ_FEATURES].values.astype(np.float32)
        cf = sub["cf"].values.astype(np.float32)
        is_day = sub["is_daytime"].values
        gid = int(sub["group_id"].iloc[0])
        sid = site_idx[site]

        for t in range(window, len(sub)):
            if not is_day[t]:
                continue
            target_dt = sub["datetime_kst"].iloc[t]
            if target_dt < target_start or target_dt >= target_end:
                continue
            seqs.append(feats[t - window:t])
            groups.append(gid)
            sites.append(sid)
            targets.append(cf[t])
            meta.append({"datetime_kst": target_dt, "site": site,
                         "site_capacity_kw": float(sub["site_capacity_kw"].iloc[t])})

    X = np.stack(seqs)
    G = np.array(groups, dtype=np.int64)
    S = np.array(sites, dtype=np.int64)
    Y = np.array(targets, dtype=np.float32)
    M = pd.DataFrame(meta)

    if mean_f is None:
        mean_f = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std_f = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-6
    X = (X - mean_f) / std_f
    return X, G, S, Y, M, (mean_f, std_f)


# ========== Train / Eval ==========

def train_one_config(cfg_name, alpha, dropout, X_tr, G_tr, S_tr, Y_tr,
                     X_v, G_v, S_v, Y_v, n_features, n_sites, device):
    print(f"\n{'='*70}", flush=True)
    print(f"Config: {cfg_name}  α(NLL)={alpha}, dropout={dropout}", flush=True)
    print(f"{'='*70}", flush=True)
    set_all_seeds(SEED)

    model = PVBaselineLSTMv2(
        n_features=n_features, n_sites=n_sites,
        hidden_shared=HIDDEN_SHARED, hidden_private=HIDDEN_PRIVATE,
        n_layers=N_LAYERS, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e3:.1f}k", flush=True)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(G_tr),
        torch.from_numpy(S_tr), torch.from_numpy(Y_tr),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_v), torch.from_numpy(G_v),
        torch.from_numpy(S_v), torch.from_numpy(Y_v),
    )
    g_loader = torch.Generator(); g_loader.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_loader)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        tr_loss = 0; tr_mae = 0; tr_nll = 0; n = 0
        for x, g, s, y in train_loader:
            x, g, s, y = x.to(device), g.to(device), s.to(device), y.to(device)
            mu, sig = model(x, g, s)
            loss, mae_v, nll_v = loss_combined(y, mu, sig, alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(y)
            tr_mae += mae_v * len(y); tr_nll += nll_v * len(y); n += len(y)
        tr_loss /= n; tr_mae /= n; tr_nll /= n
        sch.step()

        model.eval()
        v = 0; vmae = 0; vnll = 0; nv = 0
        with torch.no_grad():
            for x, g, s, y in val_loader:
                x, g, s, y = x.to(device), g.to(device), s.to(device), y.to(device)
                mu, sig = model(x, g, s)
                loss, mae_v, nll_v = loss_combined(y, mu, sig, alpha)
                v += loss.item() * len(y); vmae += mae_v * len(y); vnll += nll_v * len(y); nv += len(y)
        v /= nv; vmae /= nv; vnll /= nv

        # Use val MAE as primary tracking metric (point accuracy 우선)
        primary_metric = vmae
        if primary_metric < best:
            best = primary_metric
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            bad = 0; mark = "★"
        else:
            bad += 1; mark = ""
        if ep % 5 == 0 or mark:
            print(f"  ep {ep:>3}: tr_loss {tr_loss:.4f} (mae {tr_mae:.4f}, nll {tr_nll:.4f}) | "
                  f"val_mae {vmae:.4f} val_nll {vnll:.4f} {mark}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


def evaluate(model, X, G, S, Y, meta, device, label):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(G), torch.from_numpy(S), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 4)
    mus, sigs = [], []
    with torch.no_grad():
        for x, g, s, _ in loader:
            mu, sig = model(x.to(device), g.to(device), s.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)

    cap = meta["site_capacity_kw"].values
    err = np.abs(Y - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((Y >= mu_arr - 1.282 * sig_arr) & (Y <= mu_arr + 1.282 * sig_arr)).mean() * 100
    cov95 = ((Y >= mu_arr - 1.96 * sig_arr) & (Y <= mu_arr + 1.96 * sig_arr)).mean() * 100

    df_eval = meta.copy(); df_eval["mu"] = mu_arr; df_eval["err"] = err
    df_eval["pred_kwh"] = mu_arr * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = Y * df_eval["site_capacity_kw"]
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"))
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100

    print(f"\n=== {label} ===", flush=True)
    print(f"  rows {len(Y):,}  NMAE {nmae:.3f}%  Port {pnmae:.3f}%  Cov80 {cov80:.1f}%  Cov95 {cov95:.1f}%", flush=True)
    return {"nmae": nmae, "pnmae": pnmae, "cov80": cov80, "cov95": cov95}


def main():
    print("=" * 70, flush=True)
    print("LSTM Baseline v2 — point accuracy 튜닝 (3 configs)", flush=True)
    print("=" * 70, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = load_data()
    all_sites = sorted(df["site"].unique())
    n_sites = len(all_sites)
    print(f"\n[Data] sites: {n_sites}, groups: {N_GROUPS}", flush=True)

    print(f"\n[Build] sequences (window={WINDOW}, full_df + target filter)...", flush=True)
    X_tr, G_tr, S_tr, Y_tr, M_tr, scalers = build_sequences(
        df, all_sites, df.datetime_kst.min(), TRAIN_END, window=WINDOW)
    X_v, G_v, S_v, Y_v, M_v, _ = build_sequences(
        df, all_sites, TRAIN_END, VAL_END, window=WINDOW, scalers=scalers)
    X_te, G_te, S_te, Y_te, M_te, _ = build_sequences(
        df, all_sites, VAL_END, df.datetime_kst.max() + pd.Timedelta(hours=1),
        window=WINDOW, scalers=scalers)
    print(f"  train {len(Y_tr):,}, val {len(Y_v):,}, test {len(Y_te):,}", flush=True)
    print(f"  feature dim: {X_tr.shape[2]}, window: {X_tr.shape[1]}", flush=True)

    n_features = X_tr.shape[2]

    results = {}
    for cfg_name, cfg in CONFIGS.items():
        model = train_one_config(
            cfg_name, cfg["alpha"], cfg["dropout"],
            X_tr, G_tr, S_tr, Y_tr, X_v, G_v, S_v, Y_v,
            n_features, n_sites, device,
        )
        metrics = evaluate(model, X_te, G_te, S_te, Y_te, M_te, device, f"TEST [{cfg_name}]")
        results[cfg_name] = metrics

        out_dir = ROOT / f"pv/experiments/lstm_v2_{cfg_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "scalers": scalers,
                    "config": cfg, "metrics": metrics}, out_dir / "model.pt")

    print("\n" + "=" * 70, flush=True)
    print("최종 비교", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Config':<25} {'NMAE %':>8} {'Port %':>8} {'Cov80':>7} {'Cov95':>7}", flush=True)
    print("-" * 65, flush=True)
    print(f"  {'NGBoost baseline':<25} {6.20:>7.2f}% {5.18:>7.2f}% {80.6:>6.1f}% {92.9:>6.1f}%", flush=True)
    print(f"  {'ResMLP+AdaLN':<25} {5.12:>7.2f}% {4.13:>7.2f}% {82.6:>6.1f}% {93.2:>6.1f}%", flush=True)
    print(f"  {'LSTM v1 (baseline)':<25} {6.69:>7.2f}% {5.58:>7.2f}% {84.6:>6.1f}% {94.1:>6.1f}%", flush=True)
    print("-" * 65, flush=True)
    for cfg_name, m in results.items():
        print(f"  {'LSTM v2 ' + cfg_name:<25} {m['nmae']:>7.2f}% {m['pnmae']:>7.2f}% "
              f"{m['cov80']:>6.1f}% {m['cov95']:>6.1f}%", flush=True)


if __name__ == "__main__":
    main()
