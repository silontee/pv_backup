"""Step 3 (base) / Step 4 (trans) — ResMLP+AdaLN+TCN hybrid (UNIFIED config v2).

Architecture: 동일 (ResMLP+AdaLN static branch + shallow TCN temporal branch + fusion).

variant ∈ {base, trans}:
  base : TCN input = ["dsr_mean","zenith_center","ta","hm","ws","dc10Tca"]   (6 feats)
  trans: 위 + ["cloud_delta","dsr_delta","cloud_volatility","dsr_volatility"] (10 feats)
         all weather-only (no cf leak). centered window OK (perfect-foresight).

UNIFIED config (Step1 baseline과 동일):
  batch=64, lr=7e-4, wd=1e-4, AdamW + CosineAnnealingLR(T_max=50),
  max_ep=50, patience=8, grad_clip=1.0, loss = MAE + 0.2*GaussianNLL,
  val tracking metric = val_mae.

CLI:
  --variant {base,trans}   필수
  --seeds  "42,123"        콤마 구분 (없으면 SEEDS 전체)
  --aggregate              저장된 결과만 모아 ensemble 메트릭 출력
"""
import argparse
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

# Static features — Step 1 baseline과 동일
MAIN_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws"]
COND_NUM_FEATURES = ["dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos"]

# Temporal base features (TCN input)
TEMPORAL_BASE = ["dsr_mean", "zenith_center", "ta", "hm", "ws", "dc10Tca"]
TEMPORAL_TRANS_EXTRA = ["cloud_delta", "dsr_delta", "cloud_volatility", "dsr_volatility"]
WINDOW = 12              # 12h centered window
WINDOW_OFFSET_LEFT = 6   # 6 hours past, 6 hours future
VOL_WIN = 3              # ±3h centered std for volatility

TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

DIM = 64
N_BLOCKS = 4
DIM_TEMPORAL = 32
TCN_KERNEL = 3

# ===== UNIFIED CONFIG =====
BATCH_SIZE = 64
LR = 7e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0
ALPHA_NLL = 0.20
SEEDS_DEFAULT = [42, 123, 7]


def out_dir_for(variant):
    return ROOT / f"pv/experiments/resmlp_tcn_v2_{variant}"


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
        self.input_proj = nn.Linear(n_main, dim)
        self.blocks = nn.ModuleList([AdaLNBlock(dim, n_cond) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_final = nn.Linear(n_cond, 2 * dim)

        self.tcn = nn.Sequential(
            nn.Conv1d(n_temporal_feat, dim_temporal, kernel_size=TCN_KERNEL, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(dim_temporal, dim_temporal, kernel_size=TCN_KERNEL, padding=2, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(dim + dim_temporal, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )
        self.mu_head = nn.Linear(dim, 1)
        self.log_sig_head = nn.Linear(dim, 1)

    def forward(self, x_main, x_cond, x_temporal):
        x = self.input_proj(x_main)
        for blk in self.blocks:
            x = blk(x, x_cond)
        h = self.final_norm(x)
        scale, shift = self.cond_final(x_cond).chunk(2, dim=-1)
        h_static = h * (1 + scale) + shift
        h_temp = self.tcn(x_temporal.transpose(1, 2)).squeeze(-1)
        h_fused = self.fusion(torch.cat([h_static, h_temp], dim=-1))
        mu = self.mu_head(h_fused).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h_fused)).squeeze(-1) + 1e-3
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return 0.5 * (torch.log(2 * np.pi * sigma ** 2) + ((y - mu) ** 2) / (sigma ** 2))


def loss_combined(y, mu, sigma):
    mae = (y - mu).abs().mean()
    nll = gaussian_nll(y, mu, sigma).mean()
    return mae + ALPHA_NLL * nll


# ========== Data ==========

def load_data(variant):
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)
    df["is_daytime"] = (df["dsr_mean"].notna() & df["cf"].notna()).astype(np.int32)
    for col in ["ta", "hm", "ws", "dc10Tca", "dsr_mean", "zenith_center"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["cf"] = df["cf"].fillna(0)

    if variant == "trans":
        # weather-only transition features (no cf 사용 금지)
        # delta = current - prev (per site, hour spacing)
        df["cloud_delta"] = df.groupby("site")["dc10Tca"].diff().fillna(0)
        df["dsr_delta"] = df.groupby("site")["dsr_mean"].diff().fillna(0)
        # volatility = centered rolling std over ±VOL_WIN window
        df["cloud_volatility"] = df.groupby("site")["dc10Tca"].transform(
            lambda s: s.rolling(window=2 * VOL_WIN + 1, center=True, min_periods=1).std()).fillna(0)
        df["dsr_volatility"] = df.groupby("site")["dsr_mean"].transform(
            lambda s: s.rolling(window=2 * VOL_WIN + 1, center=True, min_periods=1).std()).fillna(0)
    return df


def build(full_df, all_sites, target_start, target_end, temporal_feats, scalers=None):
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
        t_arr = sub[temporal_feats].values.astype(np.float32)
        is_day = sub["is_daytime"].values
        right_offset = WINDOW - WINDOW_OFFSET_LEFT
        for t in range(WINDOW_OFFSET_LEFT, len(sub) - right_offset):
            if not is_day[t]:
                continue
            target_dt = sub["datetime_kst"].iloc[t]
            if target_dt < target_start or target_dt >= target_end:
                continue
            main_list.append(m_arr[t])
            cond_list.append(cond[t])
            temporal_list.append(t_arr[t - WINDOW_OFFSET_LEFT:t + right_offset])
            y_list.append(sub["cf"].iloc[t])
            meta.append({"datetime_kst": target_dt, "site": site,
                         "site_capacity_kw": float(sub["site_capacity_kw"].iloc[t])})

    X_main = np.stack(main_list)
    X_cond = np.stack(cond_list)
    X_temp = np.stack(temporal_list)
    Y = np.array(y_list, dtype=np.float32)
    M = pd.DataFrame(meta)

    if mean_m is None:
        mean_m = X_main.mean(axis=0); std_m = X_main.std(axis=0) + 1e-6
        mean_t = X_temp.reshape(-1, X_temp.shape[-1]).mean(axis=0)
        std_t = X_temp.reshape(-1, X_temp.shape[-1]).std(axis=0) + 1e-6
    X_main = (X_main - mean_m) / std_m
    X_temp = (X_temp - mean_t) / std_t
    return X_main, X_cond, X_temp, Y, M, (mean_m, std_m, mean_t, std_t)


# ========== Eval ==========

def predict(model, X_main, X_cond, X_temp, Y, device):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_main), torch.from_numpy(X_cond),
                       torch.from_numpy(X_temp), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=2048)
    mus, sigs = [], []
    with torch.no_grad():
        for xm, xc, xt, _ in loader:
            mu, sig = model(xm.to(device), xc.to(device), xt.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    return np.clip(np.concatenate(mus), 0, None), np.concatenate(sigs)


def eval_metrics(y, mu, sigma, cap):
    err = np.abs(y - mu)
    nmae = (err * cap).sum() / cap.sum() * 100
    bias = ((mu - y) * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu - 1.282 * sigma) & (y <= mu + 1.282 * sigma)).mean() * 100
    cov95 = ((y >= mu - 1.96 * sigma) & (y <= mu + 1.96 * sigma)).mean() * 100
    nll = (0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)).mean()
    return {"nmae": nmae, "bias": bias, "cov80": cov80, "cov95": cov95, "nll": float(nll)}


def per_site_metrics(y, mu, sigma, sites, cap):
    df = pd.DataFrame({"site": sites, "y": y, "mu": mu, "sigma": sigma, "cap": cap})
    df["err"] = (df.y - df.mu).abs()
    df["in_80"] = ((df.y >= df.mu - 1.282 * df.sigma) & (df.y <= df.mu + 1.282 * df.sigma)).astype(int)
    df["in_95"] = ((df.y >= df.mu - 1.96 * df.sigma) & (df.y <= df.mu + 1.96 * df.sigma)).astype(int)
    return df.groupby("site").apply(lambda g: pd.Series({
        "n": len(g),
        "nmae": (g.err * g.cap).sum() / g.cap.sum() * 100,
        "bias": ((g.mu - g.y) * g.cap).sum() / g.cap.sum() * 100,
        "cov80": g.in_80.mean() * 100,
        "cov95": g.in_95.mean() * 100,
    }), include_groups=False).reset_index()


def portfolio_nmae(y, mu, sigma, sites, datetimes, caps):
    df = pd.DataFrame({"datetime_kst": datetimes, "site": sites,
                       "y": y, "mu": mu, "sigma": sigma, "cap": caps})
    df["pred_kwh"] = df.mu * df.cap
    df["actual_kwh"] = df.y * df.cap
    df["var_kwh2"] = (df.sigma * df.cap) ** 2
    port = df.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        var_sum=("var_kwh2", "sum"), cap=("cap", "sum"))
    port["sigma_kwh"] = np.sqrt(port["var_sum"])
    nmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    cov80 = ((port["actual"] >= port["pred"] - 1.282 * port["sigma_kwh"]) &
             (port["actual"] <= port["pred"] + 1.282 * port["sigma_kwh"])).mean() * 100
    return {"port_nmae": nmae, "port_cov80": cov80}


# ========== Train ==========

def train_one(variant, seed, df, all_sites, temporal_feats, device, out_dir):
    print(f"\n{'='*70}")
    print(f"[Seed {seed}] variant={variant}  temporal_feats={len(temporal_feats)}")
    print(f"{'='*70}")
    set_all_seeds(seed)

    print(f"  build sequences (window={WINDOW} centered)...", flush=True)
    X_m_tr, X_c_tr, X_t_tr, Y_tr, M_tr, scalers = build(
        df, all_sites, df.datetime_kst.min(), TRAIN_END, temporal_feats)
    X_m_v, X_c_v, X_t_v, Y_v, M_v, _ = build(
        df, all_sites, TRAIN_END, VAL_END, temporal_feats, scalers=scalers)
    X_m_te, X_c_te, X_t_te, Y_te, M_te, _ = build(
        df, all_sites, VAL_END, df.datetime_kst.max() + pd.Timedelta(hours=1),
        temporal_feats, scalers=scalers)
    print(f"  train {len(Y_tr):,}, val {len(Y_v):,}, test {len(Y_te):,}", flush=True)
    print(f"  temporal shape: {X_t_tr.shape}", flush=True)

    n_main = X_m_tr.shape[1]; n_cond = X_c_tr.shape[1]; n_temp = X_t_tr.shape[2]
    model = ResMLP_AdaLN_TCN(n_main, n_cond, n_temp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e3:.1f}k", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_m_tr), torch.from_numpy(X_c_tr),
                             torch.from_numpy(X_t_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_m_v), torch.from_numpy(X_c_v),
                           torch.from_numpy(X_t_v), torch.from_numpy(Y_v))
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=2048)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xm, xc, xt, y in train_loader:
            xm, xc, xt, y = xm.to(device), xc.to(device), xt.to(device), y.to(device)
            mu, sig = model(xm, xc, xt)
            loss = loss_combined(y, mu, sig)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()
        model.eval()
        v_mae = 0; nv = 0
        with torch.no_grad():
            for xm, xc, xt, y in val_loader:
                xm, xc, xt, y = xm.to(device), xc.to(device), xt.to(device), y.to(device)
                mu, sig = model(xm, xc, xt)
                v_mae += (y - mu).abs().sum().item(); nv += len(y)
        v_mae /= nv
        if v_mae < best:
            best = v_mae; bad = 0
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            mark = "★"
        else:
            bad += 1; mark = ""
        if ep % 5 == 0 or mark:
            print(f"  ep {ep:>3}: val_mae {v_mae:.4f} {mark}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)

    mu_v, sig_v = predict(model, X_m_v, X_c_v, X_t_v, Y_v, device)
    mu_te, sig_te = predict(model, X_m_te, X_c_te, X_t_te, Y_te, device)

    cap_te = M_te["site_capacity_kw"].values
    sites_te = M_te["site"].values
    dt_te = M_te["datetime_kst"].values
    m = eval_metrics(Y_te, mu_te, sig_te, cap_te)
    p = portfolio_nmae(Y_te, mu_te, sig_te, sites_te, dt_te, cap_te)
    m.update(p)
    print(f"  Test: NMAE {m['nmae']:.3f}% / Port {m['port_nmae']:.3f}% / "
          f"Cov80 {m['cov80']:.1f}% / NLL {m['nll']:.4f}", flush=True)

    out = out_dir / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "datetime_kst": M_v["datetime_kst"].values, "site": M_v["site"].values,
        "site_capacity_kw": M_v["site_capacity_kw"].values,
        "cf": Y_v, "pred_mu": mu_v, "pred_sigma": sig_v,
    }).to_parquet(out / "val_predictions.parquet", index=False)
    pd.DataFrame({
        "datetime_kst": dt_te, "site": sites_te, "site_capacity_kw": cap_te,
        "cf": Y_te, "pred_mu": mu_te, "pred_sigma": sig_te,
    }).to_parquet(out / "test_predictions.parquet", index=False)
    return m


def aggregate(variant):
    out_dir = out_dir_for(variant)
    print("=" * 70)
    print(f"Aggregate — variant={variant}")
    print("=" * 70)
    seed_dirs = sorted(out_dir.glob("seed_*/test_predictions.parquet"))
    if not seed_dirs:
        print("저장된 seed 결과 없음."); return
    found_seeds = [int(p.parent.name.split("_")[-1]) for p in seed_dirs]
    print(f"  발견된 seeds: {found_seeds}")

    per_seed = []
    for seed in found_seeds:
        df_te = pd.read_parquet(out_dir / f"seed_{seed}/test_predictions.parquet")
        cap = df_te["site_capacity_kw"].values
        y = df_te["cf"].values; mu = df_te["pred_mu"].values; sig = df_te["pred_sigma"].values
        m = eval_metrics(y, mu, sig, cap)
        m.update(portfolio_nmae(y, mu, sig, df_te["site"].values, df_te["datetime_kst"].values, cap))
        m["seed"] = seed; per_seed.append(m)
    seed_df = pd.DataFrame(per_seed)
    print("\nPer-seed (test):")
    print(seed_df.round(3).to_string(index=False))
    print(f"\n  Mean ± Std:")
    for col in ["nmae", "port_nmae", "cov80", "cov95", "nll"]:
        v = seed_df[col].values
        print(f"    {col:<12} {v.mean():>7.3f} ± {v.std(ddof=1):>5.3f}")

    dfs = []
    for seed in found_seeds:
        d = pd.read_parquet(out_dir / f"seed_{seed}/test_predictions.parquet"); d["seed"] = seed
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    grp = all_df.groupby(["datetime_kst", "site"])
    ens = grp.agg(
        cf=("cf", "first"),
        site_capacity_kw=("site_capacity_kw", "first"),
        mu_mean=("pred_mu", "mean"),
        mu_var=("pred_mu", "var"),
        sigma_sq_mean=("pred_sigma", lambda s: (s ** 2).mean()),
    ).reset_index()
    ens["sigma_total"] = np.sqrt(ens["sigma_sq_mean"] + ens["mu_var"])
    cap = ens["site_capacity_kw"].values
    y = ens["cf"].values; mu = ens["mu_mean"].values; sigma = ens["sigma_total"].values
    m_ens = eval_metrics(y, mu, sigma, cap)
    p_ens = portfolio_nmae(y, mu, sigma, ens["site"].values, ens["datetime_kst"].values, cap)
    print(f"\n  Ensemble: NMAE {m_ens['nmae']:.3f}% / Port {p_ens['port_nmae']:.3f}% / "
          f"bias {m_ens['bias']:+.3f}% / Cov80 {m_ens['cov80']:.1f}% / "
          f"Cov95 {m_ens['cov95']:.1f}% / NLL {m_ens['nll']:.4f}")
    print("\n  Per-site (ensemble):")
    site_ens = per_site_metrics(y, mu, sigma, ens["site"].values, cap)
    print(site_ens.round(2).to_string(index=False))

    ens.to_parquet(out_dir / "ensemble_test.parquet", index=False)
    seed_df.to_csv(out_dir / "per_seed_summary.csv", index=False)
    site_ens.to_csv(out_dir / "ensemble_per_site.csv", index=False)
    print(f"\n저장: {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["base", "trans"], required=False, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()

    if args.aggregate:
        if not args.variant:
            raise SystemExit("--aggregate 사용 시 --variant 필요")
        aggregate(args.variant); return
    if not args.variant:
        raise SystemExit("--variant {base,trans} 필요")

    out_dir = out_dir_for(args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS_DEFAULT
    temporal_feats = TEMPORAL_BASE if args.variant == "base" \
                      else TEMPORAL_BASE + TEMPORAL_TRANS_EXTRA

    print("=" * 70)
    print(f"Step 3/4 TCN | variant={args.variant} | seeds={seeds}")
    print(f"  temporal_feats ({len(temporal_feats)}): {temporal_feats}")
    print(f"  batch={BATCH_SIZE}, lr={LR}, max_ep={MAX_EPOCHS}, "
          f"patience={PATIENCE}, loss=MAE+{ALPHA_NLL}·NLL")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data(args.variant)
    all_sites = sorted(df["site"].unique())

    per_seed = []
    for seed in seeds:
        m = train_one(args.variant, seed, df, all_sites, temporal_feats, device, out_dir)
        m["seed"] = seed
        per_seed.append(m)
    pd.DataFrame(per_seed).to_csv(
        out_dir / f"per_seed_summary_{'_'.join(map(str, seeds))}.csv", index=False)


if __name__ == "__main__":
    main()
