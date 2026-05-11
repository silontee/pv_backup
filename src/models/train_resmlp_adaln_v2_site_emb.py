"""Step 2 — Site embedding under UNIFIED config.

Backbone: ResMLP+AdaLN (DIM=64, N_BLOCKS=4) — Step 1과 동일.
Cond 구성만 비교:
  - emb  : cond = [cond_num, site_emb(16)]              # one-hot 제거
  - both : cond = [cond_num, site_one_hot(8), site_emb(16)]

UNIFIED config: batch=64, lr=7e-4, wd=1e-4, AdamW + CosineAnnealingLR(T_max=50),
max_ep=50, patience=8, grad_clip=1.0, loss = MAE + 0.2*GaussianNLL,
val tracking metric = val_mae.

CLI:
  --variant {emb,both}    필수
  --seeds  "42,123"       콤마 구분 (없으면 5 SEEDS 모두)
  --aggregate             저장된 결과만 모아 ensemble 메트릭 출력
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
sys.path.insert(0, str(ROOT / "src/models"))

from train_resmlp_adaln import (
    AdaLNBlock, MAIN_FEATURES, COND_NUM_FEATURES,
    TARGET, TRAIN_END, VAL_END, DIM, N_BLOCKS, gaussian_nll,
)

# ===== UNIFIED CONFIG =====
BATCH_SIZE = 64
LR = 7e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0
ALPHA_NLL = 0.20
SITE_EMB_DIM = 16
SEEDS = [42, 123, 7, 202, 999]


def out_dir_for(variant):
    return ROOT / f"pv/experiments/resmlp_adaln_v2_site_{variant}"


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_combined(y, mu, sigma):
    mae = (y - mu).abs().mean()
    nll = gaussian_nll(y, mu, sigma).mean()
    return mae + ALPHA_NLL * nll


def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    return df


def build(df, all_sites):
    main = df[MAIN_FEATURES].values.astype(np.float32)
    cond_num = df[COND_NUM_FEATURES].values.astype(np.float32)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    site_id = df["site"].map(site_idx).values.astype(np.int64)
    site_oh = np.eye(len(all_sites), dtype=np.float32)[site_id]
    y = df[TARGET].values.astype(np.float32)
    return main, cond_num, site_oh, site_id, y


class ResMLPAdaLN_Variant(nn.Module):
    """variant ∈ {emb, both}.
    emb : cond = [cond_num, site_emb]
    both: cond = [cond_num, site_one_hot, site_emb]
    """
    def __init__(self, n_main, n_cond_num, n_sites, variant,
                 site_emb_dim=SITE_EMB_DIM, dim=DIM, n_blocks=N_BLOCKS):
        super().__init__()
        self.variant = variant
        self.site_emb = nn.Embedding(n_sites, site_emb_dim)
        if variant == "emb":
            n_cond = n_cond_num + site_emb_dim
        elif variant == "both":
            n_cond = n_cond_num + n_sites + site_emb_dim
        else:
            raise ValueError(variant)
        self.input_proj = nn.Linear(n_main, dim)
        self.blocks = nn.ModuleList([AdaLNBlock(dim, n_cond) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_final = nn.Linear(n_cond, 2 * dim)
        self.mu_head = nn.Linear(dim, 1)
        self.log_sig_head = nn.Linear(dim, 1)

    def forward(self, x_main, x_cond_num, x_site_oh, site_id):
        s_e = self.site_emb(site_id)
        if self.variant == "emb":
            x_cond = torch.cat([x_cond_num, s_e], dim=-1)
        else:  # both
            x_cond = torch.cat([x_cond_num, x_site_oh, s_e], dim=-1)
        x = self.input_proj(x_main)
        for blk in self.blocks:
            x = blk(x, x_cond)
        h = self.final_norm(x)
        scale, shift = self.cond_final(x_cond).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h)).squeeze(-1) + 1e-3
        return mu, sigma


def predict(model, X_main, X_cond_num, X_oh, Sid, Y, device):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_main), torch.from_numpy(X_cond_num),
                       torch.from_numpy(X_oh), torch.from_numpy(Sid), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=2048)
    mus, sigs = [], []
    with torch.no_grad():
        for xm, xc, xo, sid, _ in loader:
            mu, sig = model(xm.to(device), xc.to(device), xo.to(device), sid.to(device))
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


def train_one(variant, seed, df, all_sites, device, out_dir):
    print(f"\n{'='*70}")
    print(f"[Seed {seed}] variant={variant}")
    print(f"{'='*70}")
    set_all_seeds(seed)
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]

    Xm_tr, Xcn_tr, Xoh_tr, Sid_tr, Ytr = build(train_df, all_sites)
    Xm_v, Xcn_v, Xoh_v, Sid_v, Yv = build(val_df, all_sites)
    Xm_te, Xcn_te, Xoh_te, Sid_te, Yte = build(test_df, all_sites)

    n_main = Xm_tr.shape[1]; n_cond_num = Xcn_tr.shape[1]
    model = ResMLPAdaLN_Variant(n_main, n_cond_num, len(all_sites), variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e3:.1f}k")

    train_ds = TensorDataset(torch.from_numpy(Xm_tr), torch.from_numpy(Xcn_tr),
                              torch.from_numpy(Xoh_tr), torch.from_numpy(Sid_tr),
                              torch.from_numpy(Ytr))
    val_ds = TensorDataset(torch.from_numpy(Xm_v), torch.from_numpy(Xcn_v),
                            torch.from_numpy(Xoh_v), torch.from_numpy(Sid_v),
                            torch.from_numpy(Yv))
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=2048)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xm, xc, xo, sid, y in train_loader:
            xm, xc, xo, sid, y = (xm.to(device), xc.to(device), xo.to(device),
                                   sid.to(device), y.to(device))
            mu, sig = model(xm, xc, xo, sid)
            loss = loss_combined(y, mu, sig)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()
        model.eval()
        v_mae = 0; nv = 0
        with torch.no_grad():
            for xm, xc, xo, sid, y in val_loader:
                xm, xc, xo, sid, y = (xm.to(device), xc.to(device), xo.to(device),
                                       sid.to(device), y.to(device))
                mu, sig = model(xm, xc, xo, sid)
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

    mu_v, sig_v = predict(model, Xm_v, Xcn_v, Xoh_v, Sid_v, Yv, device)
    mu_te, sig_te = predict(model, Xm_te, Xcn_te, Xoh_te, Sid_te, Yte, device)

    cap_te = test_df["site_capacity_kw"].values
    sites_te = test_df["site"].values
    dt_te = test_df["datetime_kst"].values
    m = eval_metrics(Yte, mu_te, sig_te, cap_te)
    p = portfolio_nmae(Yte, mu_te, sig_te, sites_te, dt_te, cap_te)
    m.update(p)
    print(f"  Test: NMAE {m['nmae']:.3f}% / Port {m['port_nmae']:.3f}% / "
          f"Cov80 {m['cov80']:.1f}% / NLL {m['nll']:.4f}", flush=True)

    out = out_dir / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "datetime_kst": val_df["datetime_kst"].values, "site": val_df["site"].values,
        "site_capacity_kw": val_df["site_capacity_kw"].values,
        "cf": Yv, "pred_mu": mu_v, "pred_sigma": sig_v,
    }).to_parquet(out / "val_predictions.parquet", index=False)
    pd.DataFrame({
        "datetime_kst": dt_te, "site": sites_te, "site_capacity_kw": cap_te,
        "cf": Yte, "pred_mu": mu_te, "pred_sigma": sig_te,
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
    p.add_argument("--variant", choices=["emb", "both"], required=False, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()

    if args.aggregate:
        if not args.variant:
            raise SystemExit("--aggregate 사용 시 --variant 필요")
        aggregate(args.variant); return
    if not args.variant:
        raise SystemExit("--variant {emb,both} 필요")

    out_dir = out_dir_for(args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS
    print("=" * 70)
    print(f"Step 2 — Site embedding | variant={args.variant} | seeds={seeds}")
    print(f"  batch={BATCH_SIZE}, lr={LR}, max_ep={MAX_EPOCHS}, "
          f"patience={PATIENCE}, loss=MAE+{ALPHA_NLL}·NLL")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data()
    all_sites = sorted(df["site"].unique())

    per_seed = []
    for seed in seeds:
        m = train_one(args.variant, seed, df, all_sites, device, out_dir)
        m["seed"] = seed
        per_seed.append(m)
    pd.DataFrame(per_seed).to_csv(
        out_dir / f"per_seed_summary_{'_'.join(map(str, seeds))}.csv", index=False)


if __name__ == "__main__":
    main()
