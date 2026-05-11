"""Step 8 — ResMLP+AdaLN v2 + delta/volatility feature 추가 (UNIFIED config 유지).

구조 변경 없음. 학습 설정 baseline과 100% 동일.
변경 사항: cond_num features 끝에 새 feature 1~2개 추가 + train set으로 z-score 표준화.

variant ∈ {d2, d2v3}:
  d2   : cond_num + [dsr_delta_2h]
  d2v3 : cond_num + [dsr_delta_2h, dsr_volatility_3h]

UNIFIED config (Step 1 baseline과 동일):
  batch=64, lr=7e-4, wd=1e-4, AdamW + CosineAnnealingLR(T_max=50),
  max_ep=50, patience=8, grad_clip=1.0, loss = MAE + 0.2*GaussianNLL,
  val tracking metric = val_mae.
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
    ResMLPAdaLN, MAIN_FEATURES, COND_NUM_FEATURES,
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
SEEDS_DEFAULT = [42, 123, 7]

VOL_WIN = 3  # ±3h centered std for volatility


def out_dir_for(variant):
    return ROOT / f"pv/experiments/resmlp_adaln_v2_delta_{variant}"


def variant_extra_feats(variant):
    if variant == "d2":
        return ["dsr_delta_2h"]
    elif variant == "d2v3":
        return ["dsr_delta_2h", "dsr_volatility_3h"]
    raise ValueError(variant)


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
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    # weather-only delta / volatility
    g = df.groupby("site", group_keys=False)
    df["dsr_delta_2h"] = g["dsr_mean"].diff(2).fillna(0)
    df["dsr_volatility_3h"] = g["dsr_mean"].transform(
        lambda s: s.rolling(2 * VOL_WIN + 1, center=True, min_periods=1).std()).fillna(0)
    return df


def build(df, all_sites, extra_feats, scalers=None):
    """Return main, cond, y. cond = [base_cond_num, extra_norm, site_oh]."""
    main = df[MAIN_FEATURES].values.astype(np.float32)
    cond_num = df[COND_NUM_FEATURES].values.astype(np.float32)
    extra_raw = df[extra_feats].values.astype(np.float32)
    if scalers is None:
        mu = extra_raw.mean(axis=0); sd = extra_raw.std(axis=0) + 1e-6
        scalers = (mu, sd)
    else:
        mu, sd = scalers
    extra_norm = (extra_raw - mu) / sd
    site_oh = np.zeros((len(df), len(all_sites)), dtype=np.float32)
    for i, s in enumerate(all_sites):
        site_oh[:, i] = (df["site"] == s).astype(np.float32)
    cond = np.concatenate([cond_num, extra_norm, site_oh], axis=1)
    y = df[TARGET].values.astype(np.float32)
    return main, cond, y, scalers


def predict(model, X_main, X_cond, Y, device):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_main), torch.from_numpy(X_cond), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=2048)
    mus, sigs = [], []
    with torch.no_grad():
        for xm, xc, _ in loader:
            mu, sig = model(xm.to(device), xc.to(device))
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
    extra_feats = variant_extra_feats(variant)
    print(f"\n{'='*70}")
    print(f"[Seed {seed}] variant={variant}  extra_feats={extra_feats}")
    print(f"{'='*70}")
    set_all_seeds(seed)
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]

    Xm_tr, Xc_tr, Ytr, scalers = build(train_df, all_sites, extra_feats)
    print(f"  extra-feat scalers (mean, std): {[(round(m,3), round(s,3)) for m,s in zip(*scalers)]}")
    Xm_v, Xc_v, Yv, _ = build(val_df, all_sites, extra_feats, scalers)
    Xm_te, Xc_te, Yte, _ = build(test_df, all_sites, extra_feats, scalers)
    print(f"  cond dim: {Xc_tr.shape[1]} (= {len(COND_NUM_FEATURES)} base + {len(extra_feats)} extra + {len(all_sites)} sites)")

    n_main = Xm_tr.shape[1]; n_cond = Xc_tr.shape[1]
    model = ResMLPAdaLN(n_main, n_cond).to(device)

    train_ds = TensorDataset(torch.from_numpy(Xm_tr), torch.from_numpy(Xc_tr), torch.from_numpy(Ytr))
    val_ds = TensorDataset(torch.from_numpy(Xm_v), torch.from_numpy(Xc_v), torch.from_numpy(Yv))
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=2048)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xm, xc, y in train_loader:
            xm, xc, y = xm.to(device), xc.to(device), y.to(device)
            mu, sig = model(xm, xc)
            loss = loss_combined(y, mu, sig)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()
        model.eval()
        v_mae = 0; nv = 0
        with torch.no_grad():
            for xm, xc, y in val_loader:
                xm, xc, y = xm.to(device), xc.to(device), y.to(device)
                mu, sig = model(xm, xc)
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

    mu_v, sig_v = predict(model, Xm_v, Xc_v, Yv, device)
    mu_te, sig_te = predict(model, Xm_te, Xc_te, Yte, device)

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
        "datetime_kst": dt_te, "site": sites_te, "site_capacity_kw": cap_te,
        "cf": Yte, "pred_mu": mu_te, "pred_sigma": sig_te,
    }).to_parquet(out / "test_predictions.parquet", index=False)
    pd.DataFrame({
        "datetime_kst": val_df["datetime_kst"].values, "site": val_df["site"].values,
        "site_capacity_kw": val_df["site_capacity_kw"].values,
        "cf": Yv, "pred_mu": mu_v, "pred_sigma": sig_v,
    }).to_parquet(out / "val_predictions.parquet", index=False)
    return m


def aggregate(variant):
    out_dir = out_dir_for(variant)
    print("=" * 70)
    print(f"Aggregate — variant={variant}")
    print("=" * 70)
    seed_dirs = sorted(out_dir.glob("seed_*/test_predictions.parquet"))
    if not seed_dirs:
        print("저장된 seed 결과 없음."); return
    found = sorted(int(p.parent.name.split("_")[-1]) for p in seed_dirs)
    print(f"  발견된 seeds: {found}")

    rows = []
    for seed in found:
        df_te = pd.read_parquet(out_dir / f"seed_{seed}/test_predictions.parquet")
        cap = df_te["site_capacity_kw"].values
        y = df_te["cf"].values; mu = df_te["pred_mu"].values; sig = df_te["pred_sigma"].values
        m = eval_metrics(y, mu, sig, cap)
        m.update(portfolio_nmae(y, mu, sig, df_te["site"].values, df_te["datetime_kst"].values, cap))
        m["seed"] = seed; rows.append(m)
    seed_df = pd.DataFrame(rows)
    print("\nPer-seed:")
    print(seed_df.round(3).to_string(index=False))
    print(f"\n  Mean ± Std:")
    for col in ["nmae", "port_nmae", "cov80", "cov95", "nll"]:
        v = seed_df[col].values
        print(f"    {col:<12} {v.mean():>7.3f} ± {v.std(ddof=1):>5.3f}")

    dfs = []
    for seed in found:
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
    site_ens = per_site_metrics(y, mu, sigma, ens["site"].values, cap)
    print("\n  Per-site:")
    print(site_ens.round(2).to_string(index=False))
    ens.to_parquet(out_dir / "ensemble_test.parquet", index=False)
    seed_df.to_csv(out_dir / "per_seed_summary.csv", index=False)
    site_ens.to_csv(out_dir / "ensemble_per_site.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["d2", "d2v3"], required=False, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()

    if args.aggregate:
        if not args.variant: raise SystemExit("--aggregate 사용 시 --variant 필요")
        aggregate(args.variant); return
    if not args.variant: raise SystemExit("--variant {d2,d2v3} 필요")

    out_dir = out_dir_for(args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS_DEFAULT
    print("=" * 70)
    print(f"Step 8 delta | variant={args.variant} | extra={variant_extra_feats(args.variant)} | seeds={seeds}")
    print(f"  batch={BATCH_SIZE}, lr={LR}, max_ep={MAX_EPOCHS}, "
          f"patience={PATIENCE}, loss=MAE+{ALPHA_NLL}·NLL")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data()
    all_sites = sorted(df["site"].unique())

    rows = []
    for seed in seeds:
        m = train_one(args.variant, seed, df, all_sites, device, out_dir)
        m["seed"] = seed
        rows.append(m)
    pd.DataFrame(rows).to_csv(
        out_dir / f"per_seed_summary_{'_'.join(map(str, seeds))}.csv", index=False)


if __name__ == "__main__":
    main()
