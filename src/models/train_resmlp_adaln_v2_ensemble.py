"""Step 1 — ResMLP+AdaLN ensemble (UNIFIED config v2).

이후 모든 실험의 *공통 hyperparam*:
  batch=64, lr=7e-4, wd=1e-4, AdamW, CosineAnnealingLR(T_max=50),
  max_epochs=50, patience=8, grad_clip=1.0,
  loss = MAE + 0.2 * GaussianNLL
  val tracking metric = val_mae (point accuracy 우선)

5 seeds: 42, 123, 7, 202, 999.
새 ensemble baseline 생성 → 모든 후속 실험의 기준점.
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

# ===== UNIFIED CONFIG (앞으로 모든 실험 동일) =====
BATCH_SIZE = 64
LR = 7e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0
ALPHA_NLL = 0.20

SEEDS = [42, 123, 7, 202, 999]
OUT_DIR = ROOT / "pv/experiments/resmlp_adaln_v2_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    site_oh = np.zeros((len(df), len(all_sites)), dtype=np.float32)
    for i, s in enumerate(all_sites):
        site_oh[:, i] = (df["site"] == s).astype(np.float32)
    cond = np.concatenate([cond_num, site_oh], axis=1)
    y = df[TARGET].values.astype(np.float32)
    return main, cond, y


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


def train_one(seed, df, all_sites, device):
    print(f"\n{'='*70}")
    print(f"[Seed {seed}]")
    print(f"{'='*70}")
    set_all_seeds(seed)
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]

    Xm_tr, Xc_tr, Ytr = build(train_df, all_sites)
    Xm_v, Xc_v, Yv = build(val_df, all_sites)
    Xm_te, Xc_te, Yte = build(test_df, all_sites)

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

    out = OUT_DIR / f"seed_{seed}"
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


def ensemble_combine():
    dfs = []
    for seed in SEEDS:
        p = OUT_DIR / f"seed_{seed}/test_predictions.parquet"
        df = pd.read_parquet(p); df["seed"] = seed
        dfs.append(df)
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
    return ens


def run_seeds(seeds):
    print("=" * 70)
    print(f"Step 1 — ResMLP+AdaLN v2 ensemble (UNIFIED config, seeds: {seeds})")
    print(f"  batch={BATCH_SIZE}, lr={LR}, wd={WEIGHT_DECAY}, max_ep={MAX_EPOCHS}, "
          f"patience={PATIENCE}, loss=MAE+{ALPHA_NLL}·NLL")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data()
    all_sites = sorted(df["site"].unique())

    per_seed = []
    for seed in seeds:
        m = train_one(seed, df, all_sites, device)
        m["seed"] = seed
        per_seed.append(m)
        # incremental save (multi-process safe: 각 process 별 파일)
    pd.DataFrame(per_seed).to_csv(
        OUT_DIR / f"per_seed_summary_{'_'.join(map(str, seeds))}.csv", index=False)


def aggregate():
    """All saved seeds(per-seed parquet)를 모아 ensemble baseline 산출."""
    print("=" * 70)
    print("Aggregate — ensemble baseline (mu = mean, σ² = aleatoric + epistemic)")
    print("=" * 70)
    # 디스크에 저장된 모든 seed 디렉토리 자동 발견
    seed_dirs = sorted(OUT_DIR.glob("seed_*/test_predictions.parquet"))
    if not seed_dirs:
        print("저장된 seed 결과 없음."); return
    found_seeds = [int(p.parent.name.split("_")[-1]) for p in seed_dirs]
    print(f"  발견된 seeds: {found_seeds}")

    # per-seed test 메트릭
    per_seed = []
    for seed in found_seeds:
        df_te = pd.read_parquet(OUT_DIR / f"seed_{seed}/test_predictions.parquet")
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

    # ensemble
    dfs = []
    for seed in found_seeds:
        d = pd.read_parquet(OUT_DIR / f"seed_{seed}/test_predictions.parquet"); d["seed"] = seed
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

    ens.to_parquet(OUT_DIR / "ensemble_test.parquet", index=False)
    seed_df.to_csv(OUT_DIR / "per_seed_summary.csv", index=False)
    site_ens.to_csv(OUT_DIR / "ensemble_per_site.csv", index=False)

    print(f"\n=== 새 기준점 (UNIFIED config baseline) ===")
    print(f"  Site NMAE: {m_ens['nmae']:.3f}%")
    print(f"  Port NMAE: {p_ens['port_nmae']:.3f}%")
    print(f"  Cov80: {m_ens['cov80']:.1f}% / Cov95: {m_ens['cov95']:.1f}%")
    print(f"  Per-site cov80 range: {site_ens['cov80'].min():.1f} ~ {site_ens['cov80'].max():.1f}%")
    print(f"\n저장: {OUT_DIR}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default=None,
                   help="콤마 구분 seed 목록 (예: 42,123). 없으면 전체 SEEDS.")
    p.add_argument("--aggregate", action="store_true",
                   help="저장된 seed 결과만 모아 ensemble 메트릭 출력")
    args = p.parse_args()
    if args.aggregate:
        aggregate(); return
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS
    run_seeds(seeds)


if __name__ == "__main__":
    main()
