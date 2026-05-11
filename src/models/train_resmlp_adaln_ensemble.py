"""Step 1 — ResMLP+AdaLN ensemble baseline.

5 seeds (42, 123, 7, 202, 999) 동일 설정 학습.
각 seed별 val + test predictions 저장.

Ensemble:
  μ_ens = mean(μ_k)
  σ_ens² = mean(σ_k²) + var(μ_k)   (law of total variance)
            ↑ aleatoric    ↑ epistemic

Output:
  pv/experiments/resmlp_adaln_ensemble/
    seed_{N}/test_predictions.parquet, val_predictions.parquet
    ensemble_test.parquet
    summary.csv
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
sys.path.insert(0, str(ROOT / "src/models"))

from train_resmlp_adaln import (
    ResMLPAdaLN, MAIN_FEATURES, COND_NUM_FEATURES,
    TARGET, TRAIN_END, VAL_END, DIM, N_BLOCKS, LR, BATCH_SIZE,
    MAX_EPOCHS, PATIENCE, gaussian_nll,
)

SEEDS = [42, 123, 7, 202, 999]
OUT_DIR = ROOT / "pv/experiments/resmlp_adaln_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 2)
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
    out = df.groupby("site").apply(lambda g: pd.Series({
        "n": len(g),
        "nmae": (g.err * g.cap).sum() / g.cap.sum() * 100,
        "bias": ((g.mu - g.y) * g.cap).sum() / g.cap.sum() * 100,
        "cov80": g.in_80.mean() * 100,
        "cov95": g.in_95.mean() * 100,
    }), include_groups=False).reset_index()
    return out


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
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xm, xc, y in train_loader:
            xm, xc, y = xm.to(device), xc.to(device), y.to(device)
            mu, sig = model(xm, xc)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
        model.eval()
        v = 0; nv = 0
        with torch.no_grad():
            for xm, xc, y in val_loader:
                xm, xc, y = xm.to(device), xc.to(device), y.to(device)
                mu, sig = model(xm, xc)
                v += gaussian_nll(y, mu, sig).mean().item() * len(y); nv += len(y)
        v /= nv
        if v < best:
            best = v; bad = 0
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
        if ep % 20 == 0:
            print(f"  ep {ep:>3}: val_loss {v:.4f}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)

    # Inference
    mu_v, sig_v = predict(model, Xm_v, Xc_v, Yv, device)
    mu_te, sig_te = predict(model, Xm_te, Xc_te, Yte, device)

    cap_te = test_df["site_capacity_kw"].values
    sites_te = test_df["site"].values
    m = eval_metrics(Yte, mu_te, sig_te, cap_te)
    print(f"  Test: NMAE {m['nmae']:.3f}% / Cov80 {m['cov80']:.1f}% / Cov95 {m['cov95']:.1f}% / NLL {m['nll']:.4f}",
          flush=True)

    # Save
    out = OUT_DIR / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "datetime_kst": val_df["datetime_kst"].values,
        "site": val_df["site"].values,
        "site_capacity_kw": val_df["site_capacity_kw"].values,
        "cf": Yv, "pred_mu": mu_v, "pred_sigma": sig_v,
    }).to_parquet(out / "val_predictions.parquet", index=False)
    pd.DataFrame({
        "datetime_kst": test_df["datetime_kst"].values,
        "site": test_df["site"].values,
        "site_capacity_kw": test_df["site_capacity_kw"].values,
        "cf": Yte, "pred_mu": mu_te, "pred_sigma": sig_te,
    }).to_parquet(out / "test_predictions.parquet", index=False)
    return m


def ensemble_combine():
    """Load all seeds' test predictions, compute ensemble."""
    dfs = []
    for seed in SEEDS:
        p = OUT_DIR / f"seed_{seed}/test_predictions.parquet"
        df = pd.read_parquet(p)
        df["seed"] = seed
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # Ensemble per (datetime, site)
    grp = all_df.groupby(["datetime_kst", "site"])
    ens = grp.agg(
        cf=("cf", "first"),
        site_capacity_kw=("site_capacity_kw", "first"),
        mu_mean=("pred_mu", "mean"),
        mu_var=("pred_mu", "var"),
        sigma_sq_mean=("pred_sigma", lambda s: (s ** 2).mean()),
        n_seeds=("pred_mu", "size"),
    ).reset_index()
    # Total variance: aleatoric + epistemic
    ens["sigma_total"] = np.sqrt(ens["sigma_sq_mean"] + ens["mu_var"])
    ens["sigma_aleatoric"] = np.sqrt(ens["sigma_sq_mean"])
    ens["sigma_epistemic"] = np.sqrt(ens["mu_var"])
    return ens


def main():
    print("=" * 70)
    print(f"Step 1 — ResMLP+AdaLN ensemble baseline (seeds: {SEEDS})")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data()
    all_sites = sorted(df["site"].unique())

    per_seed_metrics = []
    for seed in SEEDS:
        m = train_one(seed, df, all_sites, device)
        m["seed"] = seed
        per_seed_metrics.append(m)

    seed_df = pd.DataFrame(per_seed_metrics)
    print("\n" + "=" * 70)
    print("Per-seed (test)")
    print("=" * 70)
    print(seed_df.round(3).to_string(index=False))
    print(f"\n  Mean ± Std:")
    for col in ["nmae", "bias", "cov80", "cov95", "nll"]:
        v = seed_df[col].values
        print(f"    {col:<10} {v.mean():>7.3f} ± {v.std(ddof=1):>5.3f}")

    # ===== Ensemble =====
    print("\n" + "=" * 70)
    print("Ensemble baseline (mu = mean, σ² = aleatoric + epistemic)")
    print("=" * 70)
    ens = ensemble_combine()
    cap = ens["site_capacity_kw"].values
    y = ens["cf"].values
    mu = ens["mu_mean"].values
    sigma_total = ens["sigma_total"].values
    sigma_aleatoric = ens["sigma_aleatoric"].values

    # Total uncertainty
    m_total = eval_metrics(y, mu, sigma_total, cap)
    m_aleatoric = eval_metrics(y, mu, sigma_aleatoric, cap)

    print(f"\n  Ensemble (total σ): NMAE {m_total['nmae']:.3f}% / "
          f"bias {m_total['bias']:+.3f}% / Cov80 {m_total['cov80']:.1f}% / "
          f"Cov95 {m_total['cov95']:.1f}% / NLL {m_total['nll']:.4f}")
    print(f"  Ensemble (aleatoric only σ): NMAE {m_aleatoric['nmae']:.3f}% / "
          f"Cov80 {m_aleatoric['cov80']:.1f}% / Cov95 {m_aleatoric['cov95']:.1f}%")

    print("\n  Per-site (ensemble, total σ):")
    site_ens = per_site_metrics(y, mu, sigma_total, ens["site"].values, cap)
    print(site_ens.round(2).to_string(index=False))

    # Save
    ens.to_parquet(OUT_DIR / "ensemble_test.parquet", index=False)
    seed_df.to_csv(OUT_DIR / "per_seed_summary.csv", index=False)
    site_ens.to_csv(OUT_DIR / "ensemble_per_site.csv", index=False)

    print(f"\n저장: {OUT_DIR}")
    print(f"\n=== 기준점 (Ensemble baseline) ===")
    print(f"  NMAE {m_total['nmae']:.3f}% / Port 추정 ~{m_total['nmae']*0.85:.2f}% / "
          f"Cov80 {m_total['cov80']:.1f}% / Cov95 {m_total['cov95']:.1f}%")
    print(f"  Per-site cov80 range: {site_ens['cov80'].min():.1f} ~ {site_ens['cov80'].max():.1f}%")


if __name__ == "__main__":
    main()
