"""Phase 1 v3 — outage-masked retrain (실험 A, plan/active/lng/plan.md outage_v3.5).

기존 train_resmlp_adaln_v2_ensemble.py 와 동일한 hyperparam, seed, architecture.
유일한 차이: training set 에서 outage_v3.5 mask (data/processed/phase1_outage_mask.parquet)
            에 있는 (datetime_kst, site) row 제거 후 학습.

OUT_DIR: pv/experiments/resmlp_adaln_v3_outage_masked/
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
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/models"))

from train_resmlp_adaln import (
    ResMLPAdaLN, MAIN_FEATURES, COND_NUM_FEATURES,
    TARGET, TRAIN_END, VAL_END, gaussian_nll,
)
from train_resmlp_adaln_v2_ensemble import (
    BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, GRAD_CLIP, ALPHA_NLL,
    set_all_seeds, loss_combined, build, predict, eval_metrics,
    portfolio_nmae,
)

SEEDS = [42, 123, 7, 202, 999]
OUT_DIR = ROOT / "pv/experiments/resmlp_adaln_v3_outage_masked"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_PATH = ROOT / "data/processed/phase1_outage_mask.parquet"


def train_one(seed, df_full, mask_keys, all_sites, device):
    print(f"\n{'='*70}")
    print(f"[Seed {seed}] outage-masked retrain")
    print(f"{'='*70}")
    set_all_seeds(seed)

    train_df = df_full[df_full.datetime_kst < TRAIN_END].copy()
    val_df   = df_full[(df_full.datetime_kst >= TRAIN_END) & (df_full.datetime_kst < VAL_END)].copy()
    test_df  = df_full[df_full.datetime_kst >= VAL_END].copy()

    # Mask
    n_train_b = len(train_df); n_val_b = len(val_df)
    train_df["_key"] = list(zip(train_df.datetime_kst, train_df.site))
    val_df["_key"]   = list(zip(val_df.datetime_kst, val_df.site))
    train_df = train_df[~train_df["_key"].isin(mask_keys)].drop(columns="_key")
    val_df_clean = val_df[~val_df["_key"].isin(mask_keys)].drop(columns="_key")
    val_df = val_df.drop(columns="_key")
    print(f"  train: {n_train_b} -> {len(train_df)} ({n_train_b - len(train_df)} masked)")
    print(f"  val:   {n_val_b} -> {len(val_df_clean)} (clean for val_mae tracking)")
    print(f"  test:  {len(test_df)} (unchanged, raw eval)")

    Xm_tr, Xc_tr, Ytr = build(train_df, all_sites)
    Xm_v, Xc_v, Yv    = build(val_df_clean, all_sites)
    Xm_te, Xc_te, Yte = build(test_df, all_sites)

    model = ResMLPAdaLN(Xm_tr.shape[1], Xc_tr.shape[1]).to(device)

    train_ds = TensorDataset(torch.from_numpy(Xm_tr), torch.from_numpy(Xc_tr), torch.from_numpy(Ytr))
    val_ds   = TensorDataset(torch.from_numpy(Xm_v),  torch.from_numpy(Xc_v),  torch.from_numpy(Yv))
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader   = DataLoader(val_ds, batch_size=2048)

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

    # Save full val (raw) + test predictions
    Xm_v_full, Xc_v_full, Yv_full = build(val_df, all_sites)
    mu_v, sig_v   = predict(model, Xm_v_full, Xc_v_full, Yv_full, device)
    mu_te, sig_te = predict(model, Xm_te, Xc_te, Yte, device)

    cap_te = test_df["site_capacity_kw"].values
    sites_te = test_df["site"].values
    dt_te = test_df["datetime_kst"].values
    m = eval_metrics(Yte, mu_te, sig_te, cap_te)
    p = portfolio_nmae(Yte, mu_te, sig_te, sites_te, dt_te, cap_te)
    m.update(p)
    print(f"  Test (raw): NMAE {m['nmae']:.3f}% / Port {m['port_nmae']:.3f}% / "
          f"Cov80 {m['cov80']:.1f}% / NLL {m['nll']:.4f}", flush=True)

    out = OUT_DIR / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "datetime_kst": val_df["datetime_kst"].values, "site": val_df["site"].values,
        "site_capacity_kw": val_df["site_capacity_kw"].values,
        "cf": Yv_full, "pred_mu": mu_v, "pred_sigma": sig_v,
    }).to_parquet(out / "val_predictions.parquet", index=False)
    pd.DataFrame({
        "datetime_kst": dt_te, "site": sites_te, "site_capacity_kw": cap_te,
        "cf": Yte, "pred_mu": mu_te, "pred_sigma": sig_te,
    }).to_parquet(out / "test_predictions.parquet", index=False)
    return m


def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    df["datetime_kst"] = pd.to_datetime(df["datetime_kst"])
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    mask_df = pd.read_parquet(MASK_PATH)
    mask_df["datetime_kst"] = pd.to_datetime(mask_df.datetime_kst)
    mask_keys = set(zip(mask_df.datetime_kst, mask_df.site))
    print(f"loaded mask: {len(mask_keys)} (datetime, site) rows")

    df = load_data()
    all_sites = sorted(df["site"].unique())
    print(f"total: {len(df)}, sites: {all_sites}")

    seeds = [args.seed] if args.seed else SEEDS
    results = []
    for s in seeds:
        m = train_one(s, df, mask_keys, all_sites, device)
        m["seed"] = s
        results.append(m)

    print("\n=== Summary (test 2025, raw eval) ===")
    summary = pd.DataFrame(results)
    print(summary[["seed","nmae","port_nmae","cov80","cov95","nll"]].round(3).to_string(index=False))
    summary.to_csv(OUT_DIR / "per_seed_summary.csv", index=False)
    print(f"\n저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
