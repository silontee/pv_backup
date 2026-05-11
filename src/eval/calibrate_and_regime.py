"""Step 0 + 1 + 2: Post-hoc calibration + regime redefinition.

Step 0: Post-hoc calibration on ResMLP+AdaLN
  - Retrain ResMLP+AdaLN (seed 42) and save val(2024) + test(2025) predictions
  - Fit isotonic on val: residual_corrector(pred_mu) → bias 보정
  - Fit per-site σ scaling on val: σ × q_{0.8}(|z_site|)/1.28
  - Apply to test → metrics before/after

Step 1: Regime re-definition
  - Old: clear/cloudy/transition (단순 dc10Tca + dsr 1D)
  - New: 2D (dc10Tca + dsr) bin-based + 가능 시 zenith
  - Or error-driven binning (residual std 기준)
  - Compare: sample count / NMAE / bias / Cov80

Step 2: Per-new-regime σ scaling
  - vs old regime calibration

출력: pv/experiments/calibration/
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
from sklearn.isotonic import IsotonicRegression

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "pv/experiments/calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src/models"))
from train_resmlp_adaln import (
    ResMLPAdaLN, MAIN_FEATURES, COND_NUM_FEATURES,
    TARGET, TRAIN_END, VAL_END, DIM, N_BLOCKS, LR, BATCH_SIZE,
    MAX_EPOCHS, PATIENCE, gaussian_nll,
)


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


def metrics_table(y, mu, sigma, cap):
    err = np.abs(y - mu)
    nmae = (err * cap).sum() / cap.sum() * 100
    bias = ((mu - y) * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu - 1.282 * sigma) & (y <= mu + 1.282 * sigma)).mean() * 100
    cov95 = ((y >= mu - 1.96 * sigma) & (y <= mu + 1.96 * sigma)).mean() * 100
    return {"nmae": nmae, "bias_signed": bias, "cov80": cov80, "cov95": cov95}


def per_site_metrics(y, mu, sigma, sites, cap, label="Per-site"):
    df = pd.DataFrame({"site": sites, "y": y, "mu": mu, "sigma": sigma, "cap": cap})
    df["err"] = (df.y - df.mu).abs()
    df["in_80"] = ((df.y >= df.mu - 1.282 * df.sigma) & (df.y <= df.mu + 1.282 * df.sigma)).astype(int)
    df["in_95"] = ((df.y >= df.mu - 1.96 * df.sigma) & (df.y <= df.mu + 1.96 * df.sigma)).astype(int)
    out = df.groupby("site").apply(lambda g: pd.Series({
        "n": len(g),
        "nmae": (g.err * g.cap).sum() / g.cap.sum() * 100,
        "bias_signed": ((g.mu - g.y) * g.cap).sum() / g.cap.sum() * 100,
        "cov80": g.in_80.mean() * 100,
        "cov95": g.in_95.mean() * 100,
    }), include_groups=False).reset_index()
    return out


# ===== Regime definitions =====

def regime_old(df):
    """Old hand-crafted (1D-ish): clear/cloudy/transition."""
    cond_clear = (df["dsr_mean"] > 400) & (df["dc10Tca"] < 3)
    cond_cloudy = (df["dsr_mean"] < 200) | (df["dc10Tca"] >= 7)
    out = np.full(len(df), "transition", dtype=object)
    out[cond_clear] = "clear"
    out[cond_cloudy] = "cloudy"
    return out


def regime_2d(df):
    """New 2D rule: cloud × dsr 결합.

    Categories (error breakdown 기반):
      A. clear_full   : dsr >= 400, cloud < 1   (clear, predictable)
      B. clear_dim    : dsr 100-400, cloud < 1  (low sun clear, edge)
      C. cloudy_full  : dsr < 200, cloud >= 7   (heavy cloud, predictable low cf)
      D. partial_low  : dsr 100-400, cloud 1-7  (transition, low DSR — *worst*)
      E. partial_high : dsr >= 400, cloud 1-7   (transition, high DSR — *worst 2*)
      F. unstable     : dsr 200-400, cloud >= 7 (heavy cloud high DSR mismatch)
      G. dawn         : dsr < 100  (very low light, edge cases)
    """
    out = np.full(len(df), "other", dtype=object)
    dsr = df["dsr_mean"].values
    cld = df["dc10Tca"].values

    out[(dsr >= 400) & (cld < 1)] = "clear_full"
    out[(dsr >= 100) & (dsr < 400) & (cld < 1)] = "clear_dim"
    out[(dsr < 200) & (cld >= 7)] = "cloudy_full"
    out[(dsr >= 100) & (dsr < 400) & (cld >= 1) & (cld < 7)] = "partial_low"
    out[(dsr >= 400) & (cld >= 1) & (cld < 7)] = "partial_high"
    out[(dsr >= 200) & (dsr < 400) & (cld >= 7)] = "unstable"
    out[dsr < 100] = "dawn"
    return out


def regime_error_driven(df, residuals_val, n_bins=6):
    """Error-driven: bin by (cloud × dsr) 2D, top NMAE bins separated.

    Use val residuals to identify error-homogeneous regions.
    """
    df = df.copy()
    df["abs_res"] = np.abs(residuals_val)
    df["cloud_bin"] = pd.cut(df["dc10Tca"], bins=[-0.1, 1, 3, 5, 7, 9, 10.1], labels=False)
    df["dsr_bin"] = pd.cut(df["dsr_mean"], bins=[0, 150, 300, 500, 700, 1100], labels=False)
    df["cell"] = df["cloud_bin"].astype(str) + "_" + df["dsr_bin"].astype(str)
    # MAE per cell
    cell_mae = df.groupby("cell")["abs_res"].mean().sort_values()
    # Quantile binning of cells
    qs = np.quantile(cell_mae, np.linspace(0, 1, n_bins + 1))
    cell_to_regime = {}
    for cell, mae in cell_mae.items():
        # find which quantile bin
        idx = int(np.searchsorted(qs[1:-1], mae))
        cell_to_regime[cell] = f"e{idx}"
    # Apply
    out = df["cell"].map(cell_to_regime).fillna("e_unknown").values
    return out, cell_mae, qs


# ===== Calibration =====

def fit_isotonic(pred_mu_val, y_val):
    """Fit isotonic regression: y = f(pred_mu).
    f is monotonic. Use as: pred_mu_corrected = f(pred_mu)."""
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(pred_mu_val, y_val)
    return iso


def fit_sigma_scale(z_abs, target_q=0.8, target_z=1.282):
    """Empirical |z| quantile based scaling: σ_cal = σ × q(|z|, target) / target_z."""
    q = np.quantile(z_abs, target_q)
    return q / target_z


def per_site_sigma_scale(y_val, mu_val, sig_val, sites_val, target_q=0.8):
    """Per-site σ scaling (split conformal style)."""
    target_z = 1.282 if target_q == 0.8 else 1.96
    z_abs = np.abs(y_val - mu_val) / np.maximum(sig_val, 1e-3)
    scales = {}
    for s in np.unique(sites_val):
        idx = sites_val == s
        if idx.sum() < 30:
            scales[s] = 1.0
        else:
            scales[s] = fit_sigma_scale(z_abs[idx], target_q, target_z)
    return scales


def per_regime_sigma_scale(y_val, mu_val, sig_val, regime_val, target_q=0.8):
    target_z = 1.282
    z_abs = np.abs(y_val - mu_val) / np.maximum(sig_val, 1e-3)
    scales = {}
    for r in np.unique(regime_val):
        idx = regime_val == r
        if idx.sum() < 30:
            scales[r] = 1.0
        else:
            scales[r] = fit_sigma_scale(z_abs[idx], target_q, target_z)
    return scales


def main():
    print("=" * 70)
    print("Calibration + Regime Redefinition")
    print("=" * 70)
    set_all_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ===== Train ResMLP+AdaLN, get val + test predictions =====
    print("\n[Train] ResMLP+AdaLN (seed 42)...")
    df = load_data()
    all_sites = sorted(df["site"].unique())
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]

    Xm_tr, Xc_tr, Ytr = build(train_df, all_sites)
    Xm_v, Xc_v, Yv = build(val_df, all_sites)
    Xm_te, Xc_te, Yte = build(test_df, all_sites)
    print(f"  train {len(Ytr):,}, val {len(Yv):,}, test {len(Yte):,}")

    n_main = Xm_tr.shape[1]; n_cond = Xc_tr.shape[1]
    model = ResMLPAdaLN(n_main, n_cond).to(device)

    # train
    train_ds = TensorDataset(torch.from_numpy(Xm_tr), torch.from_numpy(Xc_tr), torch.from_numpy(Ytr))
    val_ds = TensorDataset(torch.from_numpy(Xm_v), torch.from_numpy(Xc_v), torch.from_numpy(Yv))
    g = torch.Generator(); g.manual_seed(42)
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
        if ep % 10 == 0:
            print(f"  ep {ep:>3}: val_loss {v:.4f}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)

    # Inference on val + test
    mu_v, sig_v = predict(model, Xm_v, Xc_v, Yv, device)
    mu_te, sig_te = predict(model, Xm_te, Xc_te, Yte, device)

    cap_v = val_df["site_capacity_kw"].values
    cap_te = test_df["site_capacity_kw"].values
    sites_v = val_df["site"].values
    sites_te = test_df["site"].values

    # Baseline metrics (no calibration)
    print("\n" + "=" * 70)
    print("Step 0 — Baseline (no calibration)")
    print("=" * 70)
    base_te = metrics_table(Yte, mu_te, sig_te, cap_te)
    print(f"  Test: NMAE {base_te['nmae']:.3f}% / bias {base_te['bias_signed']:+.3f}% / "
          f"Cov80 {base_te['cov80']:.1f}% / Cov95 {base_te['cov95']:.1f}%")
    site_base = per_site_metrics(Yte, mu_te, sig_te, sites_te, cap_te)
    print("\n  Per-site (test, baseline):")
    print(site_base.round(2).to_string(index=False))

    # ===== Step 0a: Isotonic on val =====
    print("\n" + "=" * 70)
    print("Step 0a — Isotonic regression on pred_mu")
    print("=" * 70)
    iso = fit_isotonic(mu_v, Yv)
    mu_te_iso = iso.predict(mu_te).astype(np.float32)
    iso_te = metrics_table(Yte, mu_te_iso, sig_te, cap_te)
    print(f"  Test (isotonic): NMAE {iso_te['nmae']:.3f}% / bias {iso_te['bias_signed']:+.3f}% / "
          f"Cov80 {iso_te['cov80']:.1f}%")
    print(f"  Δ NMAE: {iso_te['nmae'] - base_te['nmae']:+.3f}%p, "
          f"Δ bias: {iso_te['bias_signed'] - base_te['bias_signed']:+.3f}%")

    # ===== Step 0b: Per-site σ scaling =====
    print("\n" + "=" * 70)
    print("Step 0b — Per-site σ scaling (after isotonic)")
    print("=" * 70)
    site_scales = per_site_sigma_scale(Yv, iso.predict(mu_v), sig_v, sites_v)
    print("  Per-site σ scale (val→test):")
    for s in sorted(site_scales.keys()):
        print(f"    {s:<14} × {site_scales[s]:.3f}")

    # apply
    sig_te_site = sig_te.copy()
    for s, sc in site_scales.items():
        sig_te_site[sites_te == s] *= sc
    iso_site_te = metrics_table(Yte, mu_te_iso, sig_te_site, cap_te)
    print(f"\n  Test (iso + site σ): NMAE {iso_site_te['nmae']:.3f}% / "
          f"Cov80 {iso_site_te['cov80']:.1f}% / Cov95 {iso_site_te['cov95']:.1f}%")

    site_iso_site = per_site_metrics(Yte, mu_te_iso, sig_te_site, sites_te, cap_te)
    print("\n  Per-site (test, iso + site σ):")
    print(site_iso_site.round(2).to_string(index=False))

    # 광양항/예천 specific
    print("\n  ⭐ 광양항세방 + 예천 Cov80 회복:")
    for s in ["광양항세방", "예천"]:
        before = site_base[site_base.site == s].iloc[0]
        after = site_iso_site[site_iso_site.site == s].iloc[0]
        print(f"    {s}: cov80 {before['cov80']:.1f}% → {after['cov80']:.1f}%, "
              f"NMAE {before['nmae']:.2f}% → {after['nmae']:.2f}%")

    # ===== Step 1: Regime redefinition =====
    print("\n" + "=" * 70)
    print("Step 1 — Regime redefinition (old vs new 2D)")
    print("=" * 70)

    val_df = val_df.copy(); val_df["mu_raw"] = mu_v; val_df["sig_raw"] = sig_v
    val_df["mu_iso"] = iso.predict(mu_v); val_df["res"] = Yv - val_df["mu_iso"]
    test_df = test_df.copy(); test_df["mu_raw"] = mu_te; test_df["sig_raw"] = sig_te
    test_df["mu_iso"] = mu_te_iso; test_df["sig_iso_site"] = sig_te_site
    test_df["res"] = Yte - test_df["mu_iso"]

    # Old regime
    val_df["regime_old"] = regime_old(val_df)
    test_df["regime_old"] = regime_old(test_df)

    # New 2D regime
    val_df["regime_2d"] = regime_2d(val_df)
    test_df["regime_2d"] = regime_2d(test_df)

    # Error-driven
    val_resid = val_df["res"].values
    val_df["regime_err"], cell_mae, qs = regime_error_driven(val_df, val_resid, n_bins=6)
    # Apply same cell→regime mapping to test
    test_df_tmp = test_df.copy()
    test_df_tmp["abs_res"] = test_df_tmp["res"].abs()
    test_df_tmp["cloud_bin"] = pd.cut(test_df_tmp["dc10Tca"], bins=[-0.1, 1, 3, 5, 7, 9, 10.1], labels=False)
    test_df_tmp["dsr_bin"] = pd.cut(test_df_tmp["dsr_mean"], bins=[0, 150, 300, 500, 700, 1100], labels=False)
    test_df_tmp["cell"] = test_df_tmp["cloud_bin"].astype(str) + "_" + test_df_tmp["dsr_bin"].astype(str)
    # use val mapping
    val_cell_to_regime = {}
    for cell, mae in cell_mae.items():
        idx = int(np.searchsorted(qs[1:-1], mae))
        val_cell_to_regime[cell] = f"e{idx}"
    test_df["regime_err"] = test_df_tmp["cell"].map(val_cell_to_regime).fillna("e_unknown").values

    # Compare regimes on test
    def regime_summary(df, regime_col, mu_col, sig_col):
        rows = []
        cap = df["site_capacity_kw"].values
        y = df["cf"].values
        mu = df[mu_col].values
        sig = df[sig_col].values
        for r in sorted(df[regime_col].unique()):
            idx = df[regime_col].values == r
            if idx.sum() == 0:
                continue
            err = np.abs(y[idx] - mu[idx])
            cap_r = cap[idx]
            nmae = (err * cap_r).sum() / cap_r.sum() * 100
            bias = ((mu[idx] - y[idx]) * cap_r).sum() / cap_r.sum() * 100
            cov80 = ((y[idx] >= mu[idx] - 1.282 * sig[idx]) &
                     (y[idx] <= mu[idx] + 1.282 * sig[idx])).mean() * 100
            cov95 = ((y[idx] >= mu[idx] - 1.96 * sig[idx]) &
                     (y[idx] <= mu[idx] + 1.96 * sig[idx])).mean() * 100
            rows.append({"regime": r, "n": int(idx.sum()),
                         "nmae": nmae, "bias_signed": bias, "cov80": cov80, "cov95": cov95})
        return pd.DataFrame(rows)

    print("\n  Old regime (test, after iso+site σ):")
    old_sum = regime_summary(test_df, "regime_old", "mu_iso", "sig_iso_site")
    print(old_sum.round(2).to_string(index=False))

    print("\n  New 2D regime (test, after iso+site σ):")
    new_sum = regime_summary(test_df, "regime_2d", "mu_iso", "sig_iso_site")
    print(new_sum.round(2).to_string(index=False))

    print("\n  Error-driven regime (test):")
    err_sum = regime_summary(test_df, "regime_err", "mu_iso", "sig_iso_site")
    print(err_sum.round(2).to_string(index=False))

    # ===== Step 2: Per-new-regime σ scaling =====
    print("\n" + "=" * 70)
    print("Step 2 — Per-regime σ scaling (new 2D)")
    print("=" * 70)
    # σ scale fit on val (after iso+site σ)
    val_df["sig_iso_site"] = val_df["sig_raw"].copy()
    for s, sc in site_scales.items():
        val_df.loc[val_df["site"] == s, "sig_iso_site"] *= sc
    regime_scales_2d = per_regime_sigma_scale(
        Yv, val_df["mu_iso"].values, val_df["sig_iso_site"].values,
        val_df["regime_2d"].values, target_q=0.8,
    )
    print("\n  Per-regime σ scale (new 2D, on top of site σ):")
    for r, sc in sorted(regime_scales_2d.items()):
        print(f"    {r:<14} × {sc:.3f}")

    # apply
    sig_te_full = test_df["sig_iso_site"].values.copy()
    for r, sc in regime_scales_2d.items():
        sig_te_full[test_df["regime_2d"].values == r] *= sc

    full_te = metrics_table(Yte, test_df["mu_iso"].values, sig_te_full, cap_te)
    print(f"\n  Test (iso + site σ + 2D regime σ): "
          f"NMAE {full_te['nmae']:.3f}% / Cov80 {full_te['cov80']:.1f}% / Cov95 {full_te['cov95']:.1f}%")

    # Compare per-regime cov
    test_df["sig_full"] = sig_te_full
    print("\n  Per-2D-regime cov80 after full calibration:")
    final_sum = regime_summary(test_df, "regime_2d", "mu_iso", "sig_full")
    print(final_sum.round(2).to_string(index=False))

    # ===== Final summary =====
    print("\n" + "=" * 70)
    print("최종 비교")
    print("=" * 70)
    print(f"  {'Calibration step':<35} {'NMAE':>7} {'Bias':>8} {'Cov80':>7} {'Cov95':>7}")
    print(f"  {'(0) Baseline':<35} {base_te['nmae']:>6.2f}% {base_te['bias_signed']:>+7.2f}% "
          f"{base_te['cov80']:>6.1f}% {base_te['cov95']:>6.1f}%")
    print(f"  {'(0a) + isotonic':<35} {iso_te['nmae']:>6.2f}% {iso_te['bias_signed']:>+7.2f}% "
          f"{iso_te['cov80']:>6.1f}% {iso_te['cov95']:>6.1f}%")
    print(f"  {'(0b) + per-site σ':<35} {iso_site_te['nmae']:>6.2f}% {iso_site_te['bias_signed']:>+7.2f}% "
          f"{iso_site_te['cov80']:>6.1f}% {iso_site_te['cov95']:>6.1f}%")
    print(f"  {'(2)  + per-regime σ (2D)':<35} {full_te['nmae']:>6.2f}% {full_te['bias_signed']:>+7.2f}% "
          f"{full_te['cov80']:>6.1f}% {full_te['cov95']:>6.1f}%")

    # Save
    pd.DataFrame({"site": list(site_scales.keys()), "scale": list(site_scales.values())}).to_csv(
        OUT_DIR / "site_sigma_scales.csv", index=False)
    pd.DataFrame({"regime": list(regime_scales_2d.keys()), "scale": list(regime_scales_2d.values())}).to_csv(
        OUT_DIR / "regime_2d_sigma_scales.csv", index=False)
    test_df[["datetime_kst", "site", "site_capacity_kw", "cf",
             "mu_raw", "sig_raw", "mu_iso", "sig_iso_site", "sig_full",
             "regime_old", "regime_2d", "regime_err"]].to_parquet(
        OUT_DIR / "test_calibrated.parquet", index=False)
    print(f"\n저장: {OUT_DIR}")


if __name__ == "__main__":
    main()
