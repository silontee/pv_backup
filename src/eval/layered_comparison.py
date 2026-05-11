"""3-layer 복합 vs rolling baseline 종합 비교.

Setup:
  L1 (rolling baseline): ResMLP+AdaLN with current-hour weather  (cf history 안 씀)
  L2 (intraday point):    Hybrid = L1 + 1D-CNN local 6h cf history
  L3 (intraday distrib):  Bayesian reweight + B1/B2/B3 stack on L1 scenarios

Combinations:
  A. L1 alone (rolling baseline)
  B. L1 + L2 (Hybrid replaces L1 mean, L1 sigma)
  C. L1 + L3 (L1 mean + reweight distribution)
  D. L1 + L2 + L3 (Hybrid mean + reweight distribution)

Metrics (1h-ahead, eval 7-12월):
  - MAE / Portfolio NMAE
  - cov80 / cov95 (split conformal calibrated)
  - NLL (Normal vs t)
  - CRPS
  - Fuel backup cost (proxy)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

# Student-t parameters from B1
T_DF = 3.73
T_LOC = 0.09
T_SCALE = 0.688

LNG_COST_PER_KWH = 169   # 원/kWh (estimate_lng_cost.py)


def normal_nll(y, mu, sig):
    sig = np.maximum(sig, 1e-3)
    return 0.5 * np.log(2 * np.pi * sig ** 2) + ((y - mu) ** 2) / (2 * sig ** 2)


def t_nll(y, mu, sig, nu, loc_t, scale_t):
    sig = np.maximum(sig, 1e-3)
    z = (y - mu) / sig
    log_pdf_z = stats.t.logpdf(z, df=nu, loc=loc_t, scale=scale_t)
    return np.log(sig) - log_pdf_z


def crps_normal(y, mu, sig):
    sig = np.maximum(sig, 1e-3)
    z = (y - mu) / sig
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return sig * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))


def crps_t_mc(y, mu, sig, nu, loc_t, scale_t, n=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    sig = np.maximum(sig, 1e-3)
    z = stats.t.rvs(df=nu, loc=loc_t, scale=scale_t, size=(n, len(y)), random_state=rng)
    X = mu[None, :] + sig[None, :] * z
    z2 = stats.t.rvs(df=nu, loc=loc_t, scale=scale_t, size=(n, len(y)), random_state=rng)
    X2 = mu[None, :] + sig[None, :] * z2
    return np.abs(X - y[None, :]).mean(axis=0) - 0.5 * np.abs(X - X2).mean(axis=0)


def fuel_cost_proxy(y, mu, q10):
    """Hedge LNG cost proxy:
       worst-case backup = max(0, μ - q10)  (예측 mu 기준 q10까지 떨어질 가능성에 대비)
       실제 부족분 = max(0, μ - y)
       Cost = (over-hedge + under-hedge) * LNG_COST
    """
    over_hedge = np.maximum(0, mu - q10)   # 예약된 LNG (q10까지 cover)
    actual_shortage = np.maximum(0, mu - y)
    # hedge 효율: over_hedge가 actual_shortage 이상이면 cost = over_hedge (낭비)
    #             아니면 cost = over_hedge + (actual_shortage - over_hedge) (사후 비상 ramp)
    # 단순화: total LNG cost = over_hedge + actual_shortage_above_hedge
    extra_emergency = np.maximum(0, actual_shortage - over_hedge)
    total_lng_kwh = over_hedge + extra_emergency * 1.5  # 비상 ramp 50% premium
    return total_lng_kwh * LNG_COST_PER_KWH


def main():
    print("=" * 80)
    print("3-Layer 복합 vs Rolling Baseline — 종합 비교")
    print("=" * 80)

    # ===== Load data =====
    print("\n[Load] L1 (ResMLP+AdaLN) site predictions...")
    l1_site = pd.read_parquet(ROOT / "pv/experiments/resmlp_adaln/test_predictions.parquet")
    l1_site = l1_site.rename(columns={"cf": "actual_cf", "pred_cf": "mu_cf", "pred_std_cf": "sig_cf"})

    print("[Load] L2 (Hybrid) site predictions...")
    l2_site = pd.read_parquet(ROOT / "pv/experiments/hybrid_resmlp_cnn/test_predictions.parquet")
    # Hybrid columns: site, datetime_kst, site_capacity_kw, cf, pred_cf, pred_std_cf, ...
    l2_site = l2_site.rename(columns={"cf": "actual_cf", "pred_cf": "mu_cf", "pred_std_cf": "sig_cf"})

    print("[Load] L3 (reweight on portfolio) predictions...")
    l3 = pd.read_parquet(ROOT / "pv/experiments/realtime_simulation/predictions_A_vs_B.parquet")
    # L3 has horizon_h, mean_A/B/Bp/Bpf, std_A/B/Bp/Bpf
    # 1h-ahead만 사용
    l3_1h = l3[l3["horizon_h"] == 1].copy()
    print(f"  L3 (1h-ahead): {len(l3_1h):,} hours")

    # ===== Aggregate L1, L2 to portfolio level =====
    def agg_to_port(df, label):
        df = df.copy()
        df["mu_kwh"] = df["mu_cf"] * df["site_capacity_kw"]
        df["actual_kwh"] = df["actual_cf"] * df["site_capacity_kw"]
        df["sig_kwh_sq"] = (df["sig_cf"] * df["site_capacity_kw"]) ** 2
        port = df.groupby("datetime_kst", as_index=False).agg(
            mu=("mu_kwh", "sum"),
            actual=("actual_kwh", "sum"),
            var_sum=("sig_kwh_sq", "sum"),
            cap=("site_capacity_kw", "sum"),
        )
        port["sigma"] = np.sqrt(port["var_sum"])
        port = port.rename(columns={"mu": f"{label}_mu", "sigma": f"{label}_sig"})
        return port[["datetime_kst", "actual", "cap", f"{label}_mu", f"{label}_sig"]]

    l1_port = agg_to_port(l1_site, "L1")
    l2_port = agg_to_port(l2_site, "L2")
    print(f"  L1 portfolio: {len(l1_port):,}, L2 portfolio: {len(l2_port):,}")

    # ===== Merge by datetime =====
    print("\n[Merge] L1, L2, L3 by datetime...")
    l3_1h = l3_1h.copy()
    # L3 has 'date' and 'prediction_hour'. Build datetime_kst.
    l3_1h["datetime_kst"] = pd.to_datetime(l3_1h["date"]) + pd.to_timedelta(l3_1h["prediction_hour"], unit="h")
    l3_keys = l3_1h[["datetime_kst", "actual_at_h", "mean_A", "std_A", "mean_B", "std_B",
                     "mean_Bp", "std_Bp", "mean_Bpf", "std_Bpf"]]
    merged = l1_port.merge(l2_port[["datetime_kst", "L2_mu", "L2_sig"]], on="datetime_kst", how="inner")
    merged = merged.merge(l3_keys, on="datetime_kst", how="inner")
    print(f"  joined rows: {len(merged):,}")

    # Train/eval split
    merged["month"] = merged["datetime_kst"].dt.month
    fit = merged[merged["month"] <= 6].copy()
    eval_ = merged[merged["month"] >= 7].copy()
    print(f"  fit (1-6월): {len(fit):,}, eval (7-12월): {len(eval_):,}")

    # ===== Calibration (split conformal on fit) =====
    print("\n[Calibrate] Split conformal q80, q95 (fit set)...")
    cal = {}
    for label, mu_col, sig_col in [
        ("L1", "L1_mu", "L1_sig"),
        ("L2", "L2_mu", "L2_sig"),
        ("L3_B", "mean_B", "std_B"),
        ("L3_Bpf", "mean_Bpf", "std_Bpf"),
    ]:
        sig = np.maximum(fit[sig_col].values, 100)
        z_abs = np.abs(fit["actual"].values - fit[mu_col].values) / sig
        cal[label] = {"q80": np.quantile(z_abs, 0.8), "q95": np.quantile(z_abs, 0.95)}
        print(f"  {label}: q80={cal[label]['q80']:.3f}, q95={cal[label]['q95']:.3f}")

    # ===== Define 4 combinations =====
    print("\n[Combine] 4 setups:")
    print("  A. L1 only             — mu=L1, sigma=L1")
    print("  B. L1+L2 (hybrid mean) — mu=L2, sigma=L2")
    print("  C. L1+L3 (reweight)    — mu=mean_B, sigma=std_B")
    print("  D. L1+L2+L3            — mu=L2, sigma=std_B (Hybrid mean + reweight σ)")

    eval_["A_mu"] = eval_["L1_mu"]
    eval_["A_sig"] = eval_["L1_sig"]
    eval_["A_q80"] = cal["L1"]["q80"]
    eval_["A_q95"] = cal["L1"]["q95"]

    eval_["B_mu"] = eval_["L2_mu"]
    eval_["B_sig"] = eval_["L2_sig"]
    eval_["B_q80"] = cal["L2"]["q80"]
    eval_["B_q95"] = cal["L2"]["q95"]

    eval_["C_mu"] = eval_["mean_B"]
    eval_["C_sig"] = eval_["std_B"]
    eval_["C_q80"] = cal["L3_B"]["q80"]
    eval_["C_q95"] = cal["L3_B"]["q95"]

    eval_["D_mu"] = eval_["L2_mu"]
    eval_["D_sig"] = eval_["std_B"]   # Hybrid mean + reweight σ
    eval_["D_q80"] = cal["L3_B"]["q80"]
    eval_["D_q95"] = cal["L3_B"]["q95"]

    # ===== Compute metrics =====
    print("\n[Metrics] Eval 1h-ahead 종합 비교\n")
    rows = []
    for setup in ["A", "B", "C", "D"]:
        mu = eval_[f"{setup}_mu"].values
        sig = np.maximum(eval_[f"{setup}_sig"].values, 100)
        q80 = eval_[f"{setup}_q80"].values
        q95 = eval_[f"{setup}_q95"].values
        y = eval_["actual"].values
        cap = eval_["cap"].values

        mae = np.abs(y - mu).mean() / 1000
        nmae_port = np.abs(y - mu).sum() / cap.sum() * 100
        cov80 = ((y >= mu - q80 * sig) & (y <= mu + q80 * sig)).mean() * 100
        cov95 = ((y >= mu - q95 * sig) & (y <= mu + q95 * sig)).mean() * 100
        nll_n = normal_nll(y, mu, sig).mean()
        nll_t = t_nll(y, mu, sig, T_DF, T_LOC, T_SCALE).mean()
        crps_n = crps_normal(y, mu, sig).mean()
        crps_t_val = crps_t_mc(y, mu, sig, T_DF, T_LOC, T_SCALE, n=150).mean()

        # Fuel cost: hedge to q10 (Normal: mu - 1.282*sig)
        q10 = mu - 1.282 * sig
        cost_total = fuel_cost_proxy(y, mu, q10).sum() / 1e8   # 억원
        cost_per_hour = fuel_cost_proxy(y, mu, q10).mean() / 1e4   # 만원/h

        rows.append({
            "setup": setup, "MAE_MWh": mae, "NMAE_port_%": nmae_port,
            "cov80_%": cov80, "cov95_%": cov95,
            "NLL_normal": nll_n, "NLL_t": nll_t,
            "CRPS_normal": crps_n, "CRPS_t": crps_t_val,
            "fuel_cost_억": cost_total, "cost_per_hour_만": cost_per_hour,
        })

    res = pd.DataFrame(rows).set_index("setup")
    print(res.round(3).to_string())

    # ===== Incremental gain =====
    print(f"\n[Incremental gain]")
    a = res.loc["A"]
    print(f"  L1 (baseline) → 모든 setup의 reference")
    for setup in ["B", "C", "D"]:
        s = res.loc[setup]
        print(f"  {setup} vs A:")
        print(f"     MAE:    {a['MAE_MWh']:.3f} → {s['MAE_MWh']:.3f}  ({(a['MAE_MWh']-s['MAE_MWh'])/a['MAE_MWh']*100:+.1f}%)")
        print(f"     NMAE:   {a['NMAE_port_%']:.2f}% → {s['NMAE_port_%']:.2f}%")
        print(f"     cov80:  {a['cov80_%']:.1f}% → {s['cov80_%']:.1f}%")
        print(f"     NLL_t:  {a['NLL_t']:.4f} → {s['NLL_t']:.4f}  ({(a['NLL_t']-s['NLL_t']):+.4f})")
        print(f"     CRPS_t: {a['CRPS_t']:.4f} → {s['CRPS_t']:.4f}  ({(a['CRPS_t']-s['CRPS_t']):+.4f})")
        print(f"     Fuel:   {a['fuel_cost_억']:.2f}억 → {s['fuel_cost_억']:.2f}억  ({a['fuel_cost_억']-s['fuel_cost_억']:+.2f}억)")

    # Save
    out_dir = ROOT / "pv/experiments/layered_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "comparison.csv")
    print(f"\n저장: {out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
