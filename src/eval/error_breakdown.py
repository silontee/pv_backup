"""Error breakdown analysis — current winner ResMLP+AdaLN.

Upper-bound setting (perfect-foresight). 2025 test set 전체에 대해:
  1. Row-level prediction table
  2. Site / hour / month·season / cloud bin / dsr bin / cf bin / zenith bin
  3. Hand-crafted regime (clear/cloudy/transition) breakdown
  4. Residual vs (dc10Tca / dsr / zenith / actual_cf) bin mean
  5. Portfolio: hourly residual, largest errors, simultaneous miss
  6. Cov80 / Cov95 site별 + regime별 (miscalibration source)
  7. Actionable implications

출력:
  - data/processed/error_breakdown_2025.parquet (row-level)
  - pv/experiments/error_breakdown/ (모든 분석 CSV + plot)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "pv/experiments/error_breakdown"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_winner_predictions():
    """ResMLP+AdaLN test predictions."""
    p = ROOT / "pv/experiments/resmlp_adaln/test_predictions.parquet"
    pred = pd.read_parquet(p)
    pred = pred.rename(columns={"cf": "actual_cf", "pred_cf": "pred_mu",
                                 "pred_std_cf": "pred_sigma"})
    return pred[["datetime_kst", "site", "site_capacity_kw",
                 "actual_cf", "pred_mu", "pred_sigma"]]


def join_features(pred):
    """training_set의 weather features 조인."""
    ts = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    ts = ts[["datetime_kst", "site", "dsr_mean", "zenith_center",
             "ta", "hm", "ws", "dc10Tca"]].copy()
    return pred.merge(ts, on=["datetime_kst", "site"], how="left")


def add_metrics(df):
    df["residual"] = df["actual_cf"] - df["pred_mu"]   # signed
    df["abs_error"] = df["residual"].abs()
    df["squared_error"] = df["residual"] ** 2
    df["abs_err_kwh"] = df["abs_error"] * df["site_capacity_kw"]
    df["err_kwh"] = df["residual"] * df["site_capacity_kw"]   # signed kWh
    df["hour"] = df["datetime_kst"].dt.hour
    df["month"] = df["datetime_kst"].dt.month

    def season(m):
        if m in (12, 1, 2): return "winter"
        if m in (3, 4, 5): return "spring"
        if m in (6, 7, 8): return "summer"
        return "autumn"

    df["season"] = df["month"].apply(season)

    # Coverage indicators
    df["in_80"] = ((df["actual_cf"] >= df["pred_mu"] - 1.282 * df["pred_sigma"]) &
                   (df["actual_cf"] <= df["pred_mu"] + 1.282 * df["pred_sigma"])).astype(int)
    df["in_95"] = ((df["actual_cf"] >= df["pred_mu"] - 1.96 * df["pred_sigma"]) &
                   (df["actual_cf"] <= df["pred_mu"] + 1.96 * df["pred_sigma"])).astype(int)
    return df


def regime_label(df):
    """Hand-crafted regime proxy.
    clear:      dsr_mean > 400  AND  dc10Tca < 3
    cloudy:     dsr_mean < 200  OR   dc10Tca >= 7
    transition: 그 외
    """
    cond_clear = (df["dsr_mean"] > 400) & (df["dc10Tca"] < 3)
    cond_cloudy = (df["dsr_mean"] < 200) | (df["dc10Tca"] >= 7)
    df["regime"] = "transition"
    df.loc[cond_clear, "regime"] = "clear"
    df.loc[cond_cloudy, "regime"] = "cloudy"
    return df


def grouped_metrics(df, group_col):
    """Generic per-group: n, MAE, bias, NMAE %, cov80, cov95."""
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        if len(sub) == 0: continue
        rows.append({
            group_col: str(g),
            "n": len(sub),
            "MAE_cf": sub["abs_error"].mean(),
            "bias_cf": sub["residual"].mean(),
            "NMAE_%": sub["abs_err_kwh"].sum() / sub["site_capacity_kw"].sum() * 100,
            "MAE_MWh": sub["abs_err_kwh"].sum() / len(sub) / 1000,
            "cov80": sub["in_80"].mean() * 100,
            "cov95": sub["in_95"].mean() * 100,
        })
    return pd.DataFrame(rows)


def bin_breakdown(df, var, bins, var_label=None):
    """Bin var by edges, compute breakdown."""
    df = df.copy()
    df["bin"] = pd.cut(df[var], bins=bins, include_lowest=True)
    return grouped_metrics(df, "bin")


def residual_vs_bin(df, var, bins):
    """Residual mean per bin."""
    df = df.copy()
    df["bin"] = pd.cut(df[var], bins=bins, include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        n=("residual", "size"),
        residual_mean=("residual", "mean"),
        residual_std=("residual", "std"),
        abs_error_mean=("abs_error", "mean"),
    ).reset_index()
    return out


# ===== Portfolio analysis =====

def portfolio_analysis(df):
    pf = df.groupby("datetime_kst").agg(
        actual_kwh=("actual_cf", lambda s: (s * df.loc[s.index, "site_capacity_kw"]).sum()),
        pred_kwh=("pred_mu", lambda s: (s * df.loc[s.index, "site_capacity_kw"]).sum()),
        sigma_kwh2=("pred_sigma", lambda s: ((s * df.loc[s.index, "site_capacity_kw"]) ** 2).sum()),
        cap_kw=("site_capacity_kw", "sum"),
        n_sites=("site", "size"),
    ).reset_index()
    pf["sigma_kwh"] = np.sqrt(pf["sigma_kwh2"])
    pf["residual_kwh"] = pf["actual_kwh"] - pf["pred_kwh"]
    pf["abs_err_kwh"] = pf["residual_kwh"].abs()
    pf["nmae_pct"] = pf["abs_err_kwh"] / pf["cap_kw"] * 100
    return pf


def simultaneous_miss(df, threshold_z=2.0):
    """동일 시간에 여러 사이트가 동시에 큰 error 발생한 경우."""
    df = df.copy()
    df["site_z"] = (df["residual"] / df["pred_sigma"].clip(lower=1e-3)).abs()
    by_time = df.groupby("datetime_kst").agg(
        n_sites_high_err=("site_z", lambda s: (s > threshold_z).sum()),
        n_sites_total=("site_z", "size"),
        max_z=("site_z", "max"),
        mean_residual=("residual", "mean"),
    ).reset_index()
    return by_time.sort_values("n_sites_high_err", ascending=False)


# ===== Plots =====

def save_residual_plots(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # 1. residual vs cloud
    bins = np.linspace(0, 10, 11)
    cb = residual_vs_bin(df, "dc10Tca", bins)
    centers = np.array([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)])
    axes[0, 0].errorbar(centers[:len(cb)], cb["residual_mean"],
                          yerr=cb["residual_std"], marker="o", capsize=3)
    axes[0, 0].axhline(0, color="black", lw=0.6, ls="--")
    axes[0, 0].set_xlabel("dc10Tca (cloud 0-10)")
    axes[0, 0].set_ylabel("residual mean ± std")
    axes[0, 0].set_title("Residual vs Cloud")
    axes[0, 0].grid(alpha=0.3)

    # 2. residual vs dsr_mean
    bins = np.linspace(0, 1100, 12)
    cb = residual_vs_bin(df, "dsr_mean", bins)
    centers = np.array([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)])
    axes[0, 1].errorbar(centers[:len(cb)], cb["residual_mean"],
                          yerr=cb["residual_std"], marker="o", capsize=3, color="orange")
    axes[0, 1].axhline(0, color="black", lw=0.6, ls="--")
    axes[0, 1].set_xlabel("dsr_mean (W/m²)")
    axes[0, 1].set_ylabel("residual mean ± std")
    axes[0, 1].set_title("Residual vs DSR")
    axes[0, 1].grid(alpha=0.3)

    # 3. residual vs zenith
    bins = np.linspace(20, 90, 15)
    cb = residual_vs_bin(df, "zenith_center", bins)
    centers = np.array([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)])
    axes[1, 0].errorbar(centers[:len(cb)], cb["residual_mean"],
                          yerr=cb["residual_std"], marker="o", capsize=3, color="green")
    axes[1, 0].axhline(0, color="black", lw=0.6, ls="--")
    axes[1, 0].set_xlabel("zenith (deg)")
    axes[1, 0].set_ylabel("residual mean ± std")
    axes[1, 0].set_title("Residual vs Zenith")
    axes[1, 0].grid(alpha=0.3)

    # 4. residual vs actual_cf
    bins = np.linspace(0, 1.0, 11)
    cb = residual_vs_bin(df, "actual_cf", bins)
    centers = np.array([(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)])
    axes[1, 1].errorbar(centers[:len(cb)], cb["residual_mean"],
                          yerr=cb["residual_std"], marker="o", capsize=3, color="red")
    axes[1, 1].axhline(0, color="black", lw=0.6, ls="--")
    axes[1, 1].set_xlabel("actual_cf")
    axes[1, 1].set_ylabel("residual mean ± std")
    axes[1, 1].set_title("Residual vs Actual cf")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "residual_vs_features.png", dpi=110, bbox_inches="tight")
    plt.close()


# ===== Main =====

def main():
    print("=" * 70)
    print("Error Breakdown — ResMLP+AdaLN (winner, perfect-foresight UB)")
    print("=" * 70)

    pred = load_winner_predictions()
    df = join_features(pred)
    df = add_metrics(df)
    df = regime_label(df)
    print(f"\nRow-level table: {len(df):,} rows, "
          f"date {df['datetime_kst'].min()} ~ {df['datetime_kst'].max()}")

    # Save row-level
    df.to_parquet(OUT_DIR / "row_level_2025.parquet", index=False)
    print(f"  → {OUT_DIR / 'row_level_2025.parquet'}")

    # ======== Section 1: 전체 요약 ========
    overall_nmae = df["abs_err_kwh"].sum() / df["site_capacity_kw"].sum() * 100
    overall_bias = df["err_kwh"].sum() / df["site_capacity_kw"].sum() * 100
    overall_cov80 = df["in_80"].mean() * 100
    overall_cov95 = df["in_95"].mean() * 100
    print("\n" + "=" * 70)
    print("(1) 전체 요약")
    print("=" * 70)
    print(f"  Site NMAE       : {overall_nmae:.3f}%")
    print(f"  Bias (signed)   : {overall_bias:+.3f}% of capacity")
    print(f"  Cov80           : {overall_cov80:.1f}%")
    print(f"  Cov95           : {overall_cov95:.1f}%")

    # ======== Section 2: Breakdown ========
    print("\n" + "=" * 70)
    print("(2) Breakdown")
    print("=" * 70)

    # Site
    print("\n--- Site별 ---")
    by_site = grouped_metrics(df, "site").sort_values("NMAE_%", ascending=False)
    print(by_site.round(3).to_string(index=False))
    by_site.to_csv(OUT_DIR / "by_site.csv", index=False)

    # Hour
    print("\n--- Hour별 (daytime only) ---")
    by_hour = grouped_metrics(df, "hour")
    print(by_hour.round(3).to_string(index=False))
    by_hour.to_csv(OUT_DIR / "by_hour.csv", index=False)

    # Month
    print("\n--- Month별 ---")
    by_month = grouped_metrics(df, "month")
    print(by_month.round(3).to_string(index=False))
    by_month.to_csv(OUT_DIR / "by_month.csv", index=False)

    # Season
    print("\n--- Season별 ---")
    by_season = grouped_metrics(df, "season")
    print(by_season.round(3).to_string(index=False))
    by_season.to_csv(OUT_DIR / "by_season.csv", index=False)

    # Cloud bin
    print("\n--- Cloud bin (dc10Tca) ---")
    by_cloud = bin_breakdown(df, "dc10Tca", [-0.1, 1, 3, 5, 7, 9, 10.1])
    print(by_cloud.round(3).to_string(index=False))
    by_cloud.to_csv(OUT_DIR / "by_cloud.csv", index=False)

    # DSR bin
    print("\n--- DSR bin (W/m²) ---")
    by_dsr = bin_breakdown(df, "dsr_mean", [0, 100, 250, 400, 600, 800, 1100])
    print(by_dsr.round(3).to_string(index=False))
    by_dsr.to_csv(OUT_DIR / "by_dsr.csv", index=False)

    # CF level bin (실측 cf 기준)
    print("\n--- Actual cf bin ---")
    by_cf = bin_breakdown(df, "actual_cf", [0, 0.1, 0.25, 0.5, 0.75, 1.0])
    print(by_cf.round(3).to_string(index=False))
    by_cf.to_csv(OUT_DIR / "by_cf_level.csv", index=False)

    # Zenith bin
    print("\n--- Zenith bin (deg) ---")
    by_zen = bin_breakdown(df, "zenith_center", [20, 30, 40, 50, 60, 70, 80, 90])
    print(by_zen.round(3).to_string(index=False))
    by_zen.to_csv(OUT_DIR / "by_zenith.csv", index=False)

    # Regime
    print("\n--- Regime (clear / cloudy / transition) ---")
    by_regime = grouped_metrics(df, "regime")
    print(by_regime.round(3).to_string(index=False))
    by_regime.to_csv(OUT_DIR / "by_regime.csv", index=False)

    # Site × Regime
    print("\n--- Site × Regime cov80 ---")
    site_regime_cov = df.groupby(["site", "regime"]).agg(
        n=("actual_cf", "size"),
        cov80=("in_80", "mean"),
        cov95=("in_95", "mean"),
        nmae=("abs_err_kwh", lambda x: x.sum() / df.loc[x.index, "site_capacity_kw"].sum() * 100),
    ).reset_index()
    site_regime_cov["cov80"] *= 100
    site_regime_cov["cov95"] *= 100
    print(site_regime_cov.round(2).to_string(index=False))
    site_regime_cov.to_csv(OUT_DIR / "by_site_regime.csv", index=False)

    # ======== Section 3: Residual vs feature (bin mean) ========
    print("\n--- Residual vs features (bin 평균) ---")
    print("\n  vs cloud:")
    cb = residual_vs_bin(df, "dc10Tca", np.linspace(0, 10, 11))
    print(cb.round(4).to_string(index=False))
    cb.to_csv(OUT_DIR / "residual_vs_cloud.csv", index=False)

    print("\n  vs dsr_mean:")
    cb = residual_vs_bin(df, "dsr_mean", np.linspace(0, 1100, 12))
    print(cb.round(4).to_string(index=False))
    cb.to_csv(OUT_DIR / "residual_vs_dsr.csv", index=False)

    print("\n  vs zenith:")
    cb = residual_vs_bin(df, "zenith_center", np.linspace(20, 90, 15))
    print(cb.round(4).to_string(index=False))
    cb.to_csv(OUT_DIR / "residual_vs_zenith.csv", index=False)

    print("\n  vs actual_cf:")
    cb = residual_vs_bin(df, "actual_cf", np.linspace(0, 1, 11))
    print(cb.round(4).to_string(index=False))
    cb.to_csv(OUT_DIR / "residual_vs_actualcf.csv", index=False)

    save_residual_plots(df)
    print(f"\n  → plot: {OUT_DIR / 'residual_vs_features.png'}")

    # ======== Portfolio ========
    print("\n--- Portfolio analysis ---")
    pf = portfolio_analysis(df)
    print(f"  Portfolio rows: {len(pf):,}")
    pf_nmae = pf["abs_err_kwh"].sum() / pf["cap_kw"].sum() * 100
    pf_bias = pf["residual_kwh"].sum() / pf["cap_kw"].sum() * 100
    print(f"  Portfolio NMAE: {pf_nmae:.3f}%  / Bias: {pf_bias:+.3f}%")

    # 큰 error case
    print("\n  Top 10 largest portfolio absolute errors:")
    top = pf.nlargest(10, "abs_err_kwh")[["datetime_kst", "actual_kwh", "pred_kwh",
                                            "residual_kwh", "abs_err_kwh"]]
    print(top.assign(
        actual_MWh=lambda d: d["actual_kwh"] / 1000,
        pred_MWh=lambda d: d["pred_kwh"] / 1000,
        err_MWh=lambda d: d["residual_kwh"] / 1000,
    )[["datetime_kst", "actual_MWh", "pred_MWh", "err_MWh"]].round(2).to_string(index=False))
    pf.to_csv(OUT_DIR / "portfolio_hourly.csv", index=False)

    # 동시 miss
    print("\n  Top 10 hours with most simultaneous large-error sites (z>2):")
    sm = simultaneous_miss(df, threshold_z=2.0)
    top_sm = sm.head(10)
    print(top_sm.round(2).to_string(index=False))
    sm.to_csv(OUT_DIR / "simultaneous_miss.csv", index=False)

    # ======== Actionable implications ========
    print("\n" + "=" * 70)
    print("(3) Actionable Implications")
    print("=" * 70)

    # Worst site
    worst_site = by_site.iloc[0]
    print(f"\n  • Worst site: {worst_site['site']} (NMAE {worst_site['NMAE_%']:.2f}%, "
          f"bias {worst_site['bias_cf']:+.4f})")

    # Worst regime
    worst_regime = by_regime.sort_values("NMAE_%", ascending=False).iloc[0]
    print(f"  • Worst regime: {worst_regime['regime']} (NMAE {worst_regime['NMAE_%']:.2f}%, "
          f"n={worst_regime['n']:,})")

    # Bias direction
    if abs(overall_bias) > 0.5:
        direction = "over-predict" if overall_bias < 0 else "under-predict"
        print(f"  • Overall bias {overall_bias:+.3f}% → 모델이 *{direction}* 경향")
    else:
        print(f"  • Overall bias {overall_bias:+.3f}% → unbiased")

    # Coverage
    if overall_cov80 > 82:
        print(f"  • Cov80 {overall_cov80:.1f}% (target 80%) → σ가 약간 *over* (덜 confident)")
    elif overall_cov80 < 78:
        print(f"  • Cov80 {overall_cov80:.1f}% → σ가 *under* (over-confident)")
    else:
        print(f"  • Cov80 {overall_cov80:.1f}% → well-calibrated")

    # Worst regime miscalibration
    by_reg = grouped_metrics(df, "regime")
    print(f"\n  Coverage by regime:")
    for _, r in by_reg.iterrows():
        flag = "⚠️" if abs(r["cov80"] - 80) > 5 else "✅"
        print(f"    {r['regime']:<12} cov80 {r['cov80']:.1f}% {flag}")

    # Implications
    print(f"\n  Recommendations:")
    print(f"  1. Site-level: '{worst_site['site']}' specialist correction "
          f"(현 NMAE {worst_site['NMAE_%']:.2f}% > 평균)")
    print(f"  2. Regime: '{worst_regime['regime']}' regime에 *cloud-conditional bias correction*")
    print(f"  3. Coverage: regime별 σ scaling (post-hoc conformal calibration)")
    print(f"  4. 시간대별 hour-conditional residual shift 가능 (bias by hour 큰 경우)")

    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
