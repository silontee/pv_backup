"""NGBoost 출력 (μ, σ) 사후 보정 — cloud level 조건부.

문제 진단:
  Cloud level별 NGBoost calibration 비대칭:
    cloud 0-1: rel_err +1.7%, well-calibrated
    cloud 7-8: rel_err -28.8%, σ도 작음 (under-spread)
    cloud 9-10: rel_err -153% (50%+ over-prediction)

방법:
  z = (actual - μ) / σ → cloud level별 mean(z), std(z) 측정
  bias(c) = mean(z | cloud=c)        → 0이면 정확, ≠0이면 systematic bias
  spread(c) = std(z | cloud=c)       → 1이면 well-calibrated, >1이면 under-confident
  보정:
    μ_adj = μ + bias(c) * σ
    σ_adj = σ * spread(c)

Train/Eval split:
  fit: 2025-01-01 ~ 2025-06-30
  eval: 2025-07-01 ~ 2025-12-31
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def load_with_cloud():
    ts = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    port = pd.read_parquet(ROOT / "data/processed/portfolio_predictions.parquet")

    ts["date"] = ts["datetime_kst"].dt.date
    ts["hour"] = ts["datetime_kst"].dt.hour
    cloud_h = ts.groupby(["date", "hour"], as_index=False).agg(cloud=("dc10Tca", "mean"))

    port["date"] = port["datetime_kst"].dt.date
    port["hour"] = port["datetime_kst"].dt.hour
    df = port.merge(cloud_h, on=["date", "hour"], how="inner")
    df["z"] = (df["actual_kwh"] - df["mu_kwh"]) / df["sigma_kwh"]
    df["month"] = df["datetime_kst"].dt.month
    return df


def fit_calibration(fit_df: pd.DataFrame, n_bins: int = 11):
    """Cloud bin 별 (bias, spread) 측정."""
    bins = np.linspace(0, 10, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2

    rows = []
    for i in range(n_bins):
        sub = fit_df[(fit_df["cloud"] >= bins[i]) & (fit_df["cloud"] < bins[i + 1])]
        if len(sub) < 20:
            continue
        rows.append({
            "bin_center": centers[i],
            "n": len(sub),
            "bias": sub["z"].mean(),
            "spread": sub["z"].std(),
        })
    return pd.DataFrame(rows)


def make_lookup(calib: pd.DataFrame):
    """Cloud level → (bias, spread) interpolator (linear, clipped at edges)."""
    x = calib["bin_center"].values
    bias = calib["bias"].values
    spread = calib["spread"].values

    def f_bias(c):
        return np.interp(np.clip(c, x.min(), x.max()), x, bias)

    def f_spread(c):
        return np.interp(np.clip(c, x.min(), x.max()), x, spread)

    return f_bias, f_spread


def apply_correction(df: pd.DataFrame, f_bias, f_spread):
    df = df.copy()
    b = f_bias(df["cloud"].values)
    s = f_spread(df["cloud"].values)
    df["mu_adj"] = df["mu_kwh"] + b * df["sigma_kwh"]
    df["mu_adj"] = df["mu_adj"].clip(lower=0)
    df["sigma_adj"] = df["sigma_kwh"] * s
    df["z_adj"] = (df["actual_kwh"] - df["mu_adj"]) / df["sigma_adj"]
    return df


def metrics(df: pd.DataFrame, mu_col: str, sig_col: str):
    err = df["actual_kwh"] - df[mu_col]
    mae = err.abs().mean() / 1000
    nmae = err.abs().sum() / df["total_capacity_kw"].sum() * 100  # %
    z = err / df[sig_col]
    cov80 = ((z >= -1.282) & (z <= 1.282)).mean() * 100
    cov95 = ((z >= -1.96) & (z <= 1.96)).mean() * 100
    bias_mean = (df[mu_col] - df["actual_kwh"]).mean() / 1000  # MWh, +면 over-predict
    return {
        "MAE_MWh": mae,
        "NMAE_%": nmae,
        "bias_MWh": bias_mean,
        "cov80_%": cov80,
        "cov95_%": cov95,
        "z_mean": z.mean(),
        "z_std": z.std(),
    }


def main():
    print("[1] 데이터 로딩...")
    df = load_with_cloud()
    print(f"  rows: {len(df):,}, dates: {df['date'].nunique()}")

    fit = df[df["month"] <= 6].copy()
    eval_ = df[df["month"] >= 7].copy()
    print(f"  fit (1-6월): {len(fit):,}, eval (7-12월): {len(eval_):,}")

    print("\n[2] Cloud bin 별 calibration 측정 (fit set)...")
    calib = fit_calibration(fit, n_bins=11)
    print(calib.round(3).to_string(index=False))

    f_bias, f_spread = make_lookup(calib)

    print("\n[3] Eval set 보정 적용...")
    eval_corr = apply_correction(eval_, f_bias, f_spread)

    print("\n[4] 비교 metrics:")
    before = metrics(eval_corr, "mu_kwh", "sigma_kwh")
    after = metrics(eval_corr, "mu_adj", "sigma_adj")
    cmp = pd.DataFrame({"before": before, "after": after}).round(3)
    cmp["Δ"] = cmp["after"] - cmp["before"]
    print(cmp.to_string())

    print("\n[5] Cloud bin 별 *eval set* 효과:")
    bins = np.linspace(0, 10, 6)
    eval_corr["cloud_bin"] = pd.cut(eval_corr["cloud"], bins=bins, include_lowest=True)
    for label, sub in eval_corr.groupby("cloud_bin", observed=True):
        b = metrics(sub, "mu_kwh", "sigma_kwh")
        a = metrics(sub, "mu_adj", "sigma_adj")
        print(f"  {label}: n={len(sub):>4} | "
              f"MAE {b['MAE_MWh']:.2f} → {a['MAE_MWh']:.2f} ({a['MAE_MWh']-b['MAE_MWh']:+.2f}) "
              f"| cov95 {b['cov95_%']:.0f}% → {a['cov95_%']:.0f}%")

    # 시각화
    print("\n[6] PNG 저장...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(calib["bin_center"], calib["bias"], width=0.7, color="steelblue", alpha=0.7, label="bias = E[z|cloud]")
    axes[0].axhline(y=0, color="black", lw=0.8)
    axes[0].set_xlabel("Cloud level (dc10Tca)")
    axes[0].set_ylabel("Mean z-score")
    axes[0].set_title("NGBoost calibration bias by cloud (fit set)")
    axes[0].legend()

    axes[1].bar(calib["bin_center"], calib["spread"], width=0.7, color="orange", alpha=0.7, label="spread = std[z|cloud]")
    axes[1].axhline(y=1, color="black", lw=0.8, ls="--", label="ideal=1")
    axes[1].set_xlabel("Cloud level")
    axes[1].set_ylabel("Std z-score")
    axes[1].set_title("NGBoost spread (under/over-confident) by cloud")
    axes[1].legend()
    plt.tight_layout()

    out_dir = ROOT / "pv/experiments/cloud_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "calibration_curves.png", dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  {out_dir / 'calibration_curves.png'}")

    # 보정 결과 저장 (cloud_bin Interval column drop)
    eval_corr.drop(columns=["cloud_bin"]).to_parquet(out_dir / "eval_corrected.parquet", index=False)
    calib.to_csv(out_dir / "calibration_table.csv", index=False)
    print(f"  {out_dir / 'eval_corrected.parquet'}")
    print(f"  {out_dir / 'calibration_table.csv'}")


if __name__ == "__main__":
    main()
