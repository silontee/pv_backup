"""B3: Split Conformal Calibration + 종합 비교.

방법:
  1. Fit set (2025 1-6월): residual_norm = |y - μ| / σ 측정 (per mode A/B/Bp/Bpf)
  2. q_80 = 80% empirical quantile of residual_norm
  3. q_95 = 95% empirical quantile
  4. Eval set: cov80 = (|y - μ| / σ <= q_80).mean()  → exactly 80% by construction

비교:
  Raw (Normal/t implied):     scenarios assumes Normal CDF symmetry
  Conformal:                  empirical quantile, distribution-free

  추가: tail event support, outlier day ESS, 1h MAE
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def main():
    print("=" * 70)
    print("B3: Split Conformal Calibration + 종합 비교")
    print("=" * 70)

    df = pd.read_parquet(ROOT / "pv/experiments/realtime_simulation/predictions_A_vs_B.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    fit = df[df["month"] <= 6].copy()
    eval_ = df[df["month"] >= 7].copy()
    print(f"\n[Data] fit {len(fit):,}, eval {len(eval_):,}")

    # ====== Step 1: Conformal quantiles per mode ======
    print("\n[1] Conformal quantiles (fit set, 1h-ahead만):")
    fit_1h = fit[fit["horizon_h"] == 1].copy()
    print(f"  fit 1h-ahead n: {len(fit_1h):,}")

    print(f"  {'Mode':<5} {'q_80 (Normal=1.282)':<22} {'q_95 (Normal=1.960)':<22}")
    confs = {}
    for mode in ["A", "B", "Bp", "Bpf"]:
        sig = np.maximum(fit_1h[f"std_{mode}"].values, 100)
        z_abs = np.abs(fit_1h["actual_at_h"].values - fit_1h[f"mean_{mode}"].values) / sig
        q80 = np.quantile(z_abs, 0.8)
        q95 = np.quantile(z_abs, 0.95)
        confs[mode] = {"q80": q80, "q95": q95}
        print(f"  {mode:<5} {q80:<22.3f} {q95:<22.3f}")

    # ====== Step 2: Apply to eval ======
    print("\n[2] Eval set coverage:")
    print(f"  {'Mode':<5} {'cov80 raw':<12} {'cov80 conf':<12} {'cov95 raw':<12} {'cov95 conf':<12}")
    for mode in ["A", "B", "Bp", "Bpf"]:
        eval_m = eval_[eval_["horizon_h"] == 1].copy()
        sig = np.maximum(eval_m[f"std_{mode}"].values, 100)
        mu = eval_m[f"mean_{mode}"].values
        y = eval_m["actual_at_h"].values
        # Raw (assume Normal)
        cov80_raw = ((y >= mu - 1.282 * sig) & (y <= mu + 1.282 * sig)).mean() * 100
        cov95_raw = ((y >= mu - 1.96 * sig) & (y <= mu + 1.96 * sig)).mean() * 100
        # Conformal
        q80 = confs[mode]["q80"]
        q95 = confs[mode]["q95"]
        cov80_c = ((y >= mu - q80 * sig) & (y <= mu + q80 * sig)).mean() * 100
        cov95_c = ((y >= mu - q95 * sig) & (y <= mu + q95 * sig)).mean() * 100
        print(f"  {mode:<5} {cov80_raw:>5.1f}%      {cov80_c:>5.1f}%       "
              f"{cov95_raw:>5.1f}%      {cov95_c:>5.1f}%")

    # ====== Step 3: Tail event coverage ======
    print(f"\n[3] Tail event coverage (eval set, 1h-ahead):")
    eval_1h = eval_[eval_["horizon_h"] == 1].copy()
    for mode in ["A", "B", "Bp", "Bpf"]:
        sig = np.maximum(eval_1h[f"std_{mode}"].values, 100)
        z = (eval_1h["actual_at_h"].values - eval_1h[f"mean_{mode}"].values) / sig
        n3 = (np.abs(z) > 3).sum()
        n5 = (np.abs(z) > 5).sum()
        print(f"  {mode}: |z|>3 {n3:>3} ({n3/len(z)*100:.2f}%), |z|>5 {n5:>3} ({n5/len(z)*100:.2f}%)")

    # ====== Step 4: Outlier day ESS ======
    print(f"\n[4] Outlier day ESS (top 10 |z| max days):")
    eval_["abs_z_A"] = np.abs(eval_["actual_at_h"] - eval_["mean_A"]) / np.maximum(eval_["std_A"], 100)
    day_max = eval_.groupby("date")["abs_z_A"].max().sort_values(ascending=False).head(10)

    # ESS는 trajectory data 직접 봐야 — 여기선 σ 변화로 proxy
    print(f"  {'date':<12} {'max|z|':>7} {'σ_B/σ_A':>9} {'σ_Bpf/σ_A':>11} {'σ_Bp/σ_A':>10}")
    for d, mz in day_max.items():
        day_data = eval_[eval_["date"] == d]
        sig_a = day_data["std_A"].mean()
        sig_b = day_data["std_B"].mean()
        sig_bp = day_data["std_Bp"].mean()
        sig_bpf = day_data["std_Bpf"].mean()
        if sig_a > 0:
            print(f"  {str(d.date()):<12} {mz:>6.2f} "
                  f"{sig_b/sig_a:>8.3f}  {sig_bpf/sig_a:>10.3f}  {sig_bp/sig_a:>9.3f}")

    # ====== Step 5: 1h MAE 종합 ======
    print(f"\n[5] 1h-ahead MAE 종합 (eval):")
    cap_avg = 71800   # MWh per hour (8 site total)
    for mode in ["A", "B", "Bp", "Bpf"]:
        mae = (eval_1h["actual_at_h"] - eval_1h[f"mean_{mode}"]).abs().mean() / 1000
        nmae = mae * 1000 / cap_avg * 100   # rough
        print(f"  {mode}: MAE {mae:.3f} MWh / NMAE {nmae:.2f}%")

    # ====== Step 6: tail support — t-prior가 actual outlier 잡는가 ======
    print(f"\n[6] Tail support — predicted vs actual extreme:")
    print(f"  Eval에서 actual |y - μ_A|/σ_A > 5: {(eval_1h['actual_at_h'].sub(eval_1h['mean_A']).abs() / eval_1h['std_A'].clip(lower=100) > 5).sum()}건")
    print(f"  → 이 사건들 trajectory pool에 표현됐나?")
    print(f"     Normal scenarios: q05~q95가 ±1.96σ → 못 잡음")
    print(f"     t-scenarios (지금 사용 중): q05~q95가 ±1.87σ — 비슷")
    print(f"     → 분포 모양만 두꺼움 (꼬리는 t-distribution heavy)")

    # ====== Save ======
    out_dir = ROOT / "pv/experiments/conformal"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"mode": m, "q80_conformal": v["q80"], "q95_conformal": v["q95"]}
        for m, v in confs.items()
    ]).to_csv(out_dir / "conformal_quantiles.csv", index=False)
    print(f"\n저장: {out_dir / 'conformal_quantiles.csv'}")


if __name__ == "__main__":
    main()
