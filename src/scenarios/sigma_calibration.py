"""B4: Horizon × regime별 σ calibration.

목적: cov80 / cov95 정확성 보장 — over/under-confident σ 보정.

방법:
  1. (horizon h, cloud regime r) 셀별 z = (actual - μ) / σ 측정
  2. spread(h, r) = std(z | h, r)  → 1이면 well-calibrated
  3. 보정: σ_cal = σ * spread(h, r)
  4. cov80 / cov95 재측정

Regime 정의 (data-driven):
  clear:  cloud_at_h < 3
  mild:   3 <= cloud_at_h < 7
  heavy:  cloud_at_h >= 7

각 mode (A, B, B') 별로 따로 calibrate.

Train/Eval split (data leakage 방지):
  fit:  2025년 1-6월
  eval: 2025년 7-12월
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def cloud_regime(c):
    if c < 3:
        return "clear"
    elif c < 7:
        return "mild"
    else:
        return "heavy"


def main():
    print("=" * 70)
    print("B4: Horizon × Regime σ Calibration")
    print("=" * 70)

    df = pd.read_parquet(ROOT / "pv/experiments/realtime_simulation/predictions_A_vs_B.parquet")
    df["regime"] = df["cloud_at_h"].apply(cloud_regime)
    df["month"] = pd.to_datetime(df["date"]).dt.month
    fit = df[df["month"] <= 6].copy()
    eval_ = df[df["month"] >= 7].copy()
    print(f"\n[Data] total {len(df):,}, fit {len(fit):,}, eval {len(eval_):,}")

    print(f"\nRegime 분포 (fit):")
    print(fit["regime"].value_counts())

    # Compute coverage-based scale per (horizon, regime, mode)
    # spread = q_{0.8}(|z|) / 1.28  → exactly 80% coverage post-cal
    # σ_floor: ESS collapse (σ→0) 보호
    print("\n[1] Coverage-based calibration parameters 계산 (fit)...")
    cal = {}
    SIGMA_FLOOR = 100.0   # 100 kWh (0.1 MWh) — σ collapse 방지
    for mode in ["A", "B", "Bp"]:
        for h in fit["horizon_h"].unique():
            for r in ["clear", "mild", "heavy"]:
                sub = fit[(fit["horizon_h"] == h) & (fit["regime"] == r)]
                if len(sub) < 30:
                    continue
                sig = np.maximum(sub[f"std_{mode}"].values, SIGMA_FLOOR)
                z = (sub["actual_at_h"].values - sub[f"mean_{mode}"].values) / sig
                # 80% coverage target
                q80 = np.quantile(np.abs(z), 0.8)
                spread_80 = q80 / 1.28
                # 95% coverage target (다른 metric)
                q95 = np.quantile(np.abs(z), 0.95)
                spread_95 = q95 / 1.96
                cal[(mode, h, r)] = {"spread_80": float(spread_80), "spread_95": float(spread_95)}

    # Apply calibration
    print("\n[2] eval에 보정 적용 + cov 측정...")
    print(f"{'mode':<5} {'h':<3} {'regime':<8} {'n':>5} | {'cov80 raw → cal':<20} | {'cov95 raw → cal':<20} | s80 / s95")
    print("-" * 100)

    summary = []
    for mode in ["A", "B", "Bp"]:
        for r in ["clear", "mild", "heavy"]:
            for h in [1, 3, 6, 12]:
                cell = eval_[(eval_["horizon_h"] == h) & (eval_["regime"] == r)]
                if len(cell) < 20:
                    continue
                params = cal.get((mode, h, r))
                if params is None:
                    continue
                s80 = params["spread_80"]
                s95 = params["spread_95"]
                mu = cell[f"mean_{mode}"].values
                sig_raw = np.maximum(cell[f"std_{mode}"].values, SIGMA_FLOOR)
                sig_cal80 = sig_raw * s80
                sig_cal95 = sig_raw * s95
                y = cell["actual_at_h"].values
                cov80_raw = ((y >= mu - 1.28 * sig_raw) & (y <= mu + 1.28 * sig_raw)).mean() * 100
                cov80_cal = ((y >= mu - 1.28 * sig_cal80) & (y <= mu + 1.28 * sig_cal80)).mean() * 100
                cov95_raw = ((y >= mu - 1.96 * sig_raw) & (y <= mu + 1.96 * sig_raw)).mean() * 100
                cov95_cal = ((y >= mu - 1.96 * sig_cal95) & (y <= mu + 1.96 * sig_cal95)).mean() * 100
                summary.append({
                    "mode": mode, "horizon": h, "regime": r, "n": len(cell),
                    "spread_80": s80, "spread_95": s95,
                    "cov80_raw": cov80_raw, "cov80_cal": cov80_cal,
                    "cov95_raw": cov95_raw, "cov95_cal": cov95_cal,
                })
                print(f"{mode:<5} {h:<3} {r:<8} {len(cell):>5} | "
                      f"{cov80_raw:>5.1f}% → {cov80_cal:>5.1f}%       | "
                      f"{cov95_raw:>5.1f}% → {cov95_cal:>5.1f}%       | "
                      f"{s80:.2f} / {s95:.2f}")

    summary_df = pd.DataFrame(summary)

    # Aggregated 요약
    print("\n[3] Aggregated by mode (eval set):")
    for mode in ["A", "B", "Bp"]:
        sub = summary_df[summary_df["mode"] == mode]
        if len(sub) == 0:
            continue
        cov80r = (sub["cov80_raw"] * sub["n"]).sum() / sub["n"].sum()
        cov80c = (sub["cov80_cal"] * sub["n"]).sum() / sub["n"].sum()
        cov95r = (sub["cov95_raw"] * sub["n"]).sum() / sub["n"].sum()
        cov95c = (sub["cov95_cal"] * sub["n"]).sum() / sub["n"].sum()
        s80_avg = (sub["spread_80"] * sub["n"]).sum() / sub["n"].sum()
        s95_avg = (sub["spread_95"] * sub["n"]).sum() / sub["n"].sum()
        print(f"  Mode {mode}: cov80 {cov80r:.1f}% → {cov80c:.1f}%   "
              f"cov95 {cov95r:.1f}% → {cov95c:.1f}%   "
              f"s80={s80_avg:.2f}, s95={s95_avg:.2f}")

    # Save calibration table
    out_dir = ROOT / "pv/experiments/sigma_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_rows = []
    for (mode, h, r), params in cal.items():
        cal_rows.append({"mode": mode, "horizon_h": h, "regime": r,
                         "spread_80": params["spread_80"],
                         "spread_95": params["spread_95"]})
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(out_dir / "calibration_table.csv", index=False)
    summary_df.to_csv(out_dir / "eval_summary.csv", index=False)
    print(f"\n저장: {out_dir / 'calibration_table.csv'}")
    print(f"      {out_dir / 'eval_summary.csv'}")


if __name__ == "__main__":
    main()
