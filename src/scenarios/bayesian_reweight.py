"""실시간 PV 실측 (시뮬레이션) → 시나리오 trajectory weight 갱신.

Bayesian filter with cloud-aware regime detection (ACF break protection):
  매 시간 t에 PV_actual_t 들어오면:
    likelihood_i = N(traj_i[t] | actual_t, noise_std²)
    weight_i ∝ weight_i × likelihood_i

  *Regime detection (NEW)*:
    cloud(t) >= CLOUD_THRESHOLD (default 7) → "transient" regime
      → weights reset to uniform (시간 상관성 단절, ACF break 방지)
    cloud(t) <  CLOUD_THRESHOLD → standard reweight

A (baseline)         : weights uniform 유지 (D-1 forecast 그대로)
B (re-weighted)      : 표준 reweight (시간 상관성 가정)
B' (regime-aware)    : cloud check → transient면 reset, 아니면 reweight

평가:
  관측 시간 t에서 미래 시간 h (h > t) 예측 정확도 비교
  Horizon별 (h - t) MAE_A vs MAE_B vs MAE_B' → 개선 정량화

입력:
  data/processed/scenarios_trajectories.parquet  (1000 trajectory × 365 day × ~11h)
  data/processed/portfolio_predictions.parquet   (실측 + cloud_mean column)

출력:
  pv/experiments/realtime_simulation/predictions_A_vs_B.parquet (A, B, B')
  pv/experiments/realtime_simulation/horizon_metrics.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

NOISE_STD_KWH = 5_000   # 실측 매칭 허용 오차 (5 MWh, ≈ ResMLP+AdaLN portfolio σ scale)
CLOUD_THRESHOLD = 7.0   # cloud_mean >= 이 값이면 transient regime (ACF break)
ESS_RESAMPLE_FRAC = 0.3   # ESS < n_samples * ESS_RESAMPLE_FRAC면 resample
JITTER_STD_RATIO = 0.05   # resample 후 jitter (trajectory σ의 5%)


# ========== Core Bayesian filter ==========

def reweight_step(weights: np.ndarray,
                  traj_at_t: np.ndarray,
                  observation: float,
                  noise_std: float) -> np.ndarray:
    """One step Bayesian update.

    weight_i ∝ weight_i × N(traj_i[t] | observation, noise_std²)
    """
    log_lik = -0.5 * ((traj_at_t - observation) / noise_std) ** 2
    log_lik -= log_lik.max()  # numerical stability
    new_weights = weights * np.exp(log_lik)
    s = new_weights.sum()
    if s == 0:
        return weights  # all near-zero, keep prior
    return new_weights / s


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple:
    mean = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - mean) ** 2))
    return mean, float(np.sqrt(max(var, 0)))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling — low-variance, deterministic stratification."""
    n = len(weights)
    positions = (np.arange(n) + rng.uniform()) / n
    indices = np.zeros(n, dtype=np.int64)
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < n and j < n:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    if i < n:
        indices[i:] = n - 1
    return indices


def particle_resample(traj_24h: np.ndarray, weights: np.ndarray,
                      rng: np.random.Generator,
                      jitter_ratio: float = JITTER_STD_RATIO) -> tuple:
    """ESS 낮을 때 trajectory pool 재생성 + 작은 jitter (degeneracy 방지).

    Returns:
        new_traj_24h: (n, 24) — resampled trajectories
        new_weights: (n,) — uniform 1/n
    """
    n = traj_24h.shape[0]
    indices = systematic_resample(weights, rng)
    new_traj = traj_24h[indices].copy()
    if jitter_ratio > 0:
        # column별 σ 계산해서 그 비례 jitter (시간별 다른 scale)
        col_std = traj_24h.std(axis=0, keepdims=True)   # (1, 24)
        noise = rng.standard_normal(new_traj.shape) * col_std * jitter_ratio
        new_traj = np.clip(new_traj + noise.astype(new_traj.dtype), 0, None)
    new_weights = np.ones(n, dtype=np.float64) / n
    return new_traj, new_weights


# ========== Simulation ==========

def simulate_day(traj_24h: np.ndarray,
                 actual_24h: np.ndarray,
                 daytime_mask: np.ndarray,
                 cloud_24h: np.ndarray = None,
                 noise_std: float = NOISE_STD_KWH,
                 cloud_threshold: float = CLOUD_THRESHOLD) -> list[dict]:
    """단일 날 시뮬레이션 — A (baseline) vs B (reweight) vs B' (regime-aware reweight).

    Args:
        traj_24h: (1000, 24) — 1000 trajectory × 24h
        actual_24h: (24,) — 그날 실측 (kWh)
        daytime_mask: (24,) — daytime 시간
        cloud_24h: (24,) — cloud level. None이면 B' = B
        noise_std: likelihood width
        cloud_threshold: 이 값 이상 cloud면 transient → reset weights

    Returns:
        rows: list of dicts (observation_time, prediction_hour, mean_A/B/Bp, std_A/B/Bp)
    """
    n_samples = traj_24h.shape[0]
    weights_A = np.ones(n_samples) / n_samples
    weights_B = np.ones(n_samples) / n_samples
    weights_Bp = np.ones(n_samples) / n_samples   # regime-aware
    weights_Bpf = np.ones(n_samples) / n_samples  # B'' particle filter
    traj_Bpf = traj_24h.copy()                    # PF는 trajectory도 갱신
    rng = np.random.default_rng(42)

    if cloud_24h is None:
        cloud_24h = np.zeros(24)   # if missing, B' behaves like B

    rows = []
    for t in range(24):
        for h in range(t + 1, 24):
            if not daytime_mask[h]:
                continue
            mean_A, std_A = weighted_mean_std(traj_24h[:, h], weights_A)
            mean_B, std_B = weighted_mean_std(traj_24h[:, h], weights_B)
            mean_Bp, std_Bp = weighted_mean_std(traj_24h[:, h], weights_Bp)
            mean_Bpf, std_Bpf = weighted_mean_std(traj_Bpf[:, h], weights_Bpf)
            rows.append({
                "observation_time": t,
                "prediction_hour": h,
                "horizon_h": h - t,
                "mean_A": mean_A, "std_A": std_A,
                "mean_B": mean_B, "std_B": std_B,
                "mean_Bp": mean_Bp, "std_Bp": std_Bp,
                "mean_Bpf": mean_Bpf, "std_Bpf": std_Bpf,
                "actual_at_h": float(actual_24h[h]),
                "cloud_at_h": float(cloud_24h[h]) if not np.isnan(cloud_24h[h]) else 0.0,
                "is_daytime_h": bool(daytime_mask[h]),
            })

        # t시간 실측 들어오면 weights 갱신
        if daytime_mask[t]:
            # B: 항상 표준 reweight
            weights_B = reweight_step(weights_B, traj_24h[:, t], actual_24h[t], noise_std)
            # B'': particle filter — reweight 후 ESS check, 낮으면 resample
            weights_Bpf = reweight_step(weights_Bpf, traj_Bpf[:, t], actual_24h[t], noise_std)
            ess = 1.0 / (weights_Bpf ** 2).sum() if (weights_Bpf ** 2).sum() > 0 else 0
            if ess < ESS_RESAMPLE_FRAC * n_samples:
                traj_Bpf, weights_Bpf = particle_resample(traj_Bpf, weights_Bpf, rng)
            # B': cloud(t) 확인. 높으면 reset, 낮으면 reweight
            cloud_t = cloud_24h[t] if not np.isnan(cloud_24h[t]) else 0.0
            if cloud_t >= cloud_threshold:
                # Transient regime — 시간 상관성 단절
                weights_Bp = np.ones(n_samples) / n_samples
            else:
                weights_Bp = reweight_step(weights_Bp, traj_24h[:, t], actual_24h[t], noise_std)

    return rows


def main():
    print("[1/4] 데이터 로딩...")
    traj_path = ROOT / "data" / "processed" / "scenarios_trajectories.parquet"
    portfolio_path = ROOT / "data" / "processed" / "portfolio_predictions.parquet"
    trajectories = pd.read_parquet(traj_path)
    portfolio = pd.read_parquet(portfolio_path)

    portfolio["date"] = portfolio["datetime_kst"].dt.date
    portfolio["hour"] = portfolio["datetime_kst"].dt.hour
    has_cloud = "cloud_mean" in portfolio.columns
    print(f"  trajectories rows: {len(trajectories):,}")
    print(f"  portfolio rows: {len(portfolio):,}, cloud column: {has_cloud}")

    print("\n[2/4] 시뮬레이션 (날짜별)...")
    all_rows = []
    n_dates = trajectories["date"].nunique()
    n_done = 0

    for date_val, date_traj in trajectories.groupby("date"):
        pivot = date_traj.pivot_table(
            index="trajectory_idx", columns="hour", values="sample_kwh"
        )
        pivot = pivot.reindex(columns=range(24)).fillna(0)
        traj_24h = pivot.values

        date_pf = portfolio[portfolio.date == date_val]
        actual_24h = np.zeros(24)
        daytime_mask = np.zeros(24, dtype=bool)
        cloud_24h = np.zeros(24)
        for _, r in date_pf.iterrows():
            h = int(r.hour)
            actual_24h[h] = float(r.actual_kwh)
            daytime_mask[h] = True
            if has_cloud and not pd.isna(r.cloud_mean):
                cloud_24h[h] = float(r.cloud_mean)

        rows = simulate_day(traj_24h, actual_24h, daytime_mask, cloud_24h)
        for row in rows:
            row["date"] = date_val
            all_rows.append(row)

        n_done += 1
        if n_done % 50 == 0:
            print(f"  {n_done}/{n_dates} 날 처리됨")

    print(f"  완료: {n_done}/{n_dates}")

    print("\n[3/4] 결과 저장 + 평가...")
    out_dir = ROOT / "pv" / "experiments" / "realtime_simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    print(f"  total prediction rows: {len(df):,}")

    df["abs_err_A"] = (df["mean_A"] - df["actual_at_h"]).abs()
    df["abs_err_B"] = (df["mean_B"] - df["actual_at_h"]).abs()
    df["abs_err_Bp"] = (df["mean_Bp"] - df["actual_at_h"]).abs()
    df["abs_err_Bpf"] = (df["mean_Bpf"] - df["actual_at_h"]).abs()

    df.to_parquet(out_dir / "predictions_A_vs_B.parquet", index=False)
    print(f"  저장: {out_dir / 'predictions_A_vs_B.parquet'}")

    print("\n[4/4] Horizon별 metric 집계 (A vs B vs B' vs B'' particle filter)...")
    horizon_metrics = (
        df.groupby("horizon_h", as_index=False).agg(
            n=("actual_at_h", "size"),
            mae_A=("abs_err_A", "mean"),
            mae_B=("abs_err_B", "mean"),
            mae_Bp=("abs_err_Bp", "mean"),
            mae_Bpf=("abs_err_Bpf", "mean"),
            std_A_avg=("std_A", "mean"),
            std_B_avg=("std_B", "mean"),
            std_Bp_avg=("std_Bp", "mean"),
            std_Bpf_avg=("std_Bpf", "mean"),
        )
    )
    for x in ["B", "Bp", "Bpf"]:
        horizon_metrics[f"{x}_improve_%"] = (horizon_metrics["mae_A"] - horizon_metrics[f"mae_{x}"]) / horizon_metrics["mae_A"] * 100
        horizon_metrics[f"{x}_sigred_%"] = (horizon_metrics["std_A_avg"] - horizon_metrics[f"std_{x}_avg"]) / horizon_metrics["std_A_avg"] * 100
    horizon_metrics.to_csv(out_dir / "horizon_metrics.csv", index=False)
    print(f"  저장: {out_dir / 'horizon_metrics.csv'}")

    print()
    print("=" * 110)
    print("A | B (std) | B' (regime) | B'' (PF resample, ESS<30%)")
    print("=" * 110)
    print(f"{'h':>3} {'n':>7} {'MAE_A':>6} {'MAE_B':>6} {'MAE_Bp':>7} {'MAE_Bpf':>8} | "
          f"{'B%':>5} {'Bp%':>5} {'Bpf%':>5} | {'σB%':>5} {'σBp%':>5} {'σBpf%':>6}")
    print("-" * 110)
    for _, r in horizon_metrics.iterrows():
        print(f"{int(r['horizon_h']):>3} {int(r['n']):>7,} "
              f"{r['mae_A']/1000:>6.2f} {r['mae_B']/1000:>6.2f} {r['mae_Bp']/1000:>7.2f} {r['mae_Bpf']/1000:>8.2f} | "
              f"{r['B_improve_%']:>+4.1f}% {r['Bp_improve_%']:>+4.1f}% {r['Bpf_improve_%']:>+4.1f}% | "
              f"{r['B_sigred_%']:>+4.1f}% {r['Bp_sigred_%']:>+4.1f}% {r['Bpf_sigred_%']:>+5.1f}%")
    print("-" * 110)

    for x in ["A", "B", "Bp", "Bpf"]:
        overall = df[f"abs_err_{x}"].mean() / 1000
        if x == "A":
            print(f"전체: MAE_A {overall:.3f}", end="")
            overall_A = overall
        else:
            print(f" / MAE_{x} {overall:.3f} ({(overall_A-overall)/overall_A*100:+.1f}%)", end="")
    print()

    # cloud-heavy 시간대만 따로 비교
    cloud_heavy = df[df["cloud_at_h"] >= 7.0]
    if len(cloud_heavy) > 0:
        print(f"\nCloud≥7 시간 (n={len(cloud_heavy):,}):")
        a = cloud_heavy["abs_err_A"].mean() / 1000
        for x in ["A", "B", "Bp", "Bpf"]:
            v = cloud_heavy[f"abs_err_{x}"].mean() / 1000
            if x == "A":
                print(f"  A: {v:.3f}", end="")
            else:
                print(f"  / {x}: {v:.3f} ({(a-v)/a*100:+.1f}%)", end="")
        print()


if __name__ == "__main__":
    main()
