"""포트폴리오 분포 → 시나리오 풀 (시간 cor 보존 — multivariate).

방식:
  1. Quantile 5: q05, q25, q50, q75, q95 (운영자 직관 시각화)
  2. Multivariate Monte Carlo 1000:
     - ACF (auto-correlation) 추정 from 잔차
     - 24×24 covariance matrix = σ_i × σ_j × ρ(|i-j|)
     - MVN 샘플 → 24h trajectory 1000개
     - 시간 간 cor 보존 (re-weighting에 필수)

입력:
  data/processed/portfolio_predictions.parquet

출력:
  data/processed/scenarios_quantile.parquet  (4090 행, 5 quantile)
  data/processed/scenarios_trajectories.parquet  (date × trajectory_idx × hour, long format)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from scipy.stats import norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
QUANTILE_LABELS = ["worst", "pessimistic", "expected", "optimistic", "best"]
N_TRAJECTORIES = 1000
SEED = 42


def estimate_acf(residuals_per_hour: np.ndarray, max_lag: int = 23) -> np.ndarray:
    """잔차 시퀀스의 autocorrelation function 추정.

    Args:
        residuals_per_hour: (n_total_hours,) — 시간 순 잔차 시퀀스 (daytime 한정 OK)
        max_lag: 최대 lag (24h trajectory면 23)

    Returns:
        acf: (max_lag+1,) starting with 1.0
    """
    x = residuals_per_hour - residuals_per_hour.mean()
    var = x.var()
    if var == 0:
        return np.ones(max_lag + 1)
    acf = np.array([
        (x[: len(x) - lag] * x[lag:]).mean() / var for lag in range(max_lag + 1)
    ])
    return acf


def generate_quantiles(portfolio: pd.DataFrame) -> pd.DataFrame:
    """포트폴리오 분포에서 5개 quantile 추출 (Normal 가정)."""
    df = portfolio[["datetime_kst", "mu_kwh", "sigma_kwh", "actual_kwh", "total_capacity_kw"]].copy()
    z_scores = norm.ppf(QUANTILES)
    for q, z, label in zip(QUANTILES, z_scores, QUANTILE_LABELS):
        df[f"q{int(q*100):02d}_{label}"] = (df["mu_kwh"] + z * df["sigma_kwh"]).clip(lower=0)
    return df


def generate_trajectories(portfolio: pd.DataFrame, acf: np.ndarray,
                          n_samples: int = N_TRAJECTORIES, seed: int = SEED) -> pd.DataFrame:
    """포트폴리오 분포에서 *시간 cor 보존* multivariate trajectory 샘플링.

    각 *day*에 대해:
      mu_24h  = 그날 24h μ
      sigma_24h = 그날 24h σ
      Cov[i,j] = sigma_i × sigma_j × ρ(|i-j|)   (Toeplitz)
      samples ~ MVN(mu_24h, Cov)

    Returns:
        long format: (date, trajectory_idx, datetime_kst, sample_kwh)
    """
    rng = np.random.default_rng(seed)
    df = portfolio.copy()
    df["date"] = df["datetime_kst"].dt.date
    df["hour"] = df["datetime_kst"].dt.hour

    # 24h × n_days matrix 만들기 (daytime row만 채움, 야간은 NaN)
    pivot_mu = df.pivot(index="date", columns="hour", values="mu_kwh")
    pivot_sigma = df.pivot(index="date", columns="hour", values="sigma_kwh")

    # 모든 24h column 보장 (없는 시간 = 야간, μ=σ=0)
    full_hours = list(range(24))
    pivot_mu = pivot_mu.reindex(columns=full_hours).fillna(0)
    pivot_sigma = pivot_sigma.reindex(columns=full_hours).fillna(0)

    print(f"  날짜 수: {len(pivot_mu)}, 24h 시간 grid 보장")

    # ACF 길이 = 24
    rho = np.zeros(24)
    rho[: len(acf)] = acf[:24]
    cor_matrix = toeplitz(rho)   # (24, 24)

    # 양정치 보장 (numerical jitter)
    cor_matrix = (cor_matrix + cor_matrix.T) / 2
    eigvals = np.linalg.eigvalsh(cor_matrix)
    if eigvals.min() < 1e-6:
        cor_matrix += np.eye(24) * (1e-6 - eigvals.min())

    rows = []
    for date_val in pivot_mu.index:
        mu_24 = pivot_mu.loc[date_val].values
        sigma_24 = pivot_sigma.loc[date_val].values
        # σ=0인 시간 (야간): 그 row/col cov=0 → MVN 샘플도 0
        cov = sigma_24[:, None] * cor_matrix * sigma_24[None, :]
        # 양정치 보장
        cov = (cov + cov.T) / 2
        # 샘플
        samples = rng.multivariate_normal(mu_24, cov, size=n_samples)
        samples = np.clip(samples, 0, None)
        # 저장 (long format)
        for traj_idx in range(n_samples):
            for h in range(24):
                # 야간 (σ=0) skip — 데이터 양 줄이기
                if sigma_24[h] == 0 and mu_24[h] == 0:
                    continue
                rows.append({
                    "date": date_val,
                    "hour": h,
                    "trajectory_idx": traj_idx,
                    "sample_kwh": samples[traj_idx, h],
                })

    return pd.DataFrame(rows)


def evaluate_trajectories(portfolio: pd.DataFrame, trajectories: pd.DataFrame, acf: np.ndarray):
    """시나리오 풀 검증 — marginal calibration + cor 패턴."""
    # Marginal: 시간별 mean of 1000 samples ≈ μ
    portfolio_with_h = portfolio.copy()
    portfolio_with_h["date"] = portfolio_with_h["datetime_kst"].dt.date
    portfolio_with_h["hour"] = portfolio_with_h["datetime_kst"].dt.hour

    traj_marginal = trajectories.groupby(["date", "hour"], as_index=False).agg(
        traj_mean=("sample_kwh", "mean"),
        traj_std=("sample_kwh", "std"),
    )
    merged = portfolio_with_h.merge(traj_marginal, on=["date", "hour"])
    mean_diff = (merged.traj_mean - merged.mu_kwh).abs().mean() / 1000
    std_diff = (merged.traj_std - merged.sigma_kwh).abs().mean() / 1000

    print()
    print("=== Trajectory Marginal 검증 ===")
    print(f"  trajectory mean vs analytic μ 차이: {mean_diff:.3f} MWh (작을수록 정확)")
    print(f"  trajectory std vs analytic σ 차이:  {std_diff:.3f} MWh")

    # Coverage check
    portfolio_with_h["q05"] = portfolio_with_h.mu_kwh + norm.ppf(0.05) * portfolio_with_h.sigma_kwh
    portfolio_with_h["q95"] = portfolio_with_h.mu_kwh + norm.ppf(0.95) * portfolio_with_h.sigma_kwh
    cov90 = ((portfolio_with_h.actual_kwh >= portfolio_with_h.q05.clip(lower=0))
             & (portfolio_with_h.actual_kwh <= portfolio_with_h.q95)).mean()
    print(f"  실측이 [q05, q95] 범위 (90%): {cov90*100:.1f}%   (목표 90%)")

    # ACF preview
    print()
    print(f"=== ACF 추정 (잔차 시간 자기상관) ===")
    print(f"  lag 0:  {acf[0]:.3f}  (=1.0)")
    print(f"  lag 1:  {acf[1]:.3f}")
    print(f"  lag 3:  {acf[3]:.3f}")
    print(f"  lag 6:  {acf[6]:.3f}")
    print(f"  lag 12: {acf[12]:.3f}")
    print(f"  lag 23: {acf[23]:.3f}")


def main():
    print("[1/4] 포트폴리오 예측 로딩...")
    pf_path = ROOT / "data" / "processed" / "portfolio_predictions.parquet"
    portfolio = pd.read_parquet(pf_path)
    print(f"  rows: {len(portfolio):,}")

    print("\n[2/4] ACF 추정 (잔차 시간 자기상관)...")
    pf_sorted = portfolio.sort_values("datetime_kst").reset_index(drop=True)
    residuals = (pf_sorted["actual_kwh"] - pf_sorted["mu_kwh"]).values
    acf = estimate_acf(residuals, max_lag=23)
    print(f"  ACF length: {len(acf)}, lag-1 cor: {acf[1]:.3f}")

    print("\n[3/4] Quantile 시나리오 (Normal margin)...")
    quantile_df = generate_quantiles(portfolio)
    out_dir = ROOT / "data" / "processed"
    quantile_path = out_dir / "scenarios_quantile.parquet"
    quantile_df.to_parquet(quantile_path, index=False)
    print(f"  저장: {quantile_path}")

    print(f"\n[4/4] Multivariate Trajectory 샘플 ({N_TRAJECTORIES}개/날)...")
    trajectories = generate_trajectories(portfolio, acf, n_samples=N_TRAJECTORIES)
    traj_path = out_dir / "scenarios_trajectories.parquet"
    trajectories.to_parquet(traj_path, index=False)
    print(f"  저장: {traj_path}")
    print(f"  rows: {len(trajectories):,}")

    evaluate_trajectories(portfolio, trajectories, acf)


if __name__ == "__main__":
    main()
