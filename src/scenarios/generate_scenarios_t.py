"""Multivariate Student-t 기반 시나리오 생성.

vs generate_scenarios.py (Normal-based):
  - Sampling: MVN → multivariate t (heavier tails)
  - Marginally: x_i ~ μ_i + σ_i * (loc_t + scale_t * t_std_i(ν))
  - Correlation: 같은 Toeplitz ACF

post_hoc_student_t.py에서 fit된 ν, loc_t, scale_t 사용:
  ν = 3.73, loc_t = 0.09, scale_t = 0.688 (1-6월 fit)

출력:
  data/processed/scenarios_trajectories.parquet  ← 덮어씀
  data/processed/scenarios_quantile.parquet      ← 덮어씀
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz, cholesky
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
QUANTILE_LABELS = ["worst", "pessimistic", "expected", "optimistic", "best"]
N_TRAJECTORIES = 1000
SEED = 42

# Student-t parameters (post_hoc_student_t.py에서 fit)
T_DF = 3.73
T_LOC = 0.09
T_SCALE = 0.688


def estimate_acf(residuals, max_lag=23):
    x = residuals - residuals.mean()
    var = x.var()
    if var == 0:
        return np.ones(max_lag + 1)
    return np.array([
        (x[: len(x) - lag] * x[lag:]).mean() / var for lag in range(max_lag + 1)
    ])


def generate_quantiles_t(portfolio):
    """Quantile 5개 (Normal 대신 Student-t margin)."""
    df = portfolio[["datetime_kst", "mu_kwh", "sigma_kwh", "actual_kwh", "total_capacity_kw"]].copy()
    for q, label in zip(QUANTILES, QUANTILE_LABELS):
        # x ~ μ + σ * (loc + scale * t_std)
        # x_q = μ + σ * (loc + scale * t.ppf(q))
        z_q = T_LOC + T_SCALE * stats.t.ppf(q, df=T_DF)
        df[f"q{int(q*100):02d}_{label}"] = (df["mu_kwh"] + z_q * df["sigma_kwh"]).clip(lower=0)
    return df


def sample_mvt_per_day(mu_24, sigma_24, cor_matrix, n_samples, rng):
    """Multivariate Student-t sampling.

    Sampling x_h = μ_h + σ_h * (loc_t + scale_t * z_h)
    where z = MVT(df=ν, loc=0, shape=correlation matrix, dim=24)

    Approach (Cholesky):
        v = MVN(0, cor_matrix)              # (n_samples, 24)
        w = chi²(ν)                          # (n_samples,)
        z_std = v * sqrt(ν / w)              # MVT_std with df ν, shape=cor
        z_scaled = T_LOC + T_SCALE * z_std
        x = μ + σ * z_scaled
    """
    n_dim = len(mu_24)
    # σ=0인 시간 (야간) row/col은 0으로 처리
    active = sigma_24 > 0
    if active.sum() == 0:
        return np.zeros((n_samples, n_dim))

    # Effective subset
    mu_eff = mu_24[active]
    sig_eff = sigma_24[active]
    cor_eff = cor_matrix[np.ix_(active, active)]
    cor_eff = (cor_eff + cor_eff.T) / 2
    eig = np.linalg.eigvalsh(cor_eff)
    if eig.min() < 1e-6:
        cor_eff += np.eye(active.sum()) * (1e-6 - eig.min())

    L = cholesky(cor_eff, lower=True)
    n_eff = active.sum()
    v = rng.standard_normal((n_samples, n_eff)) @ L.T   # MVN(0, cor)
    w = rng.chisquare(T_DF, size=n_samples)
    z_std = v * np.sqrt(T_DF / w)[:, None]              # MVT_std
    z_scaled = T_LOC + T_SCALE * z_std
    x_eff = mu_eff[None, :] + sig_eff[None, :] * z_scaled

    out = np.zeros((n_samples, n_dim))
    out[:, active] = np.clip(x_eff, 0, None)
    return out


def generate_trajectories_t(portfolio, acf, n_samples=N_TRAJECTORIES, seed=SEED):
    rng = np.random.default_rng(seed)
    df = portfolio.copy()
    df["date"] = df["datetime_kst"].dt.date
    df["hour"] = df["datetime_kst"].dt.hour

    pivot_mu = df.pivot(index="date", columns="hour", values="mu_kwh").reindex(columns=range(24)).fillna(0)
    pivot_sigma = df.pivot(index="date", columns="hour", values="sigma_kwh").reindex(columns=range(24)).fillna(0)
    print(f"  날짜: {len(pivot_mu)}, 24h grid")

    rho = np.zeros(24)
    rho[: len(acf)] = acf[:24]
    cor_matrix = toeplitz(rho)
    cor_matrix = (cor_matrix + cor_matrix.T) / 2
    eig = np.linalg.eigvalsh(cor_matrix)
    if eig.min() < 1e-6:
        cor_matrix += np.eye(24) * (1e-6 - eig.min())

    rows = []
    for date_val in pivot_mu.index:
        mu_24 = pivot_mu.loc[date_val].values
        sig_24 = pivot_sigma.loc[date_val].values
        samples = sample_mvt_per_day(mu_24, sig_24, cor_matrix, n_samples, rng)
        for traj_idx in range(n_samples):
            for h in range(24):
                if sig_24[h] == 0 and mu_24[h] == 0:
                    continue
                rows.append({
                    "date": date_val, "hour": h,
                    "trajectory_idx": traj_idx,
                    "sample_kwh": samples[traj_idx, h],
                })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Student-t 기반 시나리오 생성")
    print(f"  ν={T_DF}, loc_t={T_LOC}, scale_t={T_SCALE}")
    print("=" * 70)

    print("\n[1/4] 포트폴리오 로딩...")
    pf = pd.read_parquet(ROOT / "data/processed/portfolio_predictions.parquet")
    print(f"  rows: {len(pf):,}")

    print("\n[2/4] ACF 추정...")
    pf_sorted = pf.sort_values("datetime_kst").reset_index(drop=True)
    residuals = (pf_sorted["actual_kwh"] - pf_sorted["mu_kwh"]).values
    acf = estimate_acf(residuals, max_lag=23)
    print(f"  lag-1 cor: {acf[1]:.3f}")

    print("\n[3/4] Quantile 시나리오 (Student-t margin)...")
    quantile_df = generate_quantiles_t(pf)
    out_dir = ROOT / "data/processed"
    quantile_df.to_parquet(out_dir / "scenarios_quantile.parquet", index=False)
    print(f"  저장: {out_dir / 'scenarios_quantile.parquet'}")

    print(f"\n[4/4] Multivariate t Trajectory ({N_TRAJECTORIES}/날)...")
    traj_df = generate_trajectories_t(pf, acf, N_TRAJECTORIES)
    traj_df.to_parquet(out_dir / "scenarios_trajectories.parquet", index=False)
    print(f"  저장: {out_dir / 'scenarios_trajectories.parquet'}")
    print(f"  rows: {len(traj_df):,}")

    # Quick verification
    pf["date"] = pf["datetime_kst"].dt.date
    pf["hour"] = pf["datetime_kst"].dt.hour
    traj_marginal = traj_df.groupby(["date", "hour"], as_index=False).agg(
        traj_mean=("sample_kwh", "mean"),
        traj_std=("sample_kwh", "std"),
        traj_q05=("sample_kwh", lambda s: s.quantile(0.05)),
        traj_q95=("sample_kwh", lambda s: s.quantile(0.95)),
    )
    merged = pf.merge(traj_marginal, on=["date", "hour"])
    print(f"\n=== Marginal verification ===")
    # NB: Marginal mean should match μ + σ * loc_t (slight shift)
    expected_mean_shift = T_LOC * merged["sigma_kwh"]
    mean_diff = (merged["traj_mean"] - merged["mu_kwh"] - expected_mean_shift).abs().mean() / 1000
    # marginal std: σ * scale_t * sqrt(ν/(ν-2))
    expected_std = merged["sigma_kwh"] * T_SCALE * np.sqrt(T_DF / (T_DF - 2))
    std_diff = (merged["traj_std"] - expected_std).abs().mean() / 1000
    print(f"  trajectory mean - (μ + loc_t*σ): {mean_diff:.3f} MWh")
    print(f"  trajectory std vs expected (scale_t*σ*√(ν/(ν-2))): {std_diff:.3f} MWh")

    cov90 = ((merged.actual_kwh >= merged.traj_q05) & (merged.actual_kwh <= merged.traj_q95)).mean() * 100
    print(f"  실측이 [traj q05, q95] 범위 (90%): {cov90:.1f}%")


if __name__ == "__main__":
    main()
