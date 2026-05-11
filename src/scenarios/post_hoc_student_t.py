"""B1 Option 2: ResMLP+AdaLN Normal output → post-hoc Student-t 변환.

방법:
  1. ResMLP+AdaLN test predictions (μ, σ) 로드
  2. fit set (2025 1-6월): z = (actual - μ) / σ 계산
  3. Student-t 분포 fit: ν, loc_t, scale_t = scipy.stats.t.fit(z)
  4. eval set (2025 7-12월): 같은 μ, σ, *분포 가정만 변경*
       - 새 80% interval: μ ± t.ppf(0.9, ν, loc_t, scale_t) * σ
       - 새 95% interval: μ ± t.ppf(0.975, ν, loc_t, scale_t) * σ
  5. metric 비교:
       - cov80, cov95
       - tail event coverage: |z|>3 / |z|>5 비율
       - NLL: Normal vs t
       - CRPS: Normal vs t (Monte Carlo)

목표:
  Normal이 얇은 꼬리라 outlier z=-8 같은 사건 안 잡음
  t-distribution은 heavy-tail이라 *동일 σ로도* 더 wide tail
  → tail coverage 개선, NLL ↓
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def normal_nll(y, mu, sig):
    return 0.5 * np.log(2 * np.pi * sig ** 2) + ((y - mu) ** 2) / (2 * sig ** 2)


def t_nll(y, mu, sig, nu, loc_t, scale_t):
    """Student-t scaled distribution: y = μ + σ * z, z ~ t(ν, loc_t, scale_t).

    Density: p(y) = (1/σ) * f_t((y-μ)/σ; ν, loc_t, scale_t)
    NLL: -log(1/σ) - log(f_t(z; ν, loc_t, scale_t))
       = log(σ) - log(f_t(z))
    """
    z = (y - mu) / sig
    log_pdf_z = stats.t.logpdf(z, df=nu, loc=loc_t, scale=scale_t)
    return np.log(sig) - log_pdf_z


def crps_normal_closed(y, mu, sig):
    """Closed-form CRPS for Normal."""
    z = (y - mu) / sig
    phi = stats.norm.pdf(z)
    Phi = stats.norm.cdf(z)
    return sig * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))


def crps_student_t_mc(y, mu, sig, nu, loc_t, scale_t, n_samples=300, rng=None):
    """Monte Carlo CRPS for scaled t-distribution.

    Definition: CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Sample z
    z = stats.t.rvs(df=nu, loc=loc_t, scale=scale_t, size=(n_samples, len(y)), random_state=rng)
    # X = mu + sig * z (broadcast)
    X = mu[None, :] + sig[None, :] * z   # (n_samples, len(y))
    # E[|X - y|]
    term1 = np.abs(X - y[None, :]).mean(axis=0)
    # E[|X - X'|] — pairwise within sample
    z2 = stats.t.rvs(df=nu, loc=loc_t, scale=scale_t, size=(n_samples, len(y)), random_state=rng)
    X2 = mu[None, :] + sig[None, :] * z2
    term2 = np.abs(X - X2).mean(axis=0)
    return term1 - 0.5 * term2


def main():
    print("=" * 70)
    print("B1 Option 2: post-hoc Student-t conversion")
    print("=" * 70)

    pred = pd.read_parquet(ROOT / "pv/experiments/resmlp_adaln/test_predictions.parquet")
    print(f"\n[Data] {len(pred):,} predictions (cols: {pred.columns.tolist()})")

    # 필요한 컬럼: datetime_kst, site, cf (actual), pred_cf (mu), pred_std_cf (sigma), site_capacity_kw
    pred = pred.rename(columns={"cf": "actual", "pred_cf": "mu", "pred_std_cf": "sigma"}).copy()
    pred["sigma"] = pred["sigma"].clip(lower=1e-3)
    pred["z"] = (pred["actual"] - pred["mu"]) / pred["sigma"]
    pred["month"] = pred["datetime_kst"].dt.month

    fit = pred[pred["month"] <= 6].copy()
    eval_ = pred[pred["month"] >= 7].copy()
    print(f"  fit (1-6월): {len(fit):,}, eval (7-12월): {len(eval_):,}")

    # ====== Step 1: Fit Student-t on fit set ======
    print("\n[Step 1] Fit Student-t to z (fit set)...")
    z_fit = fit["z"].values
    nu, loc_t, scale_t = stats.t.fit(z_fit)
    print(f"  ν (df) = {nu:.3f} (Normal에 가까울수록 ν → ∞)")
    print(f"  loc   = {loc_t:.4f}  (이상적 0)")
    print(f"  scale = {scale_t:.4f}  (이상적 1, Normal vs t 폭 차이)")

    # Compare empirical z-distribution
    print(f"\n  fit z 분포: mean={z_fit.mean():.3f}, std={z_fit.std():.3f}, "
          f"q01={np.quantile(z_fit, 0.01):.2f}, q99={np.quantile(z_fit, 0.99):.2f}")
    print(f"  fit |z|>3 비율: {(np.abs(z_fit) > 3).mean()*100:.2f}% "
          f"(Normal이면 0.27%, t(ν={nu:.1f})이면 {2*stats.t.sf(3, nu, loc_t, scale_t)*100:.2f}%)")
    print(f"  fit |z|>5 비율: {(np.abs(z_fit) > 5).mean()*100:.2f}% "
          f"(Normal이면 ~0%, t이면 {2*stats.t.sf(5, nu, loc_t, scale_t)*100:.3f}%)")

    # ====== Step 2: Apply to eval set ======
    print("\n[Step 2] eval set 평가 (Normal vs Student-t)...")
    y_e = eval_["actual"].values
    mu_e = eval_["mu"].values
    sig_e = eval_["sigma"].values
    cap_e = eval_["site_capacity_kw"].values
    z_e = eval_["z"].values

    # === Normal coverage ===
    cov80_n = ((y_e >= mu_e - 1.282 * sig_e) & (y_e <= mu_e + 1.282 * sig_e)).mean() * 100
    cov95_n = ((y_e >= mu_e - 1.96 * sig_e) & (y_e <= mu_e + 1.96 * sig_e)).mean() * 100

    # === Student-t coverage ===
    # y = μ + σ * z, z ~ t(ν, loc_t, scale_t)
    # interval: μ ± σ * t.ppf(0.9 / 0.975, ν, loc_t, scale_t)
    #   (q for upper bound)
    q_lo_80 = stats.t.ppf(0.10, df=nu, loc=loc_t, scale=scale_t)
    q_hi_80 = stats.t.ppf(0.90, df=nu, loc=loc_t, scale=scale_t)
    q_lo_95 = stats.t.ppf(0.025, df=nu, loc=loc_t, scale=scale_t)
    q_hi_95 = stats.t.ppf(0.975, df=nu, loc=loc_t, scale=scale_t)
    print(f"  Student-t 80% quantiles: [{q_lo_80:.3f}, {q_hi_80:.3f}]  (Normal은 ±1.282)")
    print(f"  Student-t 95% quantiles: [{q_lo_95:.3f}, {q_hi_95:.3f}]  (Normal은 ±1.960)")

    cov80_t = ((y_e >= mu_e + sig_e * q_lo_80) & (y_e <= mu_e + sig_e * q_hi_80)).mean() * 100
    cov95_t = ((y_e >= mu_e + sig_e * q_lo_95) & (y_e <= mu_e + sig_e * q_hi_95)).mean() * 100

    # === NLL ===
    nll_n = normal_nll(y_e, mu_e, sig_e).mean()
    nll_t = t_nll(y_e, mu_e, sig_e, nu, loc_t, scale_t).mean()

    # === CRPS (Monte Carlo, 300 samples) ===
    rng = np.random.default_rng(42)
    crps_n = crps_normal_closed(y_e, mu_e, sig_e).mean()
    print(f"  CRPS Monte Carlo 계산 (n_samples=300, len(eval)={len(y_e):,})...")
    crps_t = crps_student_t_mc(y_e, mu_e, sig_e, nu, loc_t, scale_t, n_samples=300, rng=rng).mean()

    print(f"\n=== Coverage 비교 (eval 7-12월) ===")
    print(f"  cov80:  Normal {cov80_n:.1f}% → Student-t {cov80_t:.1f}%  (목표 80%)")
    print(f"  cov95:  Normal {cov95_n:.1f}% → Student-t {cov95_t:.1f}%  (목표 95%)")
    print(f"\n=== Probabilistic score 비교 ===")
    print(f"  NLL:   Normal {nll_n:.4f} → Student-t {nll_t:.4f}  (낮을수록 좋음)")
    print(f"  CRPS:  Normal {crps_n:.4f} → Student-t {crps_t:.4f}  (낮을수록 좋음)")
    print(f"  → NLL Δ={nll_t - nll_n:+.4f}, CRPS Δ={crps_t - crps_n:+.4f}")

    # === Tail event coverage ===
    tail3_n_in_int = ((y_e >= mu_e - 3 * sig_e) & (y_e <= mu_e + 3 * sig_e)).mean() * 100
    tail3_t_in_int = ((y_e >= mu_e + sig_e * stats.t.ppf(stats.norm.cdf(-3), df=nu, loc=loc_t, scale=scale_t))
                      & (y_e <= mu_e + sig_e * stats.t.ppf(stats.norm.cdf(3), df=nu, loc=loc_t, scale=scale_t))).mean() * 100
    n_tail3 = (np.abs(z_e) > 3).sum()
    n_tail5 = (np.abs(z_e) > 5).sum()
    print(f"\n=== Tail event 분포 (eval set, 같은 ν) ===")
    print(f"  |z|>3 actual 발생: {n_tail3:>4} ({n_tail3/len(z_e)*100:.2f}%) — "
          f"Normal 예측 0.27%, t(ν={nu:.1f}) 예측 {2*stats.t.sf(3, nu, loc_t, scale_t)*100:.2f}%")
    print(f"  |z|>5 actual 발생: {n_tail5:>4} ({n_tail5/len(z_e)*100:.2f}%) — "
          f"Normal 예측 ~0%,    t(ν={nu:.1f}) 예측 {2*stats.t.sf(5, nu, loc_t, scale_t)*100:.3f}%")

    # === Outlier day별 prior tail support ===
    print(f"\n=== Outlier day prior tail support 비교 ===")
    eval_["abs_z"] = np.abs(eval_["z"])
    eval_["date"] = eval_["datetime_kst"].dt.date
    day_max_z = eval_.groupby("date")["abs_z"].max().sort_values(ascending=False).head(10)
    print(f"  Top 10 outlier day (max |z|):")
    print(f"  {'date':<12} {'max|z|':>7} {'Normal CDF tail':>17} {'t CDF tail':>13}")
    for d, mz in day_max_z.items():
        cdf_n = 2 * stats.norm.sf(mz)
        cdf_t = 2 * stats.t.sf(mz, df=nu, loc=loc_t, scale=scale_t)
        print(f"  {str(d):<12} {mz:>6.2f} "
              f"{cdf_n:>16.2e}  {cdf_t:>12.2e}")

    # === MAE 1h-ahead 비교는 같음 (μ 변경 없음) ===
    mae = (np.abs(y_e - mu_e) * cap_e).sum() / cap_e.sum() * 100
    print(f"\n=== MAE / NMAE (μ 동일이라 변화 없음) ===")
    print(f"  Site NMAE (eval): {mae:.2f}%   (참고 — μ 변경 없음)")

    # Save calibration
    out_dir = ROOT / "pv/experiments/post_hoc_t"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "nu": nu, "loc_t": loc_t, "scale_t": scale_t,
        "n_fit": len(fit), "n_eval": len(eval_),
        "cov80_normal": cov80_n, "cov80_t": cov80_t,
        "cov95_normal": cov95_n, "cov95_t": cov95_t,
        "nll_normal": nll_n, "nll_t": nll_t,
        "crps_normal": crps_n, "crps_t": crps_t,
    }]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\n저장: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
