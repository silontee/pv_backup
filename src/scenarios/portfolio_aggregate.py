"""사이트별 PV 분포 → 포트폴리오 합산.

독립 가정 (NGBoost 잔차 cor=0.03 측정 기반):
  μ_total = Σ μ_i × capacity_i           (kWh, 단위 변환)
  σ²_total = Σ (σ_i × capacity_i)²
  σ_total = √Σ(σ_i × capacity_i)²

→ 8 사이트 → 1 포트폴리오 (시간별)

입력:
  pv/experiments/ngboost_baseline/test_predictions.parquet
    columns: datetime_kst, site, cf, pred_cf, pred_std_cf, site_capacity_kw

출력:
  data/processed/portfolio_predictions.parquet
    columns: datetime_kst, mu_kwh, sigma_kwh, actual_kwh, total_capacity_kw
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


def aggregate_portfolio(predictions: pd.DataFrame) -> pd.DataFrame:
    """사이트별 cf 분포 → 포트폴리오 kWh 분포.

    Args:
        predictions: columns = [datetime_kst, site, cf, pred_cf, pred_std_cf, site_capacity_kw]

    Returns:
        columns = [datetime_kst, mu_kwh, sigma_kwh, actual_kwh, total_capacity_kw]
    """
    df = predictions.copy()
    # cf → kWh 변환 (사이트별)
    df["mu_i_kwh"] = df["pred_cf"] * df["site_capacity_kw"]
    df["sigma_i_kwh"] = df["pred_std_cf"] * df["site_capacity_kw"]
    df["actual_i_kwh"] = df["cf"] * df["site_capacity_kw"]

    # 시간별 합산
    portfolio = df.groupby("datetime_kst", as_index=False).agg(
        mu_kwh=("mu_i_kwh", "sum"),
        var_kwh=("sigma_i_kwh", lambda s: (s ** 2).sum()),  # 독립 가정: Σσ²
        actual_kwh=("actual_i_kwh", "sum"),
        total_capacity_kw=("site_capacity_kw", "sum"),
        n_sites=("site", "nunique"),
    )
    portfolio["sigma_kwh"] = np.sqrt(portfolio["var_kwh"])
    return portfolio[["datetime_kst", "mu_kwh", "sigma_kwh", "actual_kwh",
                      "total_capacity_kw", "n_sites"]]


def main():
    import sys as _sys
    # CLI arg: baseline / strong / film
    source = _sys.argv[1] if len(_sys.argv) > 1 else "film"
    src_map = {
        "baseline": "ngboost_baseline",
        "strong": "ngboost_strong",
        "film": "film_ngboost",
        "resmlp": "resmlp_adaln",
        "resmlp_adaln": "resmlp_adaln",
    }
    src_dir = src_map.get(source, "resmlp_adaln")
    print(f"[1/2] NGBoost 예측 로딩... (source={src_dir})")
    pred_path = ROOT / "pv" / "experiments" / src_dir / "test_predictions.parquet"
    df = pd.read_parquet(pred_path)
    df["pred_cf"] = df["pred_cf"].clip(lower=0)  # 음수 클립 (이미 됐지만 확인)
    print(f"  rows: {len(df):,}, sites: {df.site.nunique()}")

    print("\n[2/2] 포트폴리오 합산 (독립 가정)...")
    portfolio = aggregate_portfolio(df)
    # Cloud 정보 join (regime-aware reweight 용)
    ts = pd.read_parquet(ROOT / "data" / "processed" / "training_set.parquet")
    ts["date"] = ts["datetime_kst"].dt.date
    ts["hour"] = ts["datetime_kst"].dt.hour
    cloud_h = ts.groupby(["date", "hour"], as_index=False).agg(
        cloud_mean=("dc10Tca", "mean"),
        dsr_mean_obs=("dsr_mean", "mean"),
    )
    portfolio["date"] = portfolio["datetime_kst"].dt.date
    portfolio["hour"] = portfolio["datetime_kst"].dt.hour
    portfolio = portfolio.merge(cloud_h, on=["date", "hour"], how="left")
    portfolio = portfolio.drop(columns=["date", "hour"])
    print(f"  포트폴리오 행: {len(portfolio):,} (cloud join 됨)")
    print(f"  시간 unique: {portfolio.datetime_kst.nunique()}")
    print()
    print("=== 통계 ===")
    print(f"  실측 평균:     {portfolio.actual_kwh.mean()/1000:8.1f} MWh")
    print(f"  예측 평균:     {portfolio.mu_kwh.mean()/1000:8.1f} MWh")
    print(f"  σ 평균:        {portfolio.sigma_kwh.mean()/1000:8.1f} MWh")
    print(f"  σ/μ 비율:      {(portfolio.sigma_kwh/portfolio.mu_kwh.replace(0, np.nan)).mean()*100:.1f}%")
    print(f"  Bias (실측-예측): {(portfolio.actual_kwh - portfolio.mu_kwh).mean()/1000:8.1f} MWh")

    # 80% interval coverage 검증
    portfolio["lower_q10"] = portfolio.mu_kwh - 1.282 * portfolio.sigma_kwh
    portfolio["upper_q90"] = portfolio.mu_kwh + 1.282 * portfolio.sigma_kwh
    cov80 = ((portfolio.actual_kwh >= portfolio.lower_q10) & (portfolio.actual_kwh <= portfolio.upper_q90)).mean()
    print(f"  80% interval coverage: {cov80*100:.1f}%   (목표 80%)")

    # Save
    out_dir = ROOT / "data" / "processed"
    out_path = out_dir / "portfolio_predictions.parquet"
    portfolio.drop(columns=["lower_q10", "upper_q90"]).to_parquet(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
