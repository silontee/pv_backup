"""내일 PV 예측 — Open-Meteo 예보 + NGBoost 추론 + 시나리오.

흐름:
  1. Open-Meteo `forecast` API로 내일 24h 예보 받음 (8 사이트)
  2. zenith 계산 (pvlib)
  3. feature schema 맞춰 변환
  4. NGBoost 추론 → 사이트별 분포 (μ, σ)
  5. 포트폴리오 합산 (독립 가정)
  6. Quantile 5 + 시나리오 저장

⚠️ Caveat: MOS bias 보정 미구현. Open-Meteo 예보를 그대로 input.
   진짜 production은 Stage 0 MOS로 분포 보정 후 추론.

출력:
  data/processed/tomorrow_forecast.parquet  (포트폴리오 + 사이트별)
  data/processed/tomorrow_meta.json         (메타: 받은 시각, 대상 날짜)
"""
import sys
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pvlib
import requests
from scipy.stats import norm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# 사이트 좌표 (build_training_set.py와 동일)
SITE_COORDS = {
    "경상대":    (35.18, 128.10),
    "고흥만수상": (34.57, 127.30),
    "광양항세방": (34.93, 127.71),
    "구미":      (36.13, 128.34),
    "삼천포":    (34.95, 128.07),
    "영흥":      (37.26, 126.46),
    "예천":      (36.65, 128.46),
    "창원":      (35.21, 128.58),
}

# 사이트 capacity (data_strategy.md §2.5 합산)
SITE_CAPACITY_KW = {
    "경상대":    905,
    "고흥만수상": 63481,
    "광양항세방": 2993,
    "구미":      992,
    "삼천포":    1340,
    "영흥":      5493,
    "예천":      2000,
    "창원":      77,
}

QUANTILES = [0.05, 0.25, 0.50, 0.75, 0.95]
QUANTILE_LABELS = ["worst", "pessimistic", "expected", "optimistic", "best"]


def fetch_open_meteo(date_str: str) -> pd.DataFrame:
    """Open-Meteo forecast API 호출 → 8 사이트 24h 예보.

    Args:
        date_str: "YYYY-MM-DD" 형식

    Returns:
        long format: site, datetime_kst, dsr_mean, ta, hm, ws, dc10Tca
    """
    print(f"[1/5] Open-Meteo 예보 받기 ({date_str})...")
    rows = []
    for site, (lat, lon) in SITE_COORDS.items():
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            "&hourly=shortwave_radiation,temperature_2m,relative_humidity_2m,"
            "wind_speed_10m,cloud_cover"
            "&timezone=Asia%2FSeoul"
            "&wind_speed_unit=ms"
        )
        r = requests.get(url, timeout=30).json()
        if "hourly" not in r:
            print(f"  ⚠️ {site}: API 오류 — {r}")
            continue
        hr = r["hourly"]
        site_df = pd.DataFrame({
            "site": site,
            "datetime_kst_str": hr["time"],
            "dsr_mean": hr["shortwave_radiation"],
            "ta": hr["temperature_2m"],
            "hm": hr["relative_humidity_2m"],
            "ws": hr["wind_speed_10m"],
            "dc10Tca": hr["cloud_cover"],
        })
        rows.append(site_df)

    if not rows:
        raise RuntimeError("Open-Meteo 응답이 모두 비어있음")
    df = pd.concat(rows, ignore_index=True)

    # datetime 변환: Open-Meteo는 "2026-05-04T00:00" 같은 형식
    # 라벨 컨벤션 — Open-Meteo time = hour-beginning (00:00 = 0~1시)
    # 우리 학습은 hour-ending (1시 라벨 = [00, 01) data)
    # → Open-Meteo "2026-05-04 00:00" 데이터 → 우리 라벨 1시 (= 2026-05-04 01:00)
    df["datetime_kst"] = pd.to_datetime(df["datetime_kst_str"]) + pd.Timedelta(hours=1)
    df = df.drop(columns=["datetime_kst_str"])
    print(f"  받은 행: {len(df):,} (사이트 {df.site.nunique()} × 시간 {df.datetime_kst.nunique()})")
    return df


def add_zenith(df: pd.DataFrame) -> pd.DataFrame:
    """pvlib로 시간 중심 zenith 계산 (라벨 - 30분 기준)."""
    print("[2/5] zenith 계산 (pvlib)...")
    out = []
    for site, group in df.groupby("site", sort=False):
        lat, lon = SITE_COORDS[site]
        time_center = pd.DatetimeIndex(group["datetime_kst"]) - pd.Timedelta(minutes=30)
        time_center = time_center.tz_localize("Asia/Seoul").tz_convert("UTC")
        pos = pvlib.solarposition.get_solarposition(time_center, lat, lon)
        g = group.copy()
        g["zenith_center"] = pos.zenith.values
        g["lat"] = lat
        g["lon"] = lon
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """학습 schema 맞춰 feature 행렬 만듦."""
    print("[3/5] Feature 변환...")
    df = df.copy()
    df["site_capacity_kw"] = df["site"].map(SITE_CAPACITY_KW)
    df["hour"] = df["datetime_kst"].dt.hour
    df["month"] = df["datetime_kst"].dt.month

    # site one-hot
    site_cols = schema["site_one_hot_cols"]
    for col in site_cols:
        site_name = col.replace("site_", "")
        df[col] = (df["site"] == site_name).astype(int)

    return df


def predict_with_model(df: pd.DataFrame, model, schema: dict) -> pd.DataFrame:
    """NGBoost 추론.

    야간 후처리: zenith >= 90° 또는 forecast GHI=0 → 발전 deterministic 0.
    NGBoost는 daytime only로 학습돼서 야간엔 noise prediction.
    """
    print("[4/5] NGBoost 추론...")
    feat_cols = schema["feature_order"]
    X = df[feat_cols].values.astype(float)
    pred_dist = model.pred_dist(X)
    out = df.copy()
    out["pred_cf"] = np.clip(np.asarray(pred_dist.loc), 0, None)
    out["pred_std_cf"] = np.asarray(pred_dist.scale)

    # 야간 강제 0 (zenith 90° 이상 OR forecast GHI=0)
    night_mask = (out["zenith_center"] >= 90) | (out["dsr_mean"] <= 1)
    out.loc[night_mask, "pred_cf"] = 0.0
    out.loc[night_mask, "pred_std_cf"] = 0.0

    out["mu_kwh"] = out["pred_cf"] * out["site_capacity_kw"]
    out["sigma_kwh"] = out["pred_std_cf"] * out["site_capacity_kw"]
    n_night = night_mask.sum()
    print(f"  야간 강제 0: {n_night} 행 (zenith>=90° 또는 GHI<=1)")
    return out


def aggregate_and_quantiles(site_pred: pd.DataFrame) -> tuple:
    """포트폴리오 합산 + quantile 추출."""
    print("[5/5] 포트폴리오 합산 + quantile 추출...")
    portfolio = (
        site_pred.groupby("datetime_kst", as_index=False)
                 .agg(mu_kwh=("mu_kwh", "sum"),
                      var_kwh=("sigma_kwh", lambda s: (s ** 2).sum()),
                      total_capacity_kw=("site_capacity_kw", "sum"))
    )
    portfolio["sigma_kwh"] = np.sqrt(portfolio["var_kwh"])
    portfolio = portfolio.drop(columns=["var_kwh"])

    z_scores = norm.ppf(QUANTILES)
    for q, z, label in zip(QUANTILES, z_scores, QUANTILE_LABELS):
        portfolio[f"q{int(q*100):02d}_{label}"] = (portfolio["mu_kwh"] + z * portfolio["sigma_kwh"]).clip(lower=0)

    # 사이트별 quantile도
    for q, z, label in zip(QUANTILES, z_scores, QUANTILE_LABELS):
        site_pred[f"q{int(q*100):02d}_{label}"] = (site_pred["mu_kwh"] + z * site_pred["sigma_kwh"]).clip(lower=0)

    return portfolio, site_pred


def main(target_date: str = None):
    if target_date is None:
        # 기본: 내일 (KST 기준)
        tomorrow = datetime.now() + timedelta(days=1)
        target_date = tomorrow.strftime("%Y-%m-%d")
    print(f"=== 내일 PV 예측 — target = {target_date} ===\n")

    # 1. Open-Meteo
    raw = fetch_open_meteo(target_date)

    # 2. zenith
    raw = add_zenith(raw)

    # 3. Feature schema 맞춤
    schema = joblib.load(ROOT / "pv" / "experiments" / "ngboost_baseline" / "feature_schema.joblib")
    df_features = build_features(raw, schema)

    # 4. NGBoost predict
    model = joblib.load(ROOT / "pv" / "experiments" / "ngboost_baseline" / "model.joblib")
    site_pred = predict_with_model(df_features, model, schema)

    # 5. Aggregate + quantile
    portfolio, site_pred = aggregate_and_quantiles(site_pred)

    # 저장
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(exist_ok=True)
    portfolio.to_parquet(out_dir / "tomorrow_forecast_portfolio.parquet", index=False)
    site_pred.to_parquet(out_dir / "tomorrow_forecast_sites.parquet", index=False)

    meta = {
        "target_date": target_date,
        "fetched_at_kst": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_sites": len(SITE_COORDS),
        "n_hours": len(portfolio),
        "total_capacity_kw": int(portfolio["total_capacity_kw"].iloc[0]) if len(portfolio) else 0,
    }
    with open(out_dir / "tomorrow_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n=== 결과 ===")
    print(f"  포트폴리오 행: {len(portfolio)}, 사이트별 행: {len(site_pred)}")
    print(f"  저장: {out_dir / 'tomorrow_forecast_portfolio.parquet'}")
    print(f"  저장: {out_dir / 'tomorrow_forecast_sites.parquet'}")
    print(f"  저장: {out_dir / 'tomorrow_meta.json'}")
    print()
    print("=== 시간별 예측 (포트폴리오) ===")
    show = portfolio.copy()
    show["mu_MWh"] = (show["mu_kwh"] / 1000).round(2)
    show["sigma_MWh"] = (show["sigma_kwh"] / 1000).round(2)
    show["q05"] = (show["q05_worst"] / 1000).round(2)
    show["q50"] = (show["q50_expected"] / 1000).round(2)
    show["q95"] = (show["q95_best"] / 1000).round(2)
    show["hour"] = pd.to_datetime(show["datetime_kst"]).dt.hour
    print(show[["hour", "mu_MWh", "sigma_MWh", "q05", "q50", "q95"]].to_string(index=False))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    main(target)
