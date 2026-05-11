"""Open-Meteo historical-forecast-api로 *D-1 issued forecast* archive 수집.

목적:
  *strict D-1 baseline 학습/평가*에 필요한 진짜 forecast 데이터.

방법:
  Open-Meteo `historical-forecast-api`는 *과거 issue된 forecast*를 제공.
    URL: https://historical-forecast-api.open-meteo.com/v1/forecast

  쿼리당 site별 1년치 일괄 fetch 가능 (일별 24h × 365일 = 8760 pts).
  D-1 17:00 KST issue 가정으로 *각 target hour의 lead time*은 7~30h 범위.

변수 (Open-Meteo) → 우리 schema:
  shortwave_radiation       → dsr_fcst    (W/m²)
  temperature_2m            → ta_fcst     (°C)
  relative_humidity_2m      → hm_fcst     (%)
  wind_speed_10m            → ws_fcst     (m/s)
  cloud_cover               → cloud_fcst  (% — *우리는 dc10Tca 0-10이라 /10 변환 필요*)

출력:
  data/d1_forecast/{site}_{year}.parquet
    columns: site, datetime_kst, dsr_fcst, ta_fcst, hm_fcst, ws_fcst, cloud_fcst

⚠️ 주의:
  - Open-Meteo free tier rate limit (10,000 calls/day).
  - 8 사이트 × 4년 = 32 calls (date range 1년씩) → 1일 안에 충분.
  - 하지만 historical-forecast-api는 "보장된 issue time"이 *모델별 cycle*에 의존:
      * Best-match 모델 (default): ECMWF/GFS 등 ensemble.
      * 모델 cycle: 00, 06, 12, 18 UTC. KST 기준 09, 15, 21, 03.
      * D-1 17:00 KST 가까운 cycle = 06 UTC = 15:00 KST D-1 (2시간 전).
      * 즉 lead time = (target h) + (24 - 15) = (target h) + 9 hour.
  - 정확한 lead time 매핑은 별도 검증 필요 (이 스크립트는 일단 하루 단위 fetch).
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

# build_training_set.py와 동일
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

API_BASE = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY_VARS = (
    "shortwave_radiation,temperature_2m,relative_humidity_2m,"
    "wind_speed_10m,cloud_cover"
)

# 변수 매핑
RENAME = {
    "shortwave_radiation": "dsr_fcst",
    "temperature_2m": "ta_fcst",
    "relative_humidity_2m": "hm_fcst",
    "wind_speed_10m": "ws_fcst",
    "cloud_cover": "cloud_fcst",
}


def fetch_year(site: str, lat: float, lon: float, year: int,
               retries: int = 3, sleep_s: float = 1.0) -> pd.DataFrame:
    """1년치 hourly forecast archive 받기."""
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": HOURLY_VARS,
        "timezone": "Asia/Seoul",
        "wind_speed_unit": "ms",
    }
    for attempt in range(retries):
        try:
            r = requests.get(API_BASE, params=params, timeout=60)
            if r.status_code != 200:
                print(f"  ⚠️ {site} {year}: status {r.status_code} — {r.text[:200]}")
                time.sleep(sleep_s * (attempt + 1))
                continue
            data = r.json()
            if "hourly" not in data:
                print(f"  ⚠️ {site} {year}: no hourly key — {data}")
                return pd.DataFrame()
            hourly = data["hourly"]
            df = pd.DataFrame({
                "datetime_kst": pd.to_datetime(hourly["time"]),
                **{RENAME[v]: hourly[v] for v in RENAME if v in hourly},
            })
            df["site"] = site
            # cloud_cover 0-100% → 0-10 scale (dc10Tca 호환)
            if "cloud_fcst" in df.columns:
                df["cloud_fcst"] = df["cloud_fcst"] / 10.0
            return df[["site", "datetime_kst", "dsr_fcst", "ta_fcst", "hm_fcst", "ws_fcst", "cloud_fcst"]]
        except Exception as e:
            print(f"  ⚠️ {site} {year} attempt {attempt}: {e}")
            time.sleep(sleep_s * (attempt + 1))
    return pd.DataFrame()


def main():
    out_dir = ROOT / "data/d1_forecast"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Fetch] Open-Meteo historical-forecast → {out_dir}")
    print(f"  Sites: {len(SITE_COORDS)}, Years: 2022-2025 (4년)")
    print(f"  Total queries: {len(SITE_COORDS) * 4} = ~{len(SITE_COORDS)*4} API calls")

    all_dfs = []
    for site, (lat, lon) in SITE_COORDS.items():
        for year in [2022, 2023, 2024, 2025]:
            out_path = out_dir / f"{site}_{year}.parquet"
            if out_path.exists():
                print(f"  [skip] {out_path.name} (exists)")
                df = pd.read_parquet(out_path)
            else:
                print(f"  [fetch] {site} {year}...", flush=True)
                df = fetch_year(site, lat, lon, year)
                if len(df) == 0:
                    print(f"    ! empty, skipping")
                    continue
                df.to_parquet(out_path, index=False)
                print(f"    saved {len(df):,} rows to {out_path.name}")
                time.sleep(0.5)
            all_dfs.append(df)

    print("\n[Concat] 전체 합산...")
    full = pd.concat(all_dfs, ignore_index=True)
    full.to_parquet(out_dir / "all_sites_all_years.parquet", index=False)
    print(f"  total rows: {len(full):,}")
    print(f"  date range: {full['datetime_kst'].min()} ~ {full['datetime_kst'].max()}")
    print(f"  saved: {out_dir / 'all_sites_all_years.parquet'}")


if __name__ == "__main__":
    main()
