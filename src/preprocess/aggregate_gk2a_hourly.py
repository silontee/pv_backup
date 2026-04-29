"""GK-2A v1 (10분) → v2 (시간) 집계.

규칙:
- 입력: data/gk2a_v1/YYYYMM.csv (10분 단위)
- 사용 컬럼: datetime_kst, site, dsr (dsr_dqf는 dsr.notna()와 100% 동치라 제거)
- 시간 라벨: hour-ending KOEN 컨벤션 (라벨 N = [N-1:00, N:00) 데이터)
- DSR 집계: mean of non-NaN 10분 슬롯 (NaN은 평균 제외, n_valid로 추적)
- zenith: pvlib로 시간 중심 (라벨 - 30min) 기준 계산
- 출력: data/gk2a_v2/YYYYMM.csv
"""
import sys
from pathlib import Path
import pandas as pd
import pvlib

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "data" / "gk2a_v1"
DST_DIR = ROOT / "data" / "gk2a_v2"
DST_DIR.mkdir(parents=True, exist_ok=True)

# 사이트 좌표 (data_strategy.md §1 기반, solar_sites.csv 좌표 비어있어서 직접)
SITE_COORDS = {
    '경상대':    (35.18, 128.10),
    '고흥만수상': (34.57, 127.30),
    '광양항세방': (34.93, 127.71),
    '구미':      (36.13, 128.34),
    '삼천포':    (34.95, 128.07),
    '여수':      (34.74, 127.74),
    '영동':      (37.18, 128.46),
    '영흥':      (37.26, 126.46),
    '예천':      (36.65, 128.46),
    '창원':      (35.21, 128.58),
    '탑선':      (35.24, 126.81),
}


def load_all() -> pd.DataFrame:
    files = sorted(SRC_DIR.glob("*.csv"))
    dfs = []
    for f in files:
        d = pd.read_csv(f, parse_dates=["datetime_kst"], usecols=["datetime_kst", "site", "dsr"])
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    # hour-ending 라벨: 10분 슬롯 floor('1H') + 1h
    # 예: 12:00, 12:10, ..., 12:50 → 모두 라벨 13:00
    df = df.copy()
    df["hour_label"] = df["datetime_kst"].dt.floor("1h") + pd.Timedelta(hours=1)

    hourly = (
        df.groupby(["site", "hour_label"], sort=True)
          .agg(
              dsr_mean=("dsr", lambda s: s.mean(skipna=True)),
              dsr_n_valid=("dsr", lambda s: s.notna().sum()),
              n_slots=("dsr", "size"),
          )
          .reset_index()
          .rename(columns={"hour_label": "datetime_kst"})
    )
    return hourly


def add_zenith(hourly: pd.DataFrame) -> pd.DataFrame:
    # 시간 중심 = 라벨 - 30min (KST)
    hourly["hour_center_kst"] = hourly["datetime_kst"] - pd.Timedelta(minutes=30)

    out = []
    for site, group in hourly.groupby("site", sort=False):
        lat, lon = SITE_COORDS[site]
        # KST naive → UTC for pvlib
        center_utc = pd.DatetimeIndex(group["hour_center_kst"]).tz_localize("Asia/Seoul").tz_convert("UTC")
        pos = pvlib.solarposition.get_solarposition(center_utc, lat, lon)
        g = group.copy()
        g["zenith_center"] = pos.zenith.values
        g["lat"] = lat
        g["lon"] = lon
        out.append(g)
    return pd.concat(out, ignore_index=True).drop(columns=["hour_center_kst"])


def save_by_month(hourly: pd.DataFrame) -> None:
    hourly = hourly.copy()
    hourly["ym"] = hourly["datetime_kst"].dt.strftime("%Y%m")
    cols = ["datetime_kst", "site", "dsr_mean", "dsr_n_valid", "n_slots", "zenith_center", "lat", "lon"]
    for ym, group in hourly.groupby("ym", sort=True):
        out_path = DST_DIR / f"{ym}.csv"
        group[cols].sort_values(["datetime_kst", "site"]).to_csv(out_path, index=False)
        print(f"  saved {out_path.name}: {len(group):,} rows")


def main():
    print(f"[1/4] Loading {len(list(SRC_DIR.glob('*.csv')))} files...")
    df = load_all()
    print(f"  loaded: {len(df):,} rows, sites={df.site.nunique()}, range={df.datetime_kst.min()} ~ {df.datetime_kst.max()}")

    print("[2/4] Aggregating hourly...")
    hourly = aggregate_hourly(df)
    print(f"  hourly rows: {len(hourly):,}, hours per site: {len(hourly) // hourly.site.nunique():,}")

    print("[3/4] Computing zenith via pvlib...")
    hourly = add_zenith(hourly)
    print(f"  zenith range: [{hourly.zenith_center.min():.2f}°, {hourly.zenith_center.max():.2f}°]")

    print("[4/4] Saving by month...")
    save_by_month(hourly)
    print(f"  done. dst={DST_DIR}")


if __name__ == "__main__":
    main()
