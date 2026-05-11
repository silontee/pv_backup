"""학습용 통합 테이블 생성.

입력:
  - data/solar_hourly/  (KOEN PV 발전량, 호기별 wide format)
  - data/gk2a_v2/       (위성 GHI 시간 집계 + zenith)
  - data/asos_hourly/   (ASOS 기상 관측)

출력:
  - data/processed/training_set.parquet
  - data/processed/training_set.csv (디버깅용)

규칙:
  - data_strategy.md §2.5 확정 11 호기 → 8 사이트 합산
  - 사이트 → 가장 가까운 ASOS 관측소 매핑
  - kt (clearsky index) = dsr_mean / pvlib_clearsky_ghi
  - 시간 라벨: hour-ending (gk2a v2 컨벤션)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message=".*Length of header.*")

ROOT = Path(__file__).resolve().parents[2]

# ========== 설정 ==========

# data_strategy.md §2.5 확정 11 호기 → 8 사이트
KEEP_UNITS = [
    ("경상대태양광",     "1",   905, "경상대"),
    ("고흥만 수상태양광", "1", 63481, "고흥만수상"),
    ("광양항세방태양광",  "1",  2993, "광양항세방"),
    ("구미태양광",       "1",   992, "구미"),
    ("삼천포태양광",     "2",   990, "삼천포"),
    ("삼천포태양광",     "3",   350, "삼천포"),
    ("영흥태양광",       "1",  1000, "영흥"),
    ("영흥태양광",       "2",   993, "영흥"),
    ("영흥태양광#5",     "1",  3500, "영흥"),
    ("예천태양광",       "1",  2000, "예천"),
    ("두산엔진MG태양광",  "1",    77, "창원"),
]
UNIT_KEY = {(u, h): (cap, site) for u, h, cap, site in KEEP_UNITS}

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

# 사이트 → ASOS 가장 가까운 관측소
SITE_ASOS = {
    "경상대":    "진주",
    "고흥만수상": "고흥",
    "광양항세방": "광양시",
    "구미":      "구미",
    "삼천포":    "진주",
    "영흥":      "인천",
    "예천":      "안동",
    "창원":      "창원",
}


# ========== 데이터 로딩 ==========

def load_pv() -> pd.DataFrame:
    """KOEN solar_hourly → 사이트별 시간 합산."""
    files = sorted((ROOT / "data" / "solar_hourly").glob("solar_hourly_*.csv"))
    dfs = []
    for f in files:
        d = pd.read_csv(f, index_col=False)
        d.columns = [c.strip() for c in d.columns]
        d["발전구분"] = d["발전구분"].astype(str).str.strip()
        d["호기"] = d["호기"].astype(str).str.strip()
        d["일자_str"] = d["일자"].astype(str).str.strip()
        d = d[d["일자_str"].str.match(r"^\d{4}-\d{2}-\d{2}")].copy()
        d["일자"] = pd.to_datetime(d["일자_str"])

        hour_cols = [f"{h}시 발전량(KWh)" for h in range(1, 25)]
        long = d.melt(
            id_vars=["발전구분", "호기", "일자"],
            value_vars=hour_cols,
            var_name="hl",
            value_name="gen_kwh",
        )
        long["hour"] = long["hl"].str.extract(r"(\d+)시").astype(int)
        long["datetime_kst"] = long["일자"] + pd.to_timedelta(long["hour"], unit="h")
        long["gen_kwh"] = pd.to_numeric(long["gen_kwh"], errors="coerce")
        long["key"] = list(zip(long["발전구분"], long["호기"]))
        long = long[long["key"].isin(UNIT_KEY.keys())].copy()
        long["site_capacity_kw"] = long["key"].map(lambda k: UNIT_KEY[k][0])
        long["site"] = long["key"].map(lambda k: UNIT_KEY[k][1])
        dfs.append(long[["site", "datetime_kst", "gen_kwh", "site_capacity_kw"]])

    pv = pd.concat(dfs, ignore_index=True)
    site_pv = (
        pv.groupby(["site", "datetime_kst"], as_index=False)
          .agg(gen_kwh=("gen_kwh", "sum"),
               site_capacity_kw=("site_capacity_kw", "sum"))
    )
    site_pv["cf"] = site_pv["gen_kwh"] / site_pv["site_capacity_kw"]
    return site_pv


def load_gk2a() -> pd.DataFrame:
    files = sorted((ROOT / "data" / "gk2a_v2").glob("*.csv"))
    gk = pd.concat(
        [pd.read_csv(f, parse_dates=["datetime_kst"]) for f in files],
        ignore_index=True,
    )
    gk = gk[gk.site.isin(SITE_COORDS.keys())][
        ["datetime_kst", "site", "dsr_mean", "dsr_n_valid", "zenith_center", "lat", "lon"]
    ].copy()
    return gk


def load_asos() -> pd.DataFrame:
    files = sorted((ROOT / "data" / "asos_hourly").glob("asos_hourly_*.csv"))
    asos = pd.concat(
        [pd.read_csv(f, parse_dates=["tm"]) for f in files],
        ignore_index=True,
    )
    asos = asos[asos["stnNm"].isin(SITE_ASOS.values())].copy()
    asos = asos.rename(columns={"tm": "datetime_kst", "stnNm": "asos_stn"})
    return asos[["datetime_kst", "asos_stn", "ta", "hm", "ws", "wd", "rn", "pa", "ps", "dc10Tca"]]


# ========== Feature Engineering ==========

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime_kst"].dt.hour
    df["month"] = df["datetime_kst"].dt.month
    df["doy"] = df["datetime_kst"].dt.dayofyear
    # cyclical encoding (TFT 친화)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 366)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 366)
    return df


# ========== Join ==========

def build() -> pd.DataFrame:
    print("[1/4] PV 로딩...")
    pv = load_pv()
    print(f"  PV 행: {len(pv):,}, 사이트: {pv.site.nunique()}")

    print("[2/4] GK-2A v2 로딩...")
    gk = load_gk2a()
    print(f"  GK-2A 행: {len(gk):,}")

    print("[3/5] ASOS 로딩 + 사이트 매핑 broadcast...")
    asos = load_asos()
    site_asos_pairs = pd.DataFrame(
        [{"site": site, "asos_stn": stn} for site, stn in SITE_ASOS.items()]
    )
    asos_per_site = asos.merge(site_asos_pairs, on="asos_stn")
    print(f"  ASOS 행 (site broadcast 후): {len(asos_per_site):,}")

    print("[4/4] Join PV ⋈ GK-2A ⋈ ASOS ...")
    df = pv.merge(gk, on=["datetime_kst", "site"], how="inner")
    df = df.merge(
        asos_per_site[["datetime_kst", "site", "ta", "hm", "ws", "wd", "rn", "dc10Tca", "pa", "ps"]],
        on=["datetime_kst", "site"],
        how="left",
    )
    df = add_time_features(df)
    print(f"  최종 행: {len(df):,}, 사이트: {df.site.nunique()}")
    return df


def save(df: pd.DataFrame) -> None:
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(exist_ok=True)
    parquet = out_dir / "training_set.parquet"
    csv = out_dir / "training_set.csv"
    df.to_parquet(parquet, index=False)
    df.to_csv(csv, index=False)
    print(f"\n저장: {parquet} ({parquet.stat().st_size / 1024**2:.1f} MB)")
    print(f"저장: {csv}")


def main() -> None:
    df = build()
    save(df)
    print("\n=== 결측률 (학습 candidate) ===")
    for c in ["dsr_mean", "zenith_center", "ta", "hm", "ws", "dc10Tca"]:
        miss = df[c].isna().mean() * 100
        print(f"  {c:<20} {miss:>6.2f}%")
    print(f"\nTarget cf: mean={df['cf'].mean():.3f}, std={df['cf'].std():.3f}")


if __name__ == "__main__":
    main()
