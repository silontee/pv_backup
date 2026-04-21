"""
태양광 발전 데이터 전처리
- wide → long 변환 (시간 컬럼 → 행)
- 사용 가능 호기만 유지 (plan/main/data_strategy.md §2 화이트리스트)
- 같은 사이트 내 사용 가능 호기만 합산
- 출력: data/processed/solar_hourly_long.csv
"""

import io
import glob
import os
import sys

import pandas as pd


RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "solar_hourly")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

# 사용 가능 호기 화이트리스트 (site, unit) — SSoT: plan/main/data_strategy.md §2
# 2024 실데이터 피크 시각 ≤ 15시 조건 통과한 호기만.
USABLE_UNITS = {
    ("경상대태양광", "1"),
    ("고흥만 수상태양광", "1"),
    ("광양항세방태양광", "1"),
    ("구미태양광", "1"),
    ("두산엔진MG태양광", "1"),
    ("삼천포태양광", "1"),
    ("삼천포태양광", "2"),
    ("삼천포태양광", "3"),
    ("영흥태양광", "1"),
    ("영흥태양광", "2"),
    ("영흥태양광#5", "1"),
    ("예천태양광", "1"),
}


def load_solar_csv(fp: str) -> pd.DataFrame:
    """trailing comma 문제 해결하면서 CSV 로드"""
    with open(fp, encoding="utf-8") as f:
        header = f.readline().strip().rstrip(",")
        lines = [l.strip().rstrip(",") for l in f.readlines() if l.strip()]
    col_names = [c.strip() for c in header.split(",")]
    df = pd.read_csv(
        io.StringIO("\n".join([",".join(col_names)] + lines)),
        skipinitialspace=True,
    )
    df.columns = df.columns.str.strip()
    return df


def process_one_file(fp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """한 월 CSV → (시간별 long, 일별 통계) 반환"""
    raw = load_solar_csv(fp)

    # 사용 가능 호기만 (site, unit) 화이트리스트
    raw["발전구분"] = raw["발전구분"].astype(str).str.strip()
    raw["호기"] = raw["호기"].astype(str).str.strip()
    key = list(zip(raw["발전구분"], raw["호기"]))
    raw = raw[[k in USABLE_UNITS for k in key]].copy()

    # 시간 컬럼 식별
    hour_map = {}
    for c in raw.columns:
        if "시 발전량" in c:
            h = int(c.split("시")[0].strip())
            hour_map[c] = h
    hour_cols = list(hour_map.keys())

    # --- 시간별 long (호기 단위 유지) ---
    id_cols = ["발전구분", "호기", "일자"]
    df_long = raw[id_cols + hour_cols].copy()

    # melt (호기 합산 없이)
    df_long = df_long.melt(
        id_vars=id_cols, value_vars=hour_cols, var_name="hour_col", value_name="gen_kwh"
    )
    df_long["hour"] = df_long["hour_col"].map(hour_map)
    df_long.drop(columns="hour_col", inplace=True)

    # 컬럼 정리
    df_long.rename(columns={"발전구분": "site", "호기": "unit", "일자": "date"}, inplace=True)
    df_long["date"] = pd.to_datetime(df_long["date"].astype(str).str.strip())
    df_long["gen_kwh"] = pd.to_numeric(df_long["gen_kwh"], errors="coerce").fillna(0)
    df_long["datetime"] = df_long["date"] + pd.to_timedelta(df_long["hour"], unit="h")
    df_long["site_unit"] = df_long["site"] + " #" + df_long["unit"].astype(str)
    df_long = df_long[["site", "unit", "site_unit", "date", "hour", "datetime", "gen_kwh"]].sort_values(
        ["site", "unit", "datetime"]
    )

    # --- 일별 통계 (호기 단위 유지) ---
    stat_map = {
        "총량(KW)": "total_kwh",
        "평균(KW)": "avg_kw",
        "최대(시간별)": "max_hour_kw",
        "최소(시간별)": "min_hour_kw",
    }
    stat_cols_exist = [c for c in stat_map if c in raw.columns]
    for c in stat_cols_exist:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    df_stats = raw[["발전구분", "호기", "일자"] + stat_cols_exist].copy()
    df_stats.rename(
        columns={"발전구분": "site", "호기": "unit", "일자": "date", **stat_map}, inplace=True
    )
    df_stats["date"] = pd.to_datetime(df_stats["date"].astype(str).str.strip())

    return df_long, df_stats


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(RAW_DIR, "solar_hourly_*.csv")))
    print(f"원본 파일: {len(files)}개")

    all_long = []
    all_stats = []

    for fp in files:
        fname = os.path.basename(fp)
        df_long, df_stats = process_one_file(fp)
        all_long.append(df_long)
        all_stats.append(df_stats)
        print(f"  {fname}: {len(df_long)}행 (long), {len(df_stats)}행 (stats)")

    # 합치기
    df_long_all = pd.concat(all_long, ignore_index=True).sort_values(
        ["site", "unit", "datetime"]
    )
    df_stats_all = pd.concat(all_stats, ignore_index=True).sort_values(
        ["site", "unit", "date"]
    )

    # 저장
    long_path = os.path.join(OUT_DIR, "solar_hourly_long.csv")
    stats_path = os.path.join(OUT_DIR, "solar_daily_stats.csv")

    df_long_all.to_csv(long_path, index=False, encoding="utf-8")
    df_stats_all.to_csv(stats_path, index=False, encoding="utf-8")

    print(f"\n저장 완료:")
    print(f"  시간별: {long_path} ({len(df_long_all):,}행)")
    print(f"  일별통계: {stats_path} ({len(df_stats_all):,}행)")
    print(f"\n호기 수: {df_long_all[['site','unit']].drop_duplicates().shape[0]}개 ({df_long_all['site'].nunique()} 사이트)")
    print(f"기간: {df_long_all['date'].min().date()} ~ {df_long_all['date'].max().date()}")
    print(f"\n호기 목록:")
    for (site, unit), grp in df_long_all.groupby(["site", "unit"]):
        total = grp["gen_kwh"].sum() / 1000
        print(f"  {site:<18} #{unit:<3} 총 {total:>10,.0f} MWh")


if __name__ == "__main__":
    main()
