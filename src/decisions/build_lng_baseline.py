"""LNG baseline + Avail proxy 빌더 (단순화 버전).

per plan/active/lng/plan.md §4.1.1, §5.3.1:
  P_DA,u(t) = 2025년 시점 t의 호기 u 실측 LNG 발전량  (학습 X)
  Avail_u(t):
    - is_online(u,t) = P_DA,u(t) > P_min,u × 0.5
    - Avail = 1.0           if is_online
    - Avail = rolling 7d running rate at hour(t)   otherwise

출력:
  data/processed/lng_baseline_test.parquet
    columns: unit, datetime_kst, date, hour, month, mw, P_DA_mw, P_min, P_max,
             is_online, Avail
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/thermal_hourly"
PROC_DIR = ROOT / "data/processed"
PROFILE_PATH = PROC_DIR / "thermal_unit_profile.csv"

LNG_UNITS = ['CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8','CS1','CS2']
TEST_YEAR = 2025


def load_thermal_hourly_year(year):
    """해당 연도 hourly LNG 발전량 로드 (long format, MW)."""
    dfs = []
    for csv in sorted(RAW_DIR.glob(f"thermal_hourly_{year}*.csv")):
        try:
            df = pd.read_csv(csv, encoding='utf-8', skipinitialspace=True,
                             header=None, skiprows=1, usecols=range(27))
        except Exception as e:
            print(f"  skip {csv.name}: {e}")
            continue
        cols = ['plant','unit','date'] + [f'h{h}' for h in range(1, 25)]
        df.columns = cols
        df['unit'] = df['unit'].astype(str).str.strip()
        df = df[df.unit.isin(LNG_UNITS)].copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        long = df.melt(id_vars=['unit','date'], value_vars=[f'h{h}' for h in range(1,25)],
                       var_name='hour_str', value_name='mw_kwh')
        long['hour'] = long.hour_str.str.replace('h','').astype(int) - 1
        long['mw'] = pd.to_numeric(long['mw_kwh'], errors='coerce') / 1000.0
        dfs.append(long[['unit','date','hour','mw']])
    return pd.concat(dfs, ignore_index=True)


def build_baseline_2025():
    print("=" * 60)
    print("Building LNG baseline proxy (per plan §4.1.1, §5.3.1)")
    print("  → P_DA = 2025 actual (no training step)")
    print("=" * 60)

    print(f"\n[1] Load thermal_hourly {TEST_YEAR}")
    df = load_thermal_hourly_year(TEST_YEAR)
    print(f"  rows: {len(df):,}")
    print(f"  units: {sorted(df.unit.unique())}")
    print(f"  date range: {df.date.min()} ~ {df.date.max()}")

    print("\n[2] Load profile (P_min, P_max)")
    profile = pd.read_csv(PROFILE_PATH, encoding='utf-8')
    profile.columns = profile.columns.str.replace('﻿', '').str.strip()
    pmin_map = profile.set_index('호기')['pmin_p05_run'].to_dict()
    pmax_map = profile.set_index('호기')['pmax_p95'].to_dict()

    print("\n[3] Compute P_DA, is_online, Avail")
    df['datetime_kst'] = df.date + pd.to_timedelta(df.hour, unit='h')
    df = df.sort_values(['unit','datetime_kst']).reset_index(drop=True)
    df['P_DA_mw'] = df.mw                          # 핵심: 실측 = baseline
    df['P_min']  = df.unit.map(pmin_map)
    df['P_max']  = df.unit.map(pmax_map)
    df['threshold'] = df.P_min * 0.5
    df['is_online']  = (df.P_DA_mw > df.threshold).astype(int)
    df['is_running'] = df.is_online                # 동일 정의 (실측 기준)
    df['month'] = df.date.dt.month

    # Avail: online → 1.0, offline → rolling 7-day same-hour running rate (직전 7일)
    avail_parts = []
    for (unit, h), g in df.groupby(['unit','hour']):
        g = g.sort_values('datetime_kst').reset_index(drop=True)
        g['avail_rolling'] = g.is_running.shift(1).rolling(7, min_periods=1).mean()
        g['avail_rolling'] = g.avail_rolling.fillna(0.0)
        avail_parts.append(g[['unit','datetime_kst','avail_rolling']])
    avail_df = pd.concat(avail_parts, ignore_index=True)
    df = df.merge(avail_df, on=['unit','datetime_kst'], how='left')

    df['Avail'] = np.where(df.is_online == 1, 1.0, df.avail_rolling)

    out = df[['unit','datetime_kst','date','hour','month','mw',
               'P_DA_mw','P_min','P_max','is_online','Avail']]

    print(f"\n  Online ratio (daytime 9~17 only):")
    day = out[out.hour.between(9, 17)]
    summary = day.groupby('unit').agg(
        online_pct=('is_online', lambda s: s.mean()*100),
        avail_avg=('Avail','mean'),
        P_DA_avg=('P_DA_mw','mean'),
        P_DA_max=('P_DA_mw','max'),
    ).round(2)
    print(summary)

    # save
    out_path = PROC_DIR / "lng_baseline_test.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n저장: {out_path}  ({len(out):,} rows)")
    return out


if __name__ == "__main__":
    build_baseline_2025()
