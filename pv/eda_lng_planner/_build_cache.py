"""LNG ramp EDA 용 hourly cache 생성 (한 번만 실행).

4 년 thermal_hourly_*.csv → data/processed/lng_hourly_ramp.parquet
컬럼: unit, datetime_kst, mw, mw_diff  (운전 중 mw>5 만 포함)
"""
import sys
import glob
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data/processed/lng_hourly_ramp.parquet"
LNG_UNITS = ['CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8','CS1','CS2']

files = sorted(glob.glob(str(ROOT / "data/thermal_hourly/thermal_hourly_*.csv")))
print(f"thermal_hourly files: {len(files)}")

dfs = []
for i, f in enumerate(files, 1):
    try:
        df = pd.read_csv(f, encoding="utf-8", skipinitialspace=True,
                         header=None, skiprows=1, usecols=range(27))
    except Exception as e:
        print(f"  skip {Path(f).name}: {e}")
        continue
    df.columns = ['plant', 'unit', 'date'] + [f'h{h}' for h in range(1, 25)]
    df['unit'] = df['unit'].astype(str).str.strip()
    df = df[df.unit.isin(LNG_UNITS)].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    long = df.melt(id_vars=['unit', 'date'],
                   value_vars=[f'h{h}' for h in range(1, 25)],
                   var_name='hour_str', value_name='mw_kwh')
    long['hour'] = long.hour_str.str.replace('h', '').astype(int) - 1
    long['mw'] = pd.to_numeric(long['mw_kwh'], errors='coerce') / 1000.0
    dfs.append(long[['unit', 'date', 'hour', 'mw']])
    if i % 12 == 0:
        print(f"  {i}/{len(files)} processed")

raw = pd.concat(dfs, ignore_index=True)
raw['datetime_kst'] = raw.date + pd.to_timedelta(raw.hour, unit='h')
raw = raw.sort_values(['unit', 'datetime_kst']).reset_index(drop=True)
raw['mw_diff'] = raw.groupby('unit')['mw'].diff()

# 운전 중 (mw>5) 만 저장 — 분석 대상
running = raw[(raw.mw > 5) & (raw.mw_diff.notna())][
    ['unit', 'datetime_kst', 'mw', 'mw_diff']
].copy()

CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
running.to_parquet(CACHE_PATH, index=False)
print(f"\n[OK] saved cache: {CACHE_PATH.relative_to(ROOT)}")
print(f"     rows: {len(running):,}  ({running.unit.nunique()} units)")
print(f"     range: {running.datetime_kst.min()} ~ {running.datetime_kst.max()}")
print(f"     size: {CACHE_PATH.stat().st_size / 1024:.0f} KB")
