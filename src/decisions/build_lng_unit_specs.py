"""LNG 호기별 운영 spec v2 — data-driven (4년치 hourly 직접 계산).

기존 thermal_unit_profile.csv 와 차이:
- ramp_up_p90 / ramp_dn_p90 — 호기별 hourly 변화 분포 p90 (운전 중 only)
- mode_data — 운전 패턴 기반 분류 (baseload / mid-merit / peaker)
- backup_score — headroom × ramp × running_pct 종합

출력: data/processed/lng_unit_specs.parquet
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings, glob
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
LNG_UNITS = ['CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8','CS1','CS2']


def load_hourly():
    files = sorted(glob.glob(str(ROOT / 'data/thermal_hourly/thermal_hourly_*.csv')))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8', skipinitialspace=True,
                             header=None, skiprows=1, usecols=range(27))
        except: continue
        df.columns = ['plant','unit','date'] + [f'h{h}' for h in range(1, 25)]
        df['unit'] = df['unit'].astype(str).str.strip()
        df = df[df.unit.isin(LNG_UNITS)].copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        long = df.melt(id_vars=['unit','date'], value_vars=[f'h{h}' for h in range(1,25)],
                       var_name='hour_str', value_name='mw_kwh')
        long['hour'] = long.hour_str.str.replace('h','').astype(int) - 1
        long['mw'] = pd.to_numeric(long['mw_kwh'], errors='coerce') / 1000.0
        dfs.append(long[['unit','date','hour','mw']])
    out = pd.concat(dfs, ignore_index=True)
    out['datetime_kst'] = out.date + pd.to_timedelta(out.hour, unit='h')
    return out.sort_values(['unit','datetime_kst']).reset_index(drop=True)


def main():
    print("Loading 4-year hourly LNG data...")
    df = load_hourly()
    print(f"  rows: {len(df):,}")

    # hourly diff (운전 중 only)
    df['mw_diff'] = df.groupby('unit')['mw'].diff()
    running = df[(df.mw > 5) & (df.mw_diff.notna())].copy()

    # P_min, P_max (운전 중 통계)
    rows = []
    for u in LNG_UNITS:
        s = df[df.unit == u]
        run = s[s.mw > 5]
        diff_run = running[running.unit == u]
        pos = diff_run[diff_run.mw_diff > 0].mw_diff.values
        neg = -diff_run[diff_run.mw_diff < 0].mw_diff.values

        running_pct = (s.mw > 5).mean() * 100
        avg_run = run.mw.mean() if len(run) else 0
        median_run = run.mw.median() if len(run) else 0

        # mode classification (data-driven)
        if running_pct >= 45:
            mode = 'baseload'
        elif running_pct >= 28:
            mode = 'mid-merit'
        else:
            mode = 'peaker'

        rows.append({
            'unit': u,
            'P_min': float(np.percentile(run.mw, 5)) if len(run) else 0.0,   # p05 (운전 중 최소)
            'P_max': float(np.percentile(run.mw, 95)) if len(run) else 0.0,  # p95 (보수)
            'P_max_obs': float(run.mw.max()) if len(run) else 0.0,            # 관측 최대
            'avg_run': float(avg_run),
            'median_run': float(median_run),
            'running_pct': float(running_pct),
            'ramp_up_p50': float(np.percentile(pos, 50)) if len(pos) else 0.0,
            'ramp_up_p90': float(np.percentile(pos, 90)) if len(pos) else 0.0,
            'ramp_up_p95': float(np.percentile(pos, 95)) if len(pos) else 0.0,
            'ramp_dn_p50': float(np.percentile(neg, 50)) if len(neg) else 0.0,
            'ramp_dn_p90': float(np.percentile(neg, 90)) if len(neg) else 0.0,
            'ramp_dn_p95': float(np.percentile(neg, 95)) if len(neg) else 0.0,
            'mode': mode,
        })

    out = pd.DataFrame(rows)
    out['headroom_p90'] = out.P_max - out.median_run
    # 기본 priority: backup_score 정규화 (headroom × ramp_up_p90 × running_pct/100)
    out['backup_score'] = (out.headroom_p90 * out.ramp_up_p90 * out.running_pct / 100).round(0)
    out['priority_data'] = (out.backup_score / out.backup_score.max()).round(3)

    print()
    print(out.round(2).to_string(index=False))

    out_path = ROOT / "data/processed/lng_unit_specs.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
