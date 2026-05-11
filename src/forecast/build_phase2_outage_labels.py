"""Phase 2 학습용 outage pseudo-label 생성.

per outage_v3.5 relaxed:
  cf<0.03 AND mu_p1>0.20 AND dc10Tca<7 AND z<-3 AND neighbor (t±1)

대상: Phase 2 train period (val 2024) + test (2025).
출력: data/processed/phase2_outage_labels.parquet
  columns: datetime_kst, site, is_outage
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
PHASE1_DIR = ROOT / "pv/experiments/resmlp_adaln_v2_ensemble"
SEEDS = [42, 123, 7, 202, 999]


def load_phase1_pred(split):
    """Load Phase 1 ensemble predictions."""
    fname = f"{split}_predictions.parquet"
    dfs = []
    for s in SEEDS:
        d = pd.read_parquet(PHASE1_DIR / f"seed_{s}" / fname); d['seed'] = s
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    g = all_df.groupby(['datetime_kst','site'])
    ens = g.agg(
        cf=('cf','first'),
        mu=('pred_mu','mean'),
        mu_var=('pred_mu','var'),
        sig_sq=('pred_sigma', lambda s:(s**2).mean()),
    ).reset_index()
    ens['sigma'] = np.sqrt(ens.sig_sq + ens.mu_var)
    ens['datetime_kst'] = pd.to_datetime(ens.datetime_kst)
    return ens


def detect_outage(p1_df, weather_df):
    df = p1_df.merge(weather_df[['datetime_kst','site','dc10Tca']],
                     on=['datetime_kst','site'], how='left')
    df['hour'] = df.datetime_kst.dt.hour
    df['date'] = df.datetime_kst.dt.normalize()
    df['z'] = (df.cf - df.mu) / df.sigma.clip(lower=1e-3)

    df['is_flag'] = ((df.cf < 0.03) & (df.mu > 0.20) &
                     (df.dc10Tca < 7) & (df.z < -3.0) &
                     (df.hour.between(8, 16))).astype(int)
    df = df.sort_values(['date', 'site', 'hour']).reset_index(drop=True)

    def confirm(g):
        g = g.sort_values('hour')
        flags = g.is_flag.values
        hours = g.hour.values
        out = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            if flags[i] != 1: continue
            prev = any(hours[j]==hours[i]-1 and flags[j]==1 for j in range(i))
            nxt  = any(hours[j]==hours[i]+1 and flags[j]==1 for j in range(i+1,len(g)))
            out[i] = int(prev or nxt)
        g['confirmed'] = out
        return g

    df = df.groupby(['date','site'], group_keys=False).apply(confirm)
    return df[['datetime_kst','site','is_flag','confirmed']]


def main():
    print("=" * 60)
    print("Building Phase 2 outage labels (v3.5 relaxed + neighbor)")
    print("=" * 60)

    ts = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    ts['datetime_kst'] = pd.to_datetime(ts.datetime_kst)

    # Val (2024-10-16 ~)
    print("\n[val period]")
    val = load_phase1_pred("val")
    val_lab = detect_outage(val, ts)
    print(f"  rows: {len(val_lab):,}, raw flag: {val_lab.is_flag.sum()}, confirmed: {val_lab.confirmed.sum()}")

    # Test (2025)
    print("\n[test period]")
    test = load_phase1_pred("test")
    test_lab = detect_outage(test, ts)
    print(f"  rows: {len(test_lab):,}, raw flag: {test_lab.is_flag.sum()}, confirmed: {test_lab.confirmed.sum()}")

    combined = pd.concat([val_lab, test_lab], ignore_index=True)
    combined['is_outage'] = combined.confirmed
    out = combined[['datetime_kst','site','is_outage']].drop_duplicates(['datetime_kst','site'])
    print(f"\n[combined] rows: {len(out):,}, positive: {out.is_outage.sum()} ({out.is_outage.mean()*100:.3f}%)")

    out_path = ROOT / "data/processed/phase2_outage_labels.parquet"
    out.to_parquet(out_path, index=False)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
