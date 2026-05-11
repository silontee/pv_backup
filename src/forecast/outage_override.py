"""실험 B — Phase 2 + outage override + recovery rule (실시간 운영 시뮬레이션).

per plan/active/lng/plan.md outage_v3.5:
  outage_flag_t = (cf<0.03 AND mu_phase1>0.20 AND dc10Tca<7 AND z<-3 AND neighbor)

Real-time interpretation:
  - 시점 t에서 outage 감지 → t+1 ~ EOD 의 mu_phase2 := 0
  - actual_cf > recovery_threshold (0.10 또는 0.15) 가 1h+ 지속되면 normal mode 복귀
  - 즉 "outage detected → blackout forecast" 후 actual recovery까지 0 유지

비교:
  P1 baseline / P2 only / P2 + override (recovery 0.10) / P2 + override (recovery 0.15)
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]


def detect_outage_v3_5(uniq_df):
    """v3.5 relaxed: cf<0.03 + mu>0.20 + cloud<7 + z<-3 + neighbor (t±1)."""
    df = uniq_df[(uniq_df.target_h.between(8, 16)) & (uniq_df.mu_phase1 > 0.15)].copy()
    df['z'] = (df.cf - df.mu_phase1) / df.sigma_phase1.clip(lower=1e-3)
    df['is_flag'] = ((df.cf < 0.03) & (df.mu_phase1 > 0.20) &
                     (df.dc10Tca_target < 7) & (df.z < -3.0)).astype(int)
    df = df.sort_values(['date', 'site', 'target_h']).reset_index(drop=True)

    def confirm(g):
        g = g.sort_values('target_h')
        flags = g.is_flag.values
        hours = g.target_h.values
        out = np.zeros(len(g), dtype=int)
        for i in range(len(g)):
            if flags[i] != 1:
                continue
            prev = any(hours[j] == hours[i] - 1 and flags[j] == 1 for j in range(i))
            nxt = any(hours[j] == hours[i] + 1 and flags[j] == 1 for j in range(i + 1, len(g)))
            out[i] = int(prev or nxt)
        g['confirmed'] = out
        return g

    df = df.groupby(['date', 'site'], group_keys=False).apply(confirm)
    return df[df.confirmed == 1][['date', 'site', 'target_dt']]


def apply_realtime_override(ens, outage_keys, recovery_threshold=0.10, recovery_hours=1):
    """실시간 운영 시뮬레이션 — 사이트별로 target_dt 시간순 진행:
    - 시점 t의 outage_flag → 그 시점부터 blackout 모드 진입
    - 이후 actual이 recovery_threshold 이상 recovery_hours 시간 지속 → 해제
    - blackout 모드 동안 mu_phase2 := 0
    """
    e = ens.copy()
    e['recover_signal'] = (e.cf > recovery_threshold).astype(int)

    # 사이트별 unique target_dt 시계열로 blackout state 결정 → dict
    blackout_lookup = {}  # (site, target_dt) → blackout bool
    for site, g in e.groupby('site'):
        unique_targets = g.drop_duplicates('target_dt').sort_values('target_dt')
        # outage flag per target
        outage_at = {tdt: (g.iloc[0]['date'], site, tdt) in outage_keys
                     for tdt in unique_targets.target_dt}
        # actual recovery per target (전체 첫 row의 cf 사용 — 같은 target은 동일)
        recover_at = dict(zip(unique_targets.target_dt, unique_targets.recover_signal))

        # outage_keys는 (date, site, target_dt) — date 구해서 매핑
        date_per_t = dict(zip(unique_targets.target_dt, unique_targets.date))
        # 정확한 outage 체크
        outage_at = {tdt: (date_per_t[tdt], site, tdt) in outage_keys
                     for tdt in unique_targets.target_dt}

        blackout = False
        recover_streak = 0
        for tdt in unique_targets.target_dt:
            if blackout:
                if recover_at[tdt] == 1:
                    recover_streak += 1
                    if recover_streak >= recovery_hours:
                        blackout = False
                        recover_streak = 0
                else:
                    recover_streak = 0
            else:
                if outage_at[tdt]:
                    blackout = True
                    recover_streak = 0
            blackout_lookup[(site, tdt)] = blackout

    # apply
    e['blackout'] = e.apply(
        lambda r: int(blackout_lookup.get((r['site'], r['target_dt']), False)),
        axis=1
    )
    e['mu_p2_ovr'] = np.where(e.blackout == 1, 0.0, e.mu_phase2)
    return e


def nmae(df, col_pred):
    err = (df.cf - df[col_pred]).abs() * df.cap
    return err.sum() / df.cap.sum() * 100


def evaluate(ens, label):
    p1  = nmae(ens, 'mu_phase1')
    p2  = nmae(ens, 'mu_phase2')
    p2o = nmae(ens, 'mu_p2_ovr')
    return {'label': label,
            'P1': p1, 'P2': p2, 'P2_override': p2o,
            'Δ_override': p2o - p2}


def main():
    ens = pd.read_parquet(ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test.parquet")
    ens['target_dt'] = pd.to_datetime(ens.target_dt)
    ens['target_h']  = ens.target_dt.dt.hour
    ens['date']      = pd.to_datetime(ens.date)
    print(f"Ensemble rows: {len(ens):,}")

    # outage detection (v3.5)
    uniq = ens.drop_duplicates(['date', 'site', 'target_dt']).copy()
    out_keys_df = detect_outage_v3_5(uniq)
    outage_keys = set(zip(out_keys_df.date, out_keys_df.site, out_keys_df.target_dt))
    print(f"v3.5 confirmed outages: {len(outage_keys)}")

    # override variants
    for thr, hrs in [(0.10, 1), (0.15, 1), (0.10, 2)]:
        e = apply_realtime_override(ens, outage_keys, thr, hrs)
        e['err_p1']  = (e.cf - e.mu_phase1).abs() * e.cap
        e['err_p2']  = (e.cf - e.mu_phase2).abs() * e.cap
        e['err_p2o'] = (e.cf - e.mu_p2_ovr).abs() * e.cap

        n_blackout = (e.blackout == 1).sum()
        print(f"\n=== Recovery threshold={thr}, hours={hrs} ===")
        print(f"  blackout rows: {n_blackout:,}  ({n_blackout/len(e)*100:.2f}%)")

        def sub_nmae(df, label):
            if len(df) == 0:
                print(f"  {label:35s} n=0"); return
            p1  = df.err_p1.sum() / df.cap.sum() * 100
            p2  = df.err_p2.sum() / df.cap.sum() * 100
            p2o = df.err_p2o.sum() / df.cap.sum() * 100
            print(f"  {label:35s} n={len(df):>7,}  P1={p1:6.3f}%  P2={p2:6.3f}%  P2+ovr={p2o:6.3f}%   Δ={p2o-p2:+.3f}pp")

        sub_nmae(e, '전체')
        sub_nmae(e[e.blackout == 1], 'blackout 시점만')
        sub_nmae(e[e.blackout == 0], 'normal 시점만')
        # cloud-pass 일자
        for d in ['2025-03-23', '2025-04-26', '2025-05-04']:
            sub = e[e.date == pd.Timestamp(d)]
            sub_nmae(sub, f'  {d} 전체')
        # 광양항 10월 사건
        oct_mask = (e.site == '광양항세방') & (e.date.isin([pd.Timestamp(d) for d in ['2025-10-10','2025-10-11','2025-10-12']]))
        sub_nmae(e[oct_mask], '광양항 10/10-12')

        # 저장 (10/1 만)
        if thr == 0.10 and hrs == 1:
            e.drop(columns=['err_p1','err_p2','err_p2o']).to_parquet(
                ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet",
                index=False)
            print(f"\n저장: ensemble_test_overridden.parquet (recovery thr=0.10, hrs=1)")


if __name__ == "__main__":
    main()
