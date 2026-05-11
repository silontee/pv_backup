"""Thermal planner v3 spec — *test only* (코드 마이그레이션 전 비교 실험).

Implements plan/active/lng/plan.md v3:
- KEEP / INCREASE / HOLD / DELAYED_RELEASE state machine
- TH_NOISE=3, TH_HOLD=5, TH_ON=8, TH_RELEASE=3
- MIN_HOLD_HOURS=2, RAMP_UP_RATE=30, RELEASE_RATE=10
- reserve_carry HIGH/MED/LOW (sigma·1.282·cap_total scaling)

Outputs same KPI columns as v2 for direct comparison.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/decisions"))

from thermal_planner_v2 import load_inputs, demand_profile, THERMAL_MIN, ONLINE_MAX, RESERVE_K

# === v3 params (per plan/active/lng/plan.md §7) ===
TH_NOISE       = 3.0
TH_HOLD        = 5.0
TH_ON          = 8.0
TH_RELEASE     = 3.0
MIN_HOLD_HOURS = 2
RAMP_UP_RATE   = 30.0
RELEASE_RATE   = 10.0
LOOKBACK_K     = 2

RESERVE_FRAC = {'HIGH': 1.0, 'MEDIUM': 0.5, 'LOW': 0.0}
ACTIVE_HOURS = (9, 17)


def planner_run_v3(daily_inputs, mode, p2_full):
    """Run v3 spec planner. mode ∈ {'phase1', 'phase1+2'}."""
    rows = []
    p2 = p2_full.copy()
    p2['target_dt'] = pd.to_datetime(p2['target_dt'])
    p2['date'] = p2['target_dt'].dt.normalize()
    p2['mu_p2_kw'] = p2['mu_phase2'] * p2['cap']
    p2_port = p2.groupby(['date','issue_hour','lead','target_dt'], as_index=False).agg(
        pv_p2_mw=('mu_p2_kw', lambda s: s.sum()/1000.0))

    for date, df_day in daily_inputs:
        df_day = df_day.sort_values('datetime_kst').reset_index(drop=True)
        prev_response = 0.0
        prev_state = 'KEEP'
        hold_timer = 0
        gap_history = []  # list of (P1 - actual)

        for i, r in df_day.iterrows():
            h = r.hour
            pv_d1 = r.pv_d1_mw
            pv_actual = r.pv_actual_mw
            sigma = r.pv_sigma_mw

            # === RG (look-back avg) ===
            if len(gap_history) >= 1:
                RG = float(np.mean(gap_history[-LOOKBACK_K:]))
            else:
                RG = 0.0

            # === Forward gap stats from Phase 2 issued at hour h ===
            if mode == 'phase1+2':
                p2_for_h = p2_port[(p2_port.date == r.datetime_kst.normalize()) &
                                    (p2_port.issue_hour == h)]
                fwd_per_lead = {}  # lead -> gap (signed P1 - P2)
                for _, r2 in p2_for_h.iterrows():
                    target_t = r2.target_dt
                    d1_target_row = df_day[df_day.datetime_kst == target_t]
                    if len(d1_target_row) == 0:
                        continue
                    d1_val = float(d1_target_row.iloc[0].pv_d1_mw)
                    fwd_per_lead[int(r2.lead)] = d1_val - float(r2.pv_p2_mw)
                FG1     = fwd_per_lead.get(1, 0.0)
                near    = [fwd_per_lead.get(k, np.nan) for k in (1, 2, 3)]
                near    = [v for v in near if not np.isnan(v)]
                FGmean3 = float(np.mean(near)) if near else 0.0
                FGmax3  = float(max(near, key=abs)) if near else 0.0  # signed max-abs
            else:
                FG1, FGmean3, FGmax3 = 0.0, 0.0, 0.0

            # === response_now (current correction candidate) ===
            response_now = max(0.0, min(ONLINE_MAX - THERMAL_MIN, RG))

            # === KEEP deadband check ===
            in_deadband = (abs(RG) < TH_NOISE and
                           abs(FG1) < TH_NOISE and
                           abs(FGmean3) < TH_NOISE)

            # === Active state evaluation ===
            increase_cond = (FG1 >= TH_ON) or (FGmean3 >= TH_ON)
            hold_cond     = (FGmean3 >= TH_HOLD) or (FGmax3 >= TH_ON)
            release_ready = (RG < TH_RELEASE and FGmean3 < TH_HOLD and FGmax3 < TH_ON
                             and hold_timer >= MIN_HOLD_HOURS and prev_response > 0)

            # === State machine (priority: INCREASE > HOLD > DELAYED_RELEASE > KEEP) ===
            if in_deadband and prev_response <= TH_NOISE:
                # 8.1a: pure deadband (no carried response)
                new_state = 'KEEP'
                response_target = 0.0
                reserve = 'LOW'
            elif increase_cond:
                new_state = 'INCREASE'
                response_target = max(response_now, FG1, FGmean3)
                reserve = 'HIGH'
            elif hold_cond and prev_response > 0:
                new_state = 'HOLD'
                response_target = max(prev_response, response_now)
                reserve = 'MEDIUM'
            elif release_ready:
                # negative RG accelerates release
                rel_rate = RELEASE_RATE * (1.5 if RG < 0 else 1.0)
                new_state = 'DELAYED_RELEASE'
                response_target = max(response_now, prev_response - rel_rate)
                reserve = 'LOW'
                # 8.3 fallback: if response decayed back into deadband, transition to KEEP
                if response_target <= TH_NOISE:
                    new_state = 'KEEP'
                    response_target = 0.0
            else:
                # Fallback: stay in KEEP (or carry prev if HOLD-like but response was 0)
                new_state = 'KEEP'
                response_target = max(0.0, prev_response - RELEASE_RATE) if prev_response > 0 else 0.0
                if response_target <= TH_NOISE:
                    response_target = 0.0
                reserve = 'LOW'

            # === MIN_HOLD restriction: forbid INCREASE -> DELAYED_RELEASE before timer satisfied ===
            if prev_state == 'INCREASE' and new_state == 'DELAYED_RELEASE' and hold_timer < MIN_HOLD_HOURS:
                # force HOLD instead
                new_state = 'HOLD'
                response_target = max(prev_response, response_now)
                reserve = 'MEDIUM'

            # === Ramp-up constraint ===
            if response_target > prev_response:
                response_target = min(response_target, prev_response + RAMP_UP_RATE)
            response_target = max(0.0, min(ONLINE_MAX - THERMAL_MIN, response_target))

            # === Hold timer update ===
            if new_state == 'INCREASE':
                if prev_state != 'INCREASE':
                    hold_timer = 1   # fresh strong deficit -> reset
                else:
                    hold_timer += 1
            elif new_state == 'HOLD' and response_target > 0:
                hold_timer += 1
            elif new_state in ('DELAYED_RELEASE', 'KEEP'):
                hold_timer = 0

            # === Reserve quantitative proxy ===
            reserve_base = RESERVE_K * sigma   # MW (sigma already in MW for portfolio)
            reserve_mw = RESERVE_FRAC[reserve] * reserve_base

            # === KPI: PV-only evaluation ===
            instant_gap = pv_d1 - pv_actual
            correction = response_target  # final LNG response = correction in v2 lingo
            shortfall   = max(0.0, instant_gap - correction)
            over_commit = max(0.0, correction - instant_gap)

            rows.append({
                'datetime_kst': r.datetime_kst, 'date': r.datetime_kst.normalize(),
                'hour': h, 'mode': mode,
                'pv_d1_mw': pv_d1, 'pv_actual_mw': pv_actual, 'sigma_mw': sigma,
                'RG': RG, 'FG1': FG1, 'FGmean3': FGmean3, 'FGmax3': FGmax3,
                'instant_gap_mw': instant_gap,
                'response_now': response_now, 'response_target': correction,
                'state': new_state, 'reserve': reserve, 'reserve_mw': reserve_mw,
                'hold_timer': hold_timer,
                'shortfall_mw': shortfall, 'over_commit_mw': over_commit,
                # v2-compat aliases
                'correction': correction, 'realized_gap': RG, 'forward_gap': FGmax3,
                'signal': max([RG, FG1, FGmean3, FGmax3], key=abs),
            })
            prev_response = correction
            prev_state = new_state
            gap_history.append(pv_d1 - pv_actual)

    return pd.DataFrame(rows)


def kpis(log, label):
    sc = (log.state.values[1:] != log.state.values[:-1]).sum()
    sgn = (np.sign(log.correction.values[1:]) * np.sign(log.correction.values[:-1]) < 0).sum()
    return {
        'label': label,
        'shortfall_MWh': float(log.shortfall_mw.sum()),
        'over_commit_MWh': float(log.over_commit_mw.sum()),
        'state_changes': int(sc),
        'correction_sign_flips': int(sgn),
        'mean_correction_abs': float(log.correction.abs().mean()),
        'shortage_hours': int((log.shortfall_mw > 0.5).sum()),
        'over_commit_hours': int((log.over_commit_mw > 0.5).sum()),
    }


def event_breakdown(log, label, dates):
    rows = []
    for d in dates:
        sub = log[log.date == pd.Timestamp(d)]
        if len(sub) == 0:
            continue
        sc = (sub.state.values[1:] != sub.state.values[:-1]).sum() if len(sub) > 1 else 0
        rows.append({
            'date': d, 'mode': label,
            'shortfall': float(sub.shortfall_mw.sum()),
            'over_com':  float(sub.over_commit_mw.sum()),
            'state_chg': int(sc),
            'mean|corr|': float(sub.correction.abs().mean()),
        })
    return rows


def main():
    print("=" * 70)
    print("Thermal planner v3 — TEST run (실 코드 마이그레이션 X)")
    print("=" * 70)
    print(f"  thermal: pmin={THERMAL_MIN}, pmax={ONLINE_MAX}")
    print(f"  v3 rule: TH_NOISE={TH_NOISE}, TH_HOLD={TH_HOLD}, TH_ON={TH_ON}, "
          f"TH_RELEASE={TH_RELEASE}, MIN_HOLD={MIN_HOLD_HOURS}h, "
          f"RAMP={RAMP_UP_RATE}, RELEASE={RELEASE_RATE}")

    port1, _ = load_inputs()
    p2_full = pd.read_parquet(ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test.parquet")

    port1['hour'] = port1['datetime_kst'].dt.hour
    port1 = port1[port1.hour.between(*ACTIVE_HOURS)].copy()
    port1['date'] = port1['datetime_kst'].dt.normalize()
    daily_inputs = []
    for date, g in port1.groupby('date'):
        g['demand_mw'] = g['hour'].apply(demand_profile)
        daily_inputs.append((date, g))
    print(f"  daytime hours: {len(port1):,}, days: {len(daily_inputs)}\n")

    print("[v3 — Phase 1 only (reactive)]")
    log_v3_p1 = planner_run_v3(daily_inputs, 'phase1', p2_full)
    print(f"  rows: {len(log_v3_p1)}")

    print("\n[v3 — Phase 1+2 (predictive)]")
    log_v3_p12 = planner_run_v3(daily_inputs, 'phase1+2', p2_full)
    print(f"  rows: {len(log_v3_p12)}")

    # === KPI comparison ===
    print("\n" + "=" * 70)
    print("v3 KPIs (test 2025, daytime 9~17)")
    print("=" * 70)
    k_v3_p1  = kpis(log_v3_p1, 'v3 Phase 1 only')
    k_v3_p12 = kpis(log_v3_p12, 'v3 Phase 1+2')

    # v2 baseline for comparison
    log_v2_p1  = pd.read_parquet(ROOT / "pv/experiments/thermal_planner_v2/log_phase1.parquet")
    log_v2_p12 = pd.read_parquet(ROOT / "pv/experiments/thermal_planner_v2/log_phase1plus2.parquet")
    log_v2_p1['date'] = pd.to_datetime(log_v2_p1['date'])
    log_v2_p12['date'] = pd.to_datetime(log_v2_p12['date'])
    k_v2_p1  = kpis(log_v2_p1, 'v2 Phase 1 only')
    k_v2_p12 = kpis(log_v2_p12, 'v2 Phase 1+2')

    df_kpi = pd.DataFrame([k_v2_p1, k_v3_p1, k_v2_p12, k_v3_p12])
    print(df_kpi.to_string(index=False))

    # === Phase 2 marginal value (predictive vs reactive) ===
    print("\n[Phase 2 marginal value: Phase 1+2 − Phase 1 only]")
    print(f"  v2: shortfall {k_v2_p12['shortfall_MWh']:.1f} − {k_v2_p1['shortfall_MWh']:.1f} = "
          f"{k_v2_p12['shortfall_MWh']-k_v2_p1['shortfall_MWh']:+.1f} "
          f"({(k_v2_p12['shortfall_MWh']-k_v2_p1['shortfall_MWh'])/k_v2_p1['shortfall_MWh']*100:+.1f}%)")
    print(f"  v3: shortfall {k_v3_p12['shortfall_MWh']:.1f} − {k_v3_p1['shortfall_MWh']:.1f} = "
          f"{k_v3_p12['shortfall_MWh']-k_v3_p1['shortfall_MWh']:+.1f} "
          f"({(k_v3_p12['shortfall_MWh']-k_v3_p1['shortfall_MWh'])/k_v3_p1['shortfall_MWh']*100:+.1f}%)")

    # === Event days ===
    print("\n" + "=" * 70)
    print("Event day comparison (정오 시간대 포함)")
    print("=" * 70)
    events = ['2025-03-23','2025-04-26','2025-05-04']
    rows_e = []
    rows_e += event_breakdown(log_v2_p1, 'v2 P1', events)
    rows_e += event_breakdown(log_v3_p1, 'v3 P1', events)
    rows_e += event_breakdown(log_v2_p12, 'v2 P1+2', events)
    rows_e += event_breakdown(log_v3_p12, 'v3 P1+2', events)
    df_e = pd.DataFrame(rows_e).sort_values(['date','mode'])
    print(df_e.to_string(index=False))

    # === State distribution (v3 only) ===
    print("\n" + "=" * 70)
    print("v3 state distribution")
    print("=" * 70)
    print('Phase 1+2:')
    print(log_v3_p12.state.value_counts())
    print('\nReserve carry:')
    print(log_v3_p12.reserve.value_counts())

    # save
    out_dir = ROOT / "pv/experiments/thermal_planner_v3_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_v3_p1.to_parquet(out_dir / "log_phase1.parquet", index=False)
    log_v3_p12.to_parquet(out_dir / "log_phase1plus2.parquet", index=False)
    df_kpi.to_csv(out_dir / "kpi_comparison.csv", index=False)
    df_e.to_csv(out_dir / "event_days.csv", index=False)
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
