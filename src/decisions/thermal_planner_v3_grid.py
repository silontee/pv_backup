"""v3 grid search — RG fallback + threshold tuning.

추가 변경 (실험용):
- INCREASE 조건에 RG-based fallback 추가:  RG >= TH_ON 도 trigger
- HOLD 조건에 RG-based fallback 추가:      RG >= TH_HOLD 도 trigger
  → P1 mode (forward=0)에서도 active state 진입 가능

Grid:
- TH_ON ∈ {4, 5, 6, 8}
- TH_HOLD ∈ {3, 4, 5}
- MIN_HOLD ∈ {1, 2}
- RELEASE_RATE ∈ {6, 10}
"""
import sys, itertools
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/decisions"))

from thermal_planner_v2 import load_inputs, demand_profile, THERMAL_MIN, ONLINE_MAX, RESERVE_K

TH_NOISE       = 3.0
TH_RELEASE     = 3.0
RAMP_UP_RATE   = 30.0
LOOKBACK_K     = 2
RESERVE_FRAC = {'HIGH': 1.0, 'MEDIUM': 0.5, 'LOW': 0.0}


def planner_run_v3_tuned(daily_inputs, mode, p2_full,
                          th_on, th_hold, min_hold, release_rate, use_rg_fallback=True):
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
        gap_history = []

        for i, r in df_day.iterrows():
            h = r.hour
            pv_d1 = r.pv_d1_mw
            pv_actual = r.pv_actual_mw
            sigma = r.pv_sigma_mw

            RG = float(np.mean(gap_history[-LOOKBACK_K:])) if gap_history else 0.0

            if mode == 'phase1+2':
                p2_for_h = p2_port[(p2_port.date == r.datetime_kst.normalize()) &
                                    (p2_port.issue_hour == h)]
                fwd = {}
                for _, r2 in p2_for_h.iterrows():
                    target_t = r2.target_dt
                    d1_row = df_day[df_day.datetime_kst == target_t]
                    if len(d1_row) == 0: continue
                    fwd[int(r2.lead)] = float(d1_row.iloc[0].pv_d1_mw) - float(r2.pv_p2_mw)
                FG1 = fwd.get(1, 0.0)
                near = [fwd.get(k, np.nan) for k in (1,2,3)]
                near = [v for v in near if not np.isnan(v)]
                FGmean3 = float(np.mean(near)) if near else 0.0
                FGmax3  = float(max(near, key=abs)) if near else 0.0
            else:
                FG1, FGmean3, FGmax3 = 0.0, 0.0, 0.0

            response_now = max(0.0, min(ONLINE_MAX - THERMAL_MIN, RG))

            in_deadband = (abs(RG) < TH_NOISE and
                           abs(FG1) < TH_NOISE and
                           abs(FGmean3) < TH_NOISE)

            increase_cond = (FG1 >= th_on) or (FGmean3 >= th_on)
            hold_cond     = (FGmean3 >= th_hold) or (FGmax3 >= th_on)
            if use_rg_fallback:
                increase_cond = increase_cond or (RG >= th_on)
                hold_cond     = hold_cond or (RG >= th_hold)
            release_ready = (RG < TH_RELEASE and FGmean3 < th_hold and FGmax3 < th_on
                             and hold_timer >= min_hold and prev_response > 0)

            if in_deadband and prev_response <= TH_NOISE:
                new_state = 'KEEP'; response_target = 0.0; reserve = 'LOW'
            elif increase_cond:
                new_state = 'INCREASE'
                response_target = max(response_now, FG1, FGmean3, RG if use_rg_fallback else 0)
                reserve = 'HIGH'
            elif hold_cond and prev_response > 0:
                new_state = 'HOLD'
                response_target = max(prev_response, response_now)
                reserve = 'MEDIUM'
            elif release_ready:
                rel = release_rate * (1.5 if RG < 0 else 1.0)
                new_state = 'DELAYED_RELEASE'
                response_target = max(response_now, prev_response - rel)
                reserve = 'LOW'
                if response_target <= TH_NOISE:
                    new_state = 'KEEP'; response_target = 0.0
            else:
                new_state = 'KEEP'
                response_target = max(0.0, prev_response - release_rate) if prev_response > 0 else 0.0
                if response_target <= TH_NOISE:
                    response_target = 0.0
                reserve = 'LOW'

            # MIN_HOLD restriction
            if prev_state == 'INCREASE' and new_state == 'DELAYED_RELEASE' and hold_timer < min_hold:
                new_state = 'HOLD'
                response_target = max(prev_response, response_now)
                reserve = 'MEDIUM'

            # Ramp up
            if response_target > prev_response:
                response_target = min(response_target, prev_response + RAMP_UP_RATE)
            response_target = max(0.0, min(ONLINE_MAX - THERMAL_MIN, response_target))

            # Hold timer
            if new_state == 'INCREASE':
                hold_timer = 1 if prev_state != 'INCREASE' else hold_timer + 1
            elif new_state == 'HOLD' and response_target > 0:
                hold_timer += 1
            else:
                hold_timer = 0

            instant_gap = pv_d1 - pv_actual
            shortfall   = max(0.0, instant_gap - response_target)
            over_commit = max(0.0, response_target - instant_gap)

            rows.append({
                'datetime_kst': r.datetime_kst,
                'date': r.datetime_kst.normalize(),
                'hour': h, 'mode': mode,
                'instant_gap_mw': instant_gap,
                'correction': response_target, 'state': new_state,
                'shortfall_mw': shortfall, 'over_commit_mw': over_commit,
            })
            prev_response = response_target
            prev_state = new_state
            gap_history.append(pv_d1 - pv_actual)

    return pd.DataFrame(rows)


def kpi(log):
    sc = (log.state.values[1:] != log.state.values[:-1]).sum() if len(log) > 1 else 0
    sgn = (np.sign(log.correction.values[1:]) * np.sign(log.correction.values[:-1]) < 0).sum()
    return {
        'shortfall': float(log.shortfall_mw.sum()),
        'over_commit': float(log.over_commit_mw.sum()),
        'state_chg': int(sc),
        'sign_flips': int(sgn),
        'mean|corr|': float(log.correction.abs().mean()),
    }


def event_kpi(log, dates):
    out = []
    for d in dates:
        sub = log[log.date == pd.Timestamp(d)]
        if len(sub):
            out.append(float(sub.shortfall_mw.sum()))
        else:
            out.append(np.nan)
    return out


def main():
    print("=" * 70)
    print("v3 grid search — RG fallback + threshold tuning")
    print("=" * 70)

    port1, _ = load_inputs()
    p2_full = pd.read_parquet(ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test.parquet")
    port1['hour'] = port1['datetime_kst'].dt.hour
    port1 = port1[port1.hour.between(9,17)].copy()
    port1['date'] = port1['datetime_kst'].dt.normalize()
    daily_inputs = []
    for date, g in port1.groupby('date'):
        g['demand_mw'] = g['hour'].apply(demand_profile)
        daily_inputs.append((date, g))

    # v2 baseline (이미 저장된 결과 로드)
    log_v2_p1  = pd.read_parquet(ROOT / "pv/experiments/thermal_planner_v2/log_phase1.parquet")
    log_v2_p12 = pd.read_parquet(ROOT / "pv/experiments/thermal_planner_v2/log_phase1plus2.parquet")
    log_v2_p1['date']  = pd.to_datetime(log_v2_p1['date'])
    log_v2_p12['date'] = pd.to_datetime(log_v2_p12['date'])
    k_v2_p1  = kpi(log_v2_p1)
    k_v2_p12 = kpi(log_v2_p12)
    events = ['2025-03-23','2025-04-26','2025-05-04']
    ev_v2_p1  = event_kpi(log_v2_p1, events)
    ev_v2_p12 = event_kpi(log_v2_p12, events)
    print(f"v2 P1   : SF={k_v2_p1['shortfall']:7.1f}  OC={k_v2_p1['over_commit']:7.1f}  "
          f"chg={k_v2_p1['state_chg']:4d}  flip={k_v2_p1['sign_flips']:3d}")
    print(f"v2 P1+2 : SF={k_v2_p12['shortfall']:7.1f}  OC={k_v2_p12['over_commit']:7.1f}  "
          f"chg={k_v2_p12['state_chg']:4d}  flip={k_v2_p12['sign_flips']:3d}  "
          f"events: {ev_v2_p12[0]:.1f}/{ev_v2_p12[1]:.1f}/{ev_v2_p12[2]:.1f}")
    print()

    # Grid
    grid = list(itertools.product(
        [4, 5, 6, 8],   # th_on
        [3, 4, 5],      # th_hold
        [1, 2],         # min_hold
        [6, 10],        # release
    ))
    print(f"Grid size: {len(grid)} combos × 2 modes\n")

    results = []
    for gi, (th_on, th_hold, min_hold, rel) in enumerate(grid):
        if th_hold > th_on:
            continue  # invalid (TH_HOLD > TH_ON 무의미)
        log_p1  = planner_run_v3_tuned(daily_inputs, 'phase1', p2_full,
                                       th_on, th_hold, min_hold, rel, use_rg_fallback=True)
        log_p12 = planner_run_v3_tuned(daily_inputs, 'phase1+2', p2_full,
                                       th_on, th_hold, min_hold, rel, use_rg_fallback=True)
        k1  = kpi(log_p1)
        k12 = kpi(log_p12)
        ev_p12 = event_kpi(log_p12, events)
        ev_p1  = event_kpi(log_p1, events)

        # Phase 2 marginal value
        delta_sf = k12['shortfall'] - k1['shortfall']
        delta_pct = delta_sf / k1['shortfall'] * 100 if k1['shortfall'] > 0 else 0

        # vs v2 P1+2
        sf_vs_v2 = k12['shortfall'] - k_v2_p12['shortfall']
        sf_pct = sf_vs_v2 / k_v2_p12['shortfall'] * 100

        results.append({
            'th_on': th_on, 'th_hold': th_hold, 'min_hold': min_hold, 'release': rel,
            'p1_SF': k1['shortfall'], 'p12_SF': k12['shortfall'], 'p12_OC': k12['over_commit'],
            'p12_chg': k12['state_chg'], 'p12_flip': k12['sign_flips'],
            'mean_corr': k12['mean|corr|'],
            'P2_value_pct': delta_pct,
            'vs_v2_p12_SF_pct': sf_pct,
            'ev_03_23': ev_p12[0], 'ev_04_26': ev_p12[1], 'ev_05_04': ev_p12[2],
        })
        print(f"  {gi+1:2d}/{len(grid)} TH_ON={th_on} TH_HOLD={th_hold} MIN={min_hold} REL={rel} "
              f"→ P1+2 SF={k12['shortfall']:7.1f} ({sf_pct:+5.1f}% vs v2)  "
              f"OC={k12['over_commit']:6.0f}  chg={k12['state_chg']:4d}  flip={k12['sign_flips']:3d}")

    df = pd.DataFrame(results).sort_values('p12_SF')
    print("\n" + "=" * 70)
    print("Top 12 by P1+2 shortfall (낮을수록 좋음)")
    print("=" * 70)
    print(df.head(12).round(1).to_string(index=False))

    print("\n" + "=" * 70)
    print("Pareto candidates (낮은 SF + 낮은 sign_flip + 낮은 OC)")
    print("=" * 70)
    # composite score
    df['score'] = (df.p12_SF / k_v2_p12['shortfall']
                    + 0.3 * df.p12_OC / k_v2_p12['over_commit']
                    + 0.001 * df.p12_chg
                    + 0.005 * df.p12_flip)
    print(df.sort_values('score').head(8).round(1).to_string(index=False))

    # 저장
    out = ROOT / "pv/experiments/thermal_planner_v3_test"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "grid_search.csv", index=False)
    print(f"\n저장: {out}/grid_search.csv")


if __name__ == "__main__":
    main()
