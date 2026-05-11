"""Thermal planner v4 — multi-unit fleet allocator (TEST run, 마이그레이션 X).

per plan/active/lng/plan.md (2026-05-07 갱신):
  P_DA,u(t) = 2025 actual LNG (학습 없음, baseline proxy)
  fleet state ∈ {KEEP, INCREASE, HOLD, DELAYED_RELEASE}
  ΔP_req(t) per state (§7.3.1)
  online allocation: w_u ∝ Headroom × Ramp × Priority, normalize, cap by Headroom (§9.3)
  잔여 → §10 offline screening 입력 (이번 step에선 미구현, residual로만 보고)

비교:
  v3 (CS2 단일) vs v4 (10 LNG fleet allocation, online only)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/decisions"))

from thermal_planner_v2 import load_inputs, demand_profile

# === fleet config (plan §6, §7, §8.3) ===
LNG_UNITS = ['CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8','CS1','CS2']

# Priority per plan §11.2 (PoC initial)
PRIORITY = {
    'CS2': 1.0,
    'CG6': 0.8, 'CG7': 0.8, 'CG8': 0.7,
    'CS1': 0.7,
    'CG1': 0.5, 'CG3': 0.5, 'CG5': 0.5,
    'CG2': 0.4, 'CG4': 0.4,
}

# Per-unit ramp limit (MW/h) from thermal_unit_profile (ramp_p95)
# (load from CSV at runtime)

# === fleet state thresholds (plan §8.3 균형형) ===
TH_NOISE   = 3.0
TH_HOLD    = 5.0
TH_ON      = 5.0     # 균형형: 5
TH_RELEASE = 3.0
MIN_HOLD_HOURS = 1
RELEASE_RATE   = 10.0
LOOKBACK_K     = 2
ACTIVE_HOURS   = (9, 17)


def load_unit_specs():
    """thermal_unit_profile.csv 에서 P_min, P_max, Ramp 로드."""
    p = pd.read_csv(ROOT / "data/processed/thermal_unit_profile.csv", encoding='utf-8')
    p.columns = p.columns.str.replace('﻿', '').str.strip()
    out = {}
    for _, r in p.iterrows():
        u = r['호기']
        if u in LNG_UNITS:
            out[u] = {
                'P_min': float(r['pmin_p05_run']),
                'P_max': float(r['pmax_p95']),
                'Ramp':  float(r['ramp_up_p95']),
                'Priority': PRIORITY.get(u, 0.5),
            }
    return out


def load_lng_baseline():
    """data/processed/lng_baseline_test.parquet 로드.
    Returns dict keyed by (unit, datetime_kst) → {P_DA, is_online, Avail, P_min, P_max}
    + 빠른 lookup용 wide df.
    """
    df = pd.read_parquet(ROOT / "data/processed/lng_baseline_test.parquet")
    df['datetime_kst'] = pd.to_datetime(df['datetime_kst'])
    return df


def fleet_state_to_dPreq(state, current_support, FG1, FGmean3, prev_total):
    """plan §7.3.1 매핑."""
    if state == 'KEEP':
        return 0.0
    if state == 'INCREASE':
        return max(current_support, FG1, FGmean3, prev_total)
    if state == 'HOLD':
        return max(prev_total, current_support)
    if state == 'DELAYED_RELEASE':
        return max(current_support, prev_total - RELEASE_RATE)
    return 0.0


def online_allocate(units_at_t, dP_req):
    """plan §9.3.
    units_at_t: list of dicts with keys [unit, P_DA, P_max, Ramp, Priority, is_online]
    Returns dict {unit -> dP_online, residual: dP_residual}
    """
    online = [u for u in units_at_t if u['is_online'] == 1]
    if not online or dP_req <= 0:
        return {u['unit']: 0.0 for u in units_at_t}, dP_req

    # raw weights
    for u in online:
        u['headroom'] = max(0.0, u['P_max'] - u['P_DA'])
        u['w_raw']    = u['headroom'] * u['Ramp'] * u['Priority']

    W = sum(u['w_raw'] for u in online)
    if W <= 0:
        return {u['unit']: 0.0 for u in units_at_t}, dP_req

    alloc = {u['unit']: 0.0 for u in units_at_t}
    for u in online:
        target = (u['w_raw'] / W) * dP_req
        alloc[u['unit']] = min(target, u['headroom'])

    # 단순화: 잔여 재분배 X (plan §9.3 명시)
    total_alloc = sum(alloc.values())
    residual = max(0.0, dP_req - total_alloc)
    return alloc, residual


# === §10 Offline Startup Screening (v1: single-unit cover priority) ===
# 핵심 변경: 큰 부족(dP_req >= MIN_STARTUP)일 때 분배 대신 offline 호기 1개로 단독 cover
# 이유: 호기 켜면 P_min 이상 강제 출력 → 분배 기반 잔여 채우기는 over-commit 발생
# 단독 cover 시: dP_req가 P_min~P_max 사이 → 정확히 부족분 만큼 출력 가능
PERSIST_TH       = 5.0    # MW: forward gap 지속성 (FGmean3 >= 이 값)
MIN_STARTUP      = 30.0   # MW: 이 이상 부족이면 offline single-unit cover 검토
MIN_RESIDUAL     = 5.0    # MW: 이 이하 잔여는 무시 (operational noise)


def offline_single_cover(units_at_t, dP_req, forward_persistent):
    """plan §10 v1 — single-unit cover.
    dP_req 가 충분히 크고 (>= MIN_STARTUP) persistence 있으면,
    offline 호기 중 P_min <= dP_req <= P_max 인 호기 1개 켜서 단독 cover.

    Priority: Avail 높고 P_min 작은 호기 우선 (작은 unit이 over-commit 적음)
    """
    if dP_req < MIN_STARTUP or not forward_persistent:
        return None  # single-unit cover 안 적용

    candidates = [u for u in units_at_t
                  if u['is_online'] == 0 and u['Avail'] > 0.1
                  and u['P_min'] <= dP_req <= u['P_max']]
    if not candidates:
        return None

    # 가장 작은 P_min 우선 (over-commit 최소화), 동률시 Avail 높은 순
    chosen = sorted(candidates, key=lambda u: (u['P_min'], -u['Avail']))[0]
    alloc = {u['unit']: 0.0 for u in units_at_t}
    alloc[chosen['unit']] = dP_req   # 정확히 부족분 만큼 출력
    return alloc, chosen['unit']


def offline_residual_fill(units_at_t, dP_residual, forward_persistent):
    """plan §10 v0 — 잔여 채우기 (online cap 후).
    dP_residual >= P_min 일 때만 호기 startup. 작은 잔여는 무시.
    """
    if dP_residual < MIN_RESIDUAL or not forward_persistent:
        return {u['unit']: 0.0 for u in units_at_t}, dP_residual

    offline = [u for u in units_at_t if u['is_online'] == 0 and u['Avail'] > 0.1]
    if not offline:
        return {u['unit']: 0.0 for u in units_at_t}, dP_residual

    alloc = {u['unit']: 0.0 for u in units_at_t}
    for u in sorted(offline, key=lambda u: (u['P_min'], -u['Avail'])):
        if dP_residual < u['P_min']: continue
        give = min(dP_residual, u['P_max'])
        give = max(give, u['P_min'])
        alloc[u['unit']] = give
        dP_residual = max(0.0, dP_residual - give)
        if dP_residual <= 0: break

    return alloc, dP_residual


def planner_run_v4(daily_inputs, mode, p2_full, baseline_df, specs):
    rows = []
    p2 = p2_full.copy()
    p2['target_dt'] = pd.to_datetime(p2['target_dt'])
    p2['date'] = p2['target_dt'].dt.normalize()
    p2['mu_p2_kw'] = p2['mu_phase2'] * p2['cap']
    p2_port = p2.groupby(['date','issue_hour','lead','target_dt'], as_index=False).agg(
        pv_p2_mw=('mu_p2_kw', lambda s: s.sum()/1000.0))

    # baseline lookup: indexed by (unit, datetime_kst)
    bl = baseline_df.set_index(['unit','datetime_kst'])

    for date, df_day in daily_inputs:
        df_day = df_day.sort_values('datetime_kst').reset_index(drop=True)
        prev_total = 0.0
        prev_state = 'KEEP'
        hold_timer = 0
        gap_history = []
        prev_alloc = {u: 0.0 for u in LNG_UNITS}

        for i, r in df_day.iterrows():
            h = r.hour
            t = r.datetime_kst
            pv_d1     = r.pv_d1_mw
            pv_actual = r.pv_actual_mw

            # === RG ===
            RG = float(np.mean(gap_history[-LOOKBACK_K:])) if gap_history else 0.0
            current_support = max(0.0, RG)

            # === Forward gap (Phase 1+2 only) ===
            if mode == 'phase1+2':
                p2_for_h = p2_port[(p2_port.date == t.normalize()) &
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

            # === Fleet state machine (plan §8.2~§8.3) ===
            in_deadband = (abs(RG) < TH_NOISE and abs(FG1) < TH_NOISE and abs(FGmean3) < TH_NOISE)
            increase_cond = ((FG1 >= TH_ON) or (FGmean3 >= TH_ON) or (RG >= TH_ON))   # RG fallback
            hold_cond     = ((FGmean3 >= TH_HOLD) or (FGmax3 >= TH_ON) or (RG >= TH_HOLD))
            release_ready = (RG < TH_RELEASE and FGmean3 < TH_HOLD and FGmax3 < TH_ON
                              and hold_timer >= MIN_HOLD_HOURS and prev_total > 0)

            if in_deadband and prev_total <= TH_NOISE:
                state = 'KEEP'
            elif increase_cond:
                state = 'INCREASE'
            elif hold_cond and prev_total > 0:
                state = 'HOLD'
            elif release_ready:
                state = 'DELAYED_RELEASE'
            else:
                state = 'KEEP'

            # MIN_HOLD restriction
            if prev_state == 'INCREASE' and state == 'DELAYED_RELEASE' and hold_timer < MIN_HOLD_HOURS:
                state = 'HOLD'

            dP_req = fleet_state_to_dPreq(state, current_support, FG1, FGmean3, prev_total)

            # === Per-unit baseline at t ===
            units_at_t = []
            for u in LNG_UNITS:
                try:
                    row_b = bl.loc[(u, t)]
                    P_DA     = float(row_b.P_DA_mw)
                    is_on    = int(row_b.is_online)
                    Avail    = float(row_b.Avail)
                except KeyError:
                    P_DA, is_on, Avail = 0.0, 0, 0.0
                spec = specs[u]
                units_at_t.append({
                    'unit': u, 'P_DA': P_DA, 'P_min': spec['P_min'], 'P_max': spec['P_max'],
                    'Ramp': spec['Ramp'], 'Priority': spec['Priority'],
                    'is_online': is_on, 'Avail': Avail,
                })

            forward_persistent = (FGmean3 >= PERSIST_TH)

            # === Step 1: Single-unit offline cover 시도 (큰 부족) ===
            single_cover = offline_single_cover(units_at_t, dP_req, forward_persistent)

            if single_cover is not None:
                # offline 호기 1개로 단독 cover — online 분배 X
                alloc, started_unit = single_cover
                residual_after_online = 0.0
                residual_final = 0.0
                used_single = 1
            else:
                # === Step 2: Online allocation (§9.3) ===
                alloc_on, residual_after_online = online_allocate(units_at_t, dP_req)
                # === Step 3: Offline residual fill (§10 v0, 보완) ===
                alloc_off, residual_final = offline_residual_fill(units_at_t, residual_after_online, forward_persistent)
                alloc = {u: alloc_on.get(u, 0.0) + alloc_off.get(u, 0.0) for u in LNG_UNITS}
                used_single = 0

            # === Ramp constraint per unit (vs prev hour) ===
            for u in LNG_UNITS:
                ramp_max = specs[u]['Ramp']
                alloc[u] = min(alloc[u], prev_alloc[u] + ramp_max)
                alloc[u] = max(alloc[u], prev_alloc[u] - RELEASE_RATE)  # release rate
                alloc[u] = max(0.0, alloc[u])

            total_alloc = sum(alloc.values())
            # startup count: alloc > 0 인데 prev_alloc == 0 인 호기 (offline → online 전환)
            startup_count = sum(1 for u in LNG_UNITS if alloc[u] > 0 and prev_alloc[u] == 0)

            # === KPI ===
            instant_gap = pv_d1 - pv_actual
            shortfall   = max(0.0, instant_gap - total_alloc)
            over_commit = max(0.0, total_alloc - instant_gap)

            # Hold timer
            if state == 'INCREASE':
                hold_timer = 1 if prev_state != 'INCREASE' else hold_timer + 1
            elif state == 'HOLD' and total_alloc > 0:
                hold_timer += 1
            else:
                hold_timer = 0

            row = {
                'datetime_kst': t, 'date': t.normalize(),
                'hour': h, 'mode': mode,
                'pv_d1_mw': pv_d1, 'pv_actual_mw': pv_actual,
                'RG': RG, 'FG1': FG1, 'FGmean3': FGmean3, 'FGmax3': FGmax3,
                'state': state, 'dP_req': dP_req,
                'dP_residual_after_online': residual_after_online,
                'dP_residual_final': residual_final,
                'startup_count': startup_count,
                'used_single_cover': used_single,
                'total_alloc_mw': total_alloc,
                'instant_gap_mw': instant_gap,
                'shortfall_mw': shortfall, 'over_commit_mw': over_commit,
                'correction': total_alloc,  # v2 호환
                'dP_residual': residual_final,  # legacy 호환
            }
            for u in LNG_UNITS:
                row[f'dP_{u}'] = alloc[u]
            rows.append(row)

            prev_total = total_alloc
            prev_state = state
            prev_alloc = alloc.copy()
            gap_history.append(pv_d1 - pv_actual)

    return pd.DataFrame(rows)


def kpi(log):
    sc  = (log.state.values[1:] != log.state.values[:-1]).sum() if len(log) > 1 else 0
    sgn = (np.sign(log.correction.values[1:]) * np.sign(log.correction.values[:-1]) < 0).sum()
    return {
        'shortfall_MWh':   float(log.shortfall_mw.sum()),
        'over_commit_MWh': float(log.over_commit_mw.sum()),
        'state_changes':   int(sc),
        'sign_flips':      int(sgn),
        'mean|corr|':      float(log.correction.abs().mean()),
        'residual_MWh':    float(log.dP_residual.sum()),
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


def unit_breakdown(log):
    """호기별 누적 추가 출력량 (MWh)."""
    return {u: float(log[f'dP_{u}'].sum()) for u in LNG_UNITS}


def main():
    print("=" * 70)
    print("Thermal planner v4 — multi-unit fleet allocator (TEST)")
    print("=" * 70)
    print(f"  fleet: {LNG_UNITS}")
    print(f"  state thresholds: TH_NOISE={TH_NOISE} TH_HOLD={TH_HOLD} TH_ON={TH_ON} "
          f"MIN_HOLD={MIN_HOLD_HOURS}h RELEASE={RELEASE_RATE}")

    specs = load_unit_specs()
    baseline = load_lng_baseline()
    print(f"  unit specs loaded: {len(specs)}")
    print(f"  baseline rows: {len(baseline):,}")

    port1, _ = load_inputs()
    p2_full = pd.read_parquet(ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet")
    port1['hour'] = port1['datetime_kst'].dt.hour
    port1 = port1[port1.hour.between(*ACTIVE_HOURS)].copy()
    port1['date'] = port1['datetime_kst'].dt.normalize()
    daily_inputs = [(d, g.assign(demand_mw=g.hour.apply(demand_profile)))
                    for d, g in port1.groupby('date')]
    print(f"  daytime hours: {len(port1):,}, days: {len(daily_inputs)}\n")

    print("[v4 — Phase 1 only]")
    log_p1  = planner_run_v4(daily_inputs, 'phase1', p2_full, baseline, specs)
    print(f"  rows: {len(log_p1)}")

    print("\n[v4 — Phase 1+2]")
    log_p12 = planner_run_v4(daily_inputs, 'phase1+2', p2_full, baseline, specs)
    print(f"  rows: {len(log_p12)}")

    # === KPI ===
    print("\n" + "=" * 70)
    print("v4 KPI (test 2025, daytime 9~17)")
    print("=" * 70)
    k1  = kpi(log_p1)
    k12 = kpi(log_p12)
    df = pd.DataFrame([{'mode':'v4 P1', **k1}, {'mode':'v4 P1+2', **k12}])
    print(df.round(1).to_string(index=False))

    print(f"\n[Phase 2 marginal value]")
    print(f"  v4: shortfall {k12['shortfall_MWh']:.1f} − {k1['shortfall_MWh']:.1f} = "
          f"{k12['shortfall_MWh']-k1['shortfall_MWh']:+.1f} "
          f"({(k12['shortfall_MWh']-k1['shortfall_MWh'])/k1['shortfall_MWh']*100:+.1f}%)")

    # vs v2 (simple single-unit)
    log_v2_p1  = pd.read_parquet(ROOT / "pv/experiments/thermal_planner_v2/log_phase1.parquet")
    log_v2_p12 = pd.read_parquet(ROOT / "pv/experiments/thermal_planner_v2/log_phase1plus2.parquet")
    log_v2_p1['date']  = pd.to_datetime(log_v2_p1['date'])
    log_v2_p12['date'] = pd.to_datetime(log_v2_p12['date'])
    print(f"\n[vs v2 baseline (CS2 single)]:")
    print(f"  v2 P1+2 SF:  {log_v2_p12.shortfall_mw.sum():.1f}")
    print(f"  v4 P1+2 SF:  {k12['shortfall_MWh']:.1f}  "
          f"({(k12['shortfall_MWh']-log_v2_p12.shortfall_mw.sum())/log_v2_p12.shortfall_mw.sum()*100:+.1f}%)")

    # === Event days ===
    events = ['2025-03-23','2025-04-26','2025-05-04']
    print("\n" + "=" * 70)
    print("Event days (shortfall MWh)")
    print("=" * 70)
    ev_v4_p12 = event_kpi(log_p12, events)
    ev_v2_p12 = event_kpi(log_v2_p12, events)
    for d, v2, v4 in zip(events, ev_v2_p12, ev_v4_p12):
        delta = (v4 - v2) / v2 * 100 if v2 > 0 else 0
        print(f"  {d}: v2 P1+2={v2:.1f}  →  v4 P1+2={v4:.1f}  ({delta:+.1f}%)")

    # === State distribution ===
    print("\n" + "=" * 70)
    print("v4 P1+2 state distribution + residual")
    print("=" * 70)
    print(log_p12.state.value_counts())
    print(f"\nMean ΔP_residual (offline screening 잠재 입력): "
          f"{log_p12.dP_residual.mean():.2f} MW/h, "
          f"sum {log_p12.dP_residual.sum():.0f} MWh")

    # === Unit breakdown ===
    print("\n" + "=" * 70)
    print("Unit-level cumulative ΔP (MWh, P1+2)")
    print("=" * 70)
    ub = unit_breakdown(log_p12)
    for u in sorted(ub, key=lambda x: -ub[x]):
        print(f"  {u}:  {ub[u]:8.1f} MWh   (priority {PRIORITY[u]})")

    out = ROOT / "pv/experiments/thermal_planner_v4_test"
    out.mkdir(parents=True, exist_ok=True)
    log_p1.to_parquet(out / "log_phase1.parquet", index=False)
    log_p12.to_parquet(out / "log_phase1plus2.parquet", index=False)
    print(f"\n저장: {out}/")


if __name__ == "__main__":
    main()
