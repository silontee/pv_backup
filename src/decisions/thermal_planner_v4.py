"""Thermal planner v4 — multi-unit fleet allocator (PRODUCTION).

per plan/active/lng/plan.md (2026-05-07 갱신):
  P_DA,u(t) = 2025 actual LNG (학습 없음, baseline proxy)
  fleet state ∈ {KEEP, INCREASE, DELAYED_RELEASE}  (HOLD 제거 2026-05-08)
  ΔP_req(t) per state (§7.3.1)
  online allocation: w_u ∝ Headroom × Ramp × Priority, normalize, cap by Headroom (§9.3)
  offline single-unit cover: dP_req >= 30 MW + persistence → 호기 1개로 단독 cover
  offline residual fill: 잔여 >= P_min 인 호기 startup

Phase 2 입력: ensemble_test_overridden.parquet (outage override 적용)
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

# === fleet state thresholds (plan §7.3, 2026-05-09 data-driven) ===
# data: forward gap |FG1| p95 = 5.98, |slope| p95 = 2.24 (test 2025)
TH_NOISE_FORWARD = 3.0   # |FG1| 이 이하 = quiet (forward gap p80 근사)
TH_INCREASE      = 5.0   # FG1 이 이상 = strong (p95 round)
TH_SLOPE         = 2.0   # |slope| 이 이상 = 명확한 변화 (p95)
MIN_HOLD_HOURS   = 1
RELEASE_RATE     = 8.0   # DELAYED_RELEASE 시에만 적용
LOOKBACK_K       = 2
ACTIVE_HOURS     = (9, 17)

# === System-side reserve cushion (2026-05-09) ===
# 보조서비스 정산금 자료 분석: LNG가 전국 예비력의 ≈50% 담당.
# 우리 PV portfolio 77 MW 의 작은 변동은 계통 내 1차/2차 예비력·AGC·타 조정자원이 흡수.
# DEADBAND 이내 actual_gap 은 LNG 책임 X. 그 이상의 marginal 만 LNG event.
DEADBAND_MW = 4.0


def load_unit_specs():
    """data-driven spec from data/processed/lng_unit_specs.parquet
    (build_lng_unit_specs.py 로 사전 계산됨, 4년치 hourly p90/median 등)."""
    p = pd.read_parquet(ROOT / "data/processed/lng_unit_specs.parquet")
    out = {}
    for _, r in p.iterrows():
        u = r['unit']
        if u in LNG_UNITS:
            out[u] = {
                'P_min':   float(r['P_min']),
                'P_max':   float(r['P_max']),
                'Ramp':    float(r['ramp_up_p90']),    # 상승 한도 (data-driven p90)
                'Ramp_dn': float(r['ramp_dn_p90']),    # 하강 한도 (data-driven p90)
                'mode':    str(r['mode']),             # 'baseload' / 'mid-merit' / 'peaker'
                'Priority': float(r['priority_data']), # backup_score 정규화 (CS1=1.0)
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
    """plan §8.4 (2026-05-08 단순화).
    KEEP 도 즉시_보정필요량 (current_support) 100% 추종 — 잡음 컷 X.
    """
    if state == 'KEEP':
        return current_support   # RG 작아도 채움 (1 MW도)
    if state == 'INCREASE':
        return max(current_support, FG1, FGmean3, prev_total)
    if state == 'DELAYED_RELEASE':
        return max(current_support, prev_total - RELEASE_RATE)
    return current_support


def online_allocate(units_at_t, dP_req):
    """plan §9.3 (2026-05-09 단순화).
    가중치 = 헤드룸 × ramp_up (priority 제거, 제곱 효과 방지).
    """
    online = [u for u in units_at_t if u['is_online'] == 1]
    if not online or dP_req <= 0:
        return {u['unit']: 0.0 for u in units_at_t}, dP_req

    for u in online:
        u['headroom'] = max(0.0, u['P_max'] - u['P_DA'])
        u['w_raw']    = u['headroom'] * u['Ramp']    # priority 제거

    W = sum(u['w_raw'] for u in online)
    if W <= 0:
        return {u['unit']: 0.0 for u in units_at_t}, dP_req

    alloc = {u['unit']: 0.0 for u in units_at_t}
    for u in online:
        target = (u['w_raw'] / W) * dP_req
        alloc[u['unit']] = min(target, u['headroom'])

    total_alloc = sum(alloc.values())
    residual = max(0.0, dP_req - total_alloc)
    return alloc, residual


# === GT vs ST 구분 (분당 LNG CCGT 단지) ===
# CG* = Gas Turbine (빠른 startup, 빠른 ramp)
# CS* = Steam Turbine (느린 startup, 효율 높음)
def unit_type(unit_name):
    return 'GT' if unit_name.startswith('CG') else 'ST'


# === Layer A 호기 켜기 (즉시 출력) — plan §10.5 ===
# actual gap이 P_min 이상 + online cap 초과 시 호기 새로 켬 (즉시 P_min 출력)
# GT 우선 (cold start 빠름 = 즉시 가용)
def layer_a_startup(units_at_t, actual_gap_remaining):
    """Layer A 흐름 안에서 호기 새로 켜는 결정.
    actual_gap_remaining = online 분배 후 잔여 (현재 부족 중 미충족분)
    조건: P_min ≤ actual_gap_remaining ≤ P_max
    GT 우선 (CG*) — 빠른 startup
    """
    candidates = [u for u in units_at_t
                  if u['is_online'] == 0 and u['Avail'] > 0.1
                  and u['P_min'] <= actual_gap_remaining <= u['P_max']]
    if not candidates:
        return None

    chosen = sorted(candidates, key=lambda u: (
        0 if unit_type(u['unit']) == 'GT' else 1,    # ★ GT 우선
        -(u['P_max'] - u['P_min']),                    # 헤드룸 큰 순
        u['P_min'],                                    # P_min 작은 순
        -u['Avail'],
    ))[0]
    return chosen['unit'], actual_gap_remaining


# === Layer B 호기 warm-up (예열 명령) — plan §10.2 ===
# 전기 저장 X → 미리 만들기 X. 다만 호기 *예열* 은 가능.
# trigger: 다음 시각 expected_next_gap > expected_online_cap + GT 가용 가능
# 효과: 출력 0, 다음 시각 online subset에 추가 (예열된 상태)
WARMUP_MIN_GAP = 5.0   # 다음 시각 부족이 이 이상이면 warm-up 검토

def layer_b_warmup(units_at_t, expected_next_gap, online_headroom_total,
                    forward_persistent, warm_up_set):
    """Layer B 예열 명령.
    출력 영향 X (지금 시각). 다음 시각 가용성 (예열 완료 상태) 만.

    GT (CG*) 우선:
      - cold start 30min — lead 1h 안에 가용
      - 작은 부족(P_min 미만)이어도 warm-up 시작 — 다음 시각 분배에 후보
      - 실제 출력은 다음 시각 actual gap 기반 (Layer A) → 작으면 0 유지

    ST (CS*) 는 cold start 1-3h라 lead 1h 부족 → warm-up 후보 X.
    """
    if not forward_persistent: return None
    if expected_next_gap <= max(WARMUP_MIN_GAP, online_headroom_total): return None

    # GT 만 candidate (ST는 lead time 부족)
    candidates = [u for u in units_at_t
                  if u['is_online'] == 0
                  and u['unit'] not in warm_up_set
                  and u['Avail'] > 0.1
                  and unit_type(u['unit']) == 'GT'   # ★ GT 만
                  and u['P_max'] >= WARMUP_MIN_GAP]   # 어느 정도 cover 가능
    if not candidates: return None

    chosen = sorted(candidates, key=lambda u: (
        -(u['P_max'] - u['P_min']),    # 헤드룸 큰 순
        u['P_min'],                     # P_min 작은 순
        -u['Avail'],
    ))[0]
    return chosen['unit']


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
        warm_up_set = set()        # 직전 시각에 warm-up 명령된 호기 (이번 시각 online 후보)
        next_warm_up_set = set()   # 이번 시각에 warm-up 명령 → 다음 시각 online

        for i, r in df_day.iterrows():
            h = r.hour
            t = r.datetime_kst
            pv_d1     = r.pv_d1_mw
            pv_actual = r.pv_actual_mw

            # === Step 1: 신호 계산 ===
            RG = float(np.mean(gap_history[-LOOKBACK_K:])) if gap_history else 0.0  # log/대시보드용만
            instant_gap_now = pv_d1 - pv_actual   # 현재 시각 실측 부족
            # actual gap → effective gap (DEADBAND 차감) 만 LNG 책임
            # DEADBAND 이내 = 계통 자체 흡수 cushion (1차/2차 예비력·AGC·타 조정자원)
            effective_gap_now = max(0.0, instant_gap_now - DEADBAND_MW)
            current_support = effective_gap_now

            # Forward gap + slope (Phase 1+2 only)
            FG1, FG2, FG3, FGmean3, FGmax3, slope = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
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
                FG2 = fwd.get(2, 0.0)
                FG3 = fwd.get(3, 0.0)
                near = [v for v in (FG1, FG2, FG3) if v != 0 or fwd]
                FGmean3 = float(np.mean([fwd[k] for k in (1,2,3) if k in fwd])) if any(k in fwd for k in (1,2,3)) else 0.0
                FGmax3  = float(max([fwd[k] for k in (1,2,3) if k in fwd], key=abs)) if any(k in fwd for k in (1,2,3)) else 0.0
                # slope: lead 1 → lead 3 변화량 / 2시간
                if 1 in fwd and 3 in fwd:
                    slope = (fwd[3] - fwd[1]) / 2

            # === Step 2: Forward signal 분류 (slope-based, plan §7.3) ===
            forward_quiet    = abs(FG1) < TH_NOISE_FORWARD
            forward_strong   = FG1 >= TH_INCREASE
            forward_growing  = (TH_NOISE_FORWARD <= FG1 < TH_INCREASE) and (slope >= TH_SLOPE)
            forward_recovery = (abs(FG1) <= TH_INCREASE) and (slope <= -TH_SLOPE) and (prev_total > 0)

            # === Step 3: 응답상태 (3-state) ===
            if forward_quiet and current_support < 1.0 and prev_total <= 1.0:
                state = 'KEEP'
            elif forward_strong or forward_growing:
                state = 'INCREASE'
            elif forward_recovery and hold_timer >= MIN_HOLD_HOURS:
                state = 'DELAYED_RELEASE'
            else:
                state = 'KEEP'   # 애매 영역 (4.8%) + 작은 신호

            # MIN_HOLD restriction
            if prev_state == 'INCREASE' and state == 'DELAYED_RELEASE' and hold_timer < MIN_HOLD_HOURS:
                state = 'INCREASE'

            # === Step 4: 호기 baseline 정보 (warm_up_set 호기는 online으로 처리) ===
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
                # warm-up 호기는 이번 시각 online subset에 합류 (P_DA = 0, 분배 후보)
                # 실제 출력은 actual gap 기반. P_min 강제 X (over-commit 방지).
                if u in warm_up_set and is_on == 0:
                    P_DA = 0.0          # baseline 그대로 0
                    is_on = 1            # 분배 후보 (예열 완료, 가용)
                    Avail = 1.0
                units_at_t.append({
                    'unit': u, 'P_DA': P_DA, 'P_min': spec['P_min'], 'P_max': spec['P_max'],
                    'Ramp': spec['Ramp'], 'Ramp_dn': spec['Ramp_dn'],
                    'Priority': spec['Priority'], 'mode': spec['mode'],
                    'is_online': is_on, 'Avail': Avail,
                })

            online_headroom_total = sum(max(0, u['P_max'] - u['P_DA'])
                                        for u in units_at_t if u['is_online'] == 1)

            # === Step 5: Layer A — actual gap 분배 ===
            # online 호기에 즉시_보정필요량 분배
            alloc_a, residual_a = online_allocate(units_at_t, current_support)

            # Layer A 잔여 (online cap 초과) — actual gap이 P_min 이상이면 새 호기 켜기
            startup_a = None
            if residual_a >= 1.0:
                startup_a = layer_a_startup(units_at_t, residual_a)

            alloc = {u: alloc_a.get(u, 0.0) for u in LNG_UNITS}
            if startup_a is not None:
                started_u, give = startup_a
                alloc[started_u] = give
                residual_a = 0.0  # 즉시 cover

            # === Step 6: Layer B — warm-up only (출력 X, 다음 시각 가용성만) ===
            forward_persistent = forward_strong or forward_growing
            # 다음 시각 expected_next_gap = max(현재 부족, FG1)
            expected_next_gap = max(current_support, FG1)
            warmup_unit = layer_b_warmup(units_at_t, expected_next_gap,
                                          online_headroom_total, forward_persistent,
                                          warm_up_set)
            if warmup_unit is not None:
                next_warm_up_set.add(warmup_unit)
                # ΔP에 영향 X (출력 0 유지)

            # === Step 7: Ramp constraint (비대칭 + 상태별) ===
            for u in LNG_UNITS:
                ramp_up = specs[u]['Ramp']
                ramp_dn = specs[u]['Ramp_dn']
                # 상승
                alloc[u] = min(alloc[u], prev_alloc[u] + ramp_up)
                # 하강 — state 별
                if state == 'DELAYED_RELEASE':
                    alloc[u] = max(alloc[u], prev_alloc[u] - RELEASE_RATE)
                else:
                    alloc[u] = max(alloc[u], prev_alloc[u] - ramp_dn)
                alloc[u] = max(0.0, alloc[u])

            total_alloc = sum(alloc.values())
            # startup 분리:
            #   startup_real  = 호기가 offline (is_online==0) 인데 ΔP 발생 → 진짜 신규 가동
            #   startup_ramp  = 호기는 online 이지만 prev ΔP=0 → 추가 발전 전환
            #   startup_count = 둘 합 (legacy 호환)
            online_now = {u['unit']: u['is_online'] for u in units_at_t}
            startup_real = sum(1 for u in LNG_UNITS
                                if alloc[u] > 0 and prev_alloc[u] == 0 and online_now.get(u, 0) == 0)
            startup_ramp = sum(1 for u in LNG_UNITS
                                if alloc[u] > 0 and prev_alloc[u] == 0 and online_now.get(u, 0) == 1)
            startup_count = startup_real + startup_ramp

            # === KPI (DEADBAND 적용 — LNG가 책임지는 effective_gap 기준) ===
            instant_gap = pv_d1 - pv_actual
            effective_gap = max(0.0, instant_gap - DEADBAND_MW)
            shortfall   = max(0.0, effective_gap - total_alloc)
            over_commit = max(0.0, total_alloc - effective_gap)

            # Hold timer
            if state == 'INCREASE':
                hold_timer = 1 if prev_state != 'INCREASE' else hold_timer + 1
            else:  # KEEP, DELAYED_RELEASE
                hold_timer = 0

            # dashboard 호환 aliases
            signal = max([RG, FG1, FGmean3, FGmax3], key=abs) if mode == 'phase1+2' else RG
            baseline_lng_mw = sum(u['P_DA'] for u in units_at_t)
            row = {
                'datetime_kst': t, 'date': t.normalize(),
                'hour': h, 'mode': mode,
                'pv_d1_mw': pv_d1, 'pv_actual_mw': pv_actual,
                'RG': RG, 'FG1': FG1, 'FGmean3': FGmean3, 'FGmax3': FGmax3, 'slope': slope,
                # backward-compat (dashboard pages 2/3에서 사용)
                'realized_gap': RG, 'forward_gap': FGmax3, 'signal': signal,
                'state': state, 'dP_req': current_support,    # Layer A의 dP만 (Layer B는 출력 X)
                'dP_residual_after_online': residual_a,
                'dP_residual_final': residual_a if startup_a is None else 0.0,
                'startup_count': startup_count,        # 합 (legacy)
                'startup_real': startup_real,          # 진짜 신규 가동 (offline → 발전)
                'startup_ramp': startup_ramp,          # 추가 발전 전환 (이미 운전 중 + ΔP 시작)
                'used_single_cover': 1 if startup_a is not None else 0,
                'warm_up_unit': warmup_unit if warmup_unit else '',
                # backward-compat (page 3 시각화)
                'baseline_lng_mw': baseline_lng_mw,
                'thermal_dispatch_mw': baseline_lng_mw + total_alloc,
                'demand_mw': 0.0,        # v4: demand 모델 없음 (PV-only KPI)
                'peaker_mw': 0.0,        # v4: peaker 개념 없음 (10 호기 fleet)
                'peaker_start': 0,
                'total_alloc_mw': total_alloc,
                'instant_gap_mw': instant_gap,
                'effective_gap_mw': effective_gap,   # DEADBAND 차감 후 LNG 책임분
                'shortfall_mw': shortfall, 'over_commit_mw': over_commit,
                'correction': total_alloc,  # v2 호환
                'dP_residual': residual_a if startup_a is None else 0.0,  # legacy 호환
            }
            for u in LNG_UNITS:
                row[f'dP_{u}'] = alloc[u]
            rows.append(row)

            prev_total = total_alloc
            prev_state = state
            prev_alloc = alloc.copy()
            gap_history.append(pv_d1 - pv_actual)
            # 다음 시각용 warm-up set 갱신
            warm_up_set = next_warm_up_set
            next_warm_up_set = set()

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
    print(f"  state thresholds (3-state, slope-based): "
          f"TH_NOISE_FORWARD={TH_NOISE_FORWARD} TH_INCREASE={TH_INCREASE} "
          f"TH_SLOPE={TH_SLOPE} MIN_HOLD={MIN_HOLD_HOURS}h RELEASE={RELEASE_RATE}")

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
    print(f"\nMean ΔP_residual: "
          f"{log_p12.dP_residual.mean():.2f} MW/h, "
          f"sum {log_p12.dP_residual.sum():.0f} MWh")

    # === Warm-up + startup 분석 ===
    print("\n" + "=" * 70)
    print("Warm-up (예열 명령) + Startup 분석")
    print("=" * 70)
    n_warm_p1  = (log_p1.warm_up_unit != '').sum()
    n_warm_p12 = (log_p12.warm_up_unit != '').sum()
    print(f"  Phase 1 only — warm-up 발동: {n_warm_p1}회 (forward gap 부재 → 0이 정상)")
    print(f"  Phase 1+2     — warm-up 발동: {n_warm_p12}회")
    if n_warm_p12 > 0:
        print(f"  warm-up 호기 분포 (P1+2):")
        print(log_p12[log_p12.warm_up_unit != ''].warm_up_unit.value_counts().to_string())

    print(f"\n  Layer A startup count (P1):   {log_p1.startup_count.sum()}")
    print(f"  Layer A startup count (P1+2): {log_p12.startup_count.sum()}")

    # === Unit breakdown ===
    print("\n" + "=" * 70)
    print("Unit-level cumulative ΔP (MWh, P1+2)")
    print("=" * 70)
    ub = unit_breakdown(log_p12)
    for u in sorted(ub, key=lambda x: -ub[x]):
        print(f"  {u}:  {ub[u]:8.1f} MWh   (priority {PRIORITY[u]})")

    out = ROOT / "pv/experiments/thermal_planner_v4"
    out.mkdir(parents=True, exist_ok=True)
    log_p1.to_parquet(out / "log_phase1.parquet", index=False)
    log_p12.to_parquet(out / "log_phase1plus2.parquet", index=False)

    # Summary CSV (dashboard load_planner_summary 호환)
    summary = pd.DataFrame([
        {'label': 'Phase 1 only (reactive)',  **k1},
        {'label': 'Phase 1+2 (predictive)',   **k12},
    ])
    summary.to_csv(out / "summary.csv", index=False)
    print(f"\n저장: {out}/")


if __name__ == "__main__":
    main()
