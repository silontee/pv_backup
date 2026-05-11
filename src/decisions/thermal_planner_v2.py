"""Thermal balancing planner v2 — short-horizon variability absorber.

핵심 재정의 (v1과 다른 점):
  - 하루 전체 base UC가 아니라, **1~3h variability balancing planner**
  - PV D-1 baseline 위에 balancing correction을 얹어 thermal action 결정
  - Phase 2 update를 그대로 추종하지 않음 — threshold + partial response + hysteresis
  - novelty는 stable balancing action에 있음 (action oscillation 최소화)

Gap 정의 (사용자 명세):
  A. realized_gap_t   = PV_D1_t-K..t-1의 (PV_D1 - PV_actual) 평균  (look-back)
  B. forward_3h_gap_t = max_{k=1..3} (PV_D1_{t+k} - PV_p2_{t+k})    (forward, Phase 2만)
  C. recommended balancing response = balancing rule(signal)         (action)

Balancing rule:
  signal = forward_3h_gap  (Case 2)  또는  realized_gap  (Case 1)
  if |signal| < THR_LOW:  keep, correction_target = 0
  elif |signal| < THR_HIGH: ramp_up/down, correction_target = ALPHA * signal
  else:                    start/standby, correction_target = signal

Hysteresis:
  correction_target < correction_prev → release_rate로 천천히 감소
  correction_target > correction_prev → ramp_rate 제약 내 증가

비교:
  Case 1: Phase 1 only — D-1 forecast 그대로 + look-back realized gap 보정
  Case 2: Phase 1+2    — D-1 forecast + Phase 2 forward gap 기반 선제 보정

KPI:
  shortfall, over_commit, peaker_starts, action_oscillation, correction_volatility,
  state_changes, top event response
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

# ===== stylized demand & thermal =====
# Thermal spec: 분당 LNG CS2 호기 (4년치 hourly 데이터 기반)
# - cf 0.298, running_pct 52%, daytime running 64%
# - pmax_observed 198 MW, pmin_p05 43 MW, ramp_p95 21 MW/h, ramp_max 179 MW/h
DEMAND_BASE = 80.0
DEMAND_AMP  = 25.0
THERMAL_MIN = 43.0       # CS2 pmin (must-run when running)
ONLINE_MAX  = 200.0      # CS2 pmax_observed (호기 1대로 PV 변동 충분히 cover)
PEAKER_MAX  = 0.0        # peaker 불필요 (CS2 alone covers PV swing)
THERMAL_MAX = ONLINE_MAX + PEAKER_MAX
RAMP_RATE   = 30.0       # CS2 hourly ramp typical (p95 21, max 179) — 보수 30 MW/h (legacy 호환)
RESERVE_K   = 1.282

# ===== balancing rule params (단순화 — 부분추종 제거) =====
# 룰: 매 시각 t에서
#   1) realized_gap = look-back 평균 (현재 시각의 진짜 부족분)
#   2) forward_gap  = 1시간 앞 Phase 2 예상 (Phase 1+2 모드에서만 사용)
#      → 사전 ramp 신호로 활용. 기본 신호는 realized.
#   3) signal = max-abs(realized, forward) ← Phase 1+2일 때만 forward 후보 추가
#   4) |signal| < THR_LOW → 잡음 무시, correction = 0
#      그 외 → correction = signal (100% 추종)
#   5) ramp/release 제약 적용
# (기존 ALPHA 부분추종, THR_HIGH peaker 영역 모두 제거)
THR_LOW       = 3.0      # MW: 잡음 무시 임계 (forward p75 / realized 중간 근처)
RAMP_UP_RATE  = 30.0     # MW/h: 출력 상승 한도 (CS2 ramp 보수)
RELEASE_RATE  = 4.0      # MW/h: 출력 하강 한도 (한 번 올린 출력 천천히 내림 = 진동 방지)
LOOKBACK_K    = 2        # realized gap look-back hours


def demand_profile(hour):
    return DEMAND_BASE + DEMAND_AMP * np.sin(np.pi * (hour - 7) / 12)


def load_inputs():
    p1 = pd.read_parquet(ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/ensemble_test.parquet")
    p1['datetime_kst'] = pd.to_datetime(p1['datetime_kst'])
    p1['mu_kw'] = p1['mu_mean'] * p1['site_capacity_kw']
    p1['act_kw'] = p1['cf'] * p1['site_capacity_kw']
    p1['var_kw2'] = (p1['sigma_total'] * p1['site_capacity_kw']) ** 2
    port1 = p1.groupby('datetime_kst', as_index=False).agg(
        pv_d1=('mu_kw','sum'), pv_actual=('act_kw','sum'),
        pv_var=('var_kw2','sum'), cap=('site_capacity_kw','sum'))
    port1['pv_d1_mw'] = port1['pv_d1'] / 1000
    port1['pv_actual_mw'] = port1['pv_actual'] / 1000
    port1['pv_sigma_mw'] = np.sqrt(port1['pv_var']) / 1000

    p2 = pd.read_parquet(ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet")
    p2['target_dt'] = pd.to_datetime(p2['target_dt'])
    p2['mu_p2_kw'] = p2['mu_phase2'] * p2['cap']
    # 각 (target_dt, lead) 별로 portfolio 합산
    p2_port = p2.groupby(['target_dt','lead'], as_index=False).agg(pv_p2=('mu_p2_kw','sum'))
    p2_port['pv_p2_mw'] = p2_port['pv_p2'] / 1000
    return port1[['datetime_kst','pv_d1_mw','pv_actual_mw','pv_sigma_mw']], p2_port[['target_dt','lead','pv_p2_mw']]


def build_day(date, port1_day, p2_day):
    """One day's hourly table.
       Columns: datetime_kst, hour, demand, pv_d1, pv_actual, sigma,
                pv_p2_lead1, pv_p2_lead2, pv_p2_lead3 (= forecast issued at t for t+1,t+2,t+3)
       The Phase 2 forecast for hour h that planner would actually use *at hour h-1*
       is the lead=1 from issue=h-1.
    """
    df = port1_day.copy().sort_values('datetime_kst').reset_index(drop=True)
    df['hour'] = df.datetime_kst.dt.hour
    df['demand_mw'] = df.hour.apply(demand_profile)
    # 매 시각 t (issue_time)에서 forecast를 t+1, t+2, t+3에 대해 갖음
    # planner가 시각 h에서 의사결정할 때 forward gap = PV_D1 - PV_p2 (lead=1,2,3 issued at h-1,...wait)
    # 가장 자연스러운 정의: 시각 t (decision time)에 forward_3h_gap_t = max_{k=1..3} (PV_D1_{t+k} - PV_p2_{t+k from issue=t})
    # 즉 issue_time=t, lead=1,2,3
    # We need: at decision time t, look up p2 forecast for (t+1, t+2, t+3) from issue_time=t
    # Since p2 stored as (target_dt, lead) where target_dt = issue + lead*1h,
    # for decision at t: target_dt={t+1,t+2,t+3}, lead=1,2,3 — matches when issue=t.
    # But the saved parquet has all (target, lead) combinations for all issues. We need to filter by "issue_time = decision_time".
    # 단순화: 각 hour h에 대해, p2 이용 forecast는 "issue=h, lead=1,2,3"의 (target_dt, lead) row → target_dt = h+1, h+2, h+3 with lead=1,2,3.
    # 그래서 join은 (h, h+1, h+2, h+3) 형태로 다양함. 더 깔끔: 같은 issue_hour 인 행들을 모아서 매 issue 별 forward forecast를 계산.

    # 위 단순화를 위해 다른 접근: Phase 2 ensemble parquet은 (date, site, issue_hour, lead, target_dt) 보유.
    # ensemble parquet은 portfolio aggregation 안 됨. 다시 site→portfolio 합산하면서 issue_hour 보존.
    return df


def planner_run(daily_inputs, mode, p2_full):
    """daily_inputs: list of (date, df_day with hourly portfolio rows).
       p2_full: full Phase 2 ensemble parquet (per site, issue_hour, lead, target_dt).
       mode: 'phase1' or 'phase1+2'.
    """
    rows = []
    p2_full = p2_full.copy()
    p2_full['target_dt'] = pd.to_datetime(p2_full['target_dt'])
    p2_full['date'] = p2_full['target_dt'].dt.normalize()
    # portfolio aggregate: sum over sites (per issue_hour, lead, target_dt)
    p2_port = p2_full.groupby(['date','issue_hour','lead','target_dt'], as_index=False).agg(
        pv_p2_kw=('mu_phase2','sum'),  # wait — need *cap weighted
    )
    # actually mu_phase2 is per-site cf, need × cap
    p2_full['mu_p2_kw'] = p2_full['mu_phase2'] * p2_full['cap']
    p2_port = p2_full.groupby(['date','issue_hour','lead','target_dt'], as_index=False).agg(
        pv_p2_mw=('mu_p2_kw', lambda s: s.sum()/1000.0))

    for date, df_day in daily_inputs:
        df_day = df_day.sort_values('datetime_kst').reset_index(drop=True)
        # state
        prev_correction = 0.0
        prev_state = 'keep'
        prev_peaker_on = 0
        # realized gap rolling (look-back of past actual vs D-1)
        gap_history = []  # list of (PV_D1 - PV_actual) per past hour

        for i, r in df_day.iterrows():
            h = r.hour
            pv_d1 = r.pv_d1_mw
            pv_actual = r.pv_actual_mw
            sigma = r.pv_sigma_mw

            # === 1) realized_gap = 현재까지 누적된 PV 부족분 (look-back avg) ===
            if len(gap_history) >= 1:
                realized_gap = float(np.mean(gap_history[-LOOKBACK_K:]))
            else:
                realized_gap = 0.0

            # === 2) forward_gap = 1h 앞 Phase 2 갱신 (Phase 1+2 모드에서만 사전 ramp 신호로 활용) ===
            if mode == 'phase1+2':
                p2_for_h = p2_port[(p2_port.date == r.datetime_kst.normalize()) &
                                    (p2_port.issue_hour == h)]
                if len(p2_for_h) > 0:
                    fwd_gaps = []
                    for _, r2 in p2_for_h.iterrows():
                        target_t = r2.target_dt
                        d1_target_row = df_day[df_day.datetime_kst == target_t]
                        if len(d1_target_row) == 0: continue
                        d1_val = d1_target_row.iloc[0].pv_d1_mw
                        fwd_gaps.append(d1_val - r2.pv_p2_mw)
                    forward_gap = max(fwd_gaps, key=abs) if fwd_gaps else 0.0
                else:
                    forward_gap = 0.0
                # signal = realized vs forward 중 큰 쪽 (사전 ramp 효과)
                signal = forward_gap if abs(forward_gap) > abs(realized_gap) else realized_gap
            else:
                forward_gap = float('nan')
                signal = realized_gap

            # === 3) 단순 룰: 잡음 컷 + 100% 추종 (부분추종 없음) ===
            if abs(signal) < THR_LOW:
                action_state = 'keep'
                correction_target = 0.0
            else:
                action_state = 'ramp_up' if signal > 0 else 'ramp_down'
                correction_target = signal

            # === 4) ramp/release 제약 (한 번 올린 출력 천천히 내림 = 진동 방지) ===
            if correction_target > prev_correction:
                correction = min(correction_target, prev_correction + RAMP_UP_RATE)
            else:
                correction = max(correction_target, prev_correction - RELEASE_RATE)

            # === 5) 평가: PV 부족분 = pv_d1 - pv_actual 을 correction이 얼마나 따라잡았나 ===
            instant_gap = pv_d1 - pv_actual                          # 즉시 부족분 (look-back 아님)
            shortfall    = max(0.0, instant_gap - correction)        # 부족분 중 못 채운 부분
            over_commit  = max(0.0, correction - instant_gap)        # 과대 보충

            rows.append({
                'datetime_kst': r.datetime_kst, 'date': r.datetime_kst.normalize(),
                'hour': h, 'mode': mode,
                'pv_d1_mw': pv_d1, 'pv_actual_mw': pv_actual, 'sigma_mw': sigma,
                'realized_gap': realized_gap, 'forward_gap': forward_gap, 'signal': signal,
                'instant_gap_mw': instant_gap,
                'correction_target': correction_target, 'correction': correction,
                'state': action_state,
                'reserve_adj_mw': RESERVE_K * sigma,
                'shortfall_mw': shortfall, 'over_commit_mw': over_commit,
                # legacy 호환 (대시보드/구버전 KPI)
                'demand_mw': 0.0, 'base_thermal_mw': 0.0,
                'online_mw': max(THERMAL_MIN, THERMAL_MIN + correction),
                'peaker_mw': 0.0, 'thermal_dispatch_mw': max(THERMAL_MIN, THERMAL_MIN + correction),
                'peaker_on': 0, 'peaker_start': 0,
                'residual_mw': instant_gap - correction,
            })
            prev_correction = correction
            prev_state = action_state
            prev_peaker_on = 0
            gap_history.append(pv_d1 - pv_actual)

    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Thermal balancing planner v2 — short-horizon variability absorber")
    print("=" * 70)
    print(f"  thermal: pmin={THERMAL_MIN}, pmax={ONLINE_MAX}")
    print(f"  rule (단순화): THR_LOW={THR_LOW} MW (잡음 컷), 100% 추종, RAMP={RAMP_UP_RATE}/RELEASE={RELEASE_RATE} MW/h")
    print()

    port1, _ = load_inputs()
    p2_full = pd.read_parquet(ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet")
    print(f"  Phase 1 portfolio rows: {len(port1):,}")
    print(f"  Phase 2 ensemble rows : {len(p2_full):,}")

    # filter daytime
    port1['hour'] = port1['datetime_kst'].dt.hour
    port1 = port1[port1.hour.between(9, 17)].copy()
    port1['date'] = port1['datetime_kst'].dt.normalize()
    daily_inputs = []
    for date, g in port1.groupby('date'):
        g['demand_mw'] = g['hour'].apply(demand_profile)
        daily_inputs.append((date, g))
    print(f"  daytime hours: {len(port1):,}, days: {len(daily_inputs)}")

    out_dir = ROOT / "pv/experiments/thermal_planner_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Phase 1 only — reactive balancing on look-back realized gap]")
    log1 = planner_run(daily_inputs, 'phase1', p2_full)
    print(f"  rows: {len(log1)}")
    log1.to_parquet(out_dir / "log_phase1.parquet", index=False)

    print("\n[Phase 1+2 — predictive balancing on forward gap]")
    log2 = planner_run(daily_inputs, 'phase1+2', p2_full)
    print(f"  rows: {len(log2)}")
    log2.to_parquet(out_dir / "log_phase1plus2.parquet", index=False)

    # KPIs
    def stability_metrics(df):
        n = len(df)
        # state changes (per day)
        sc = (df.state.values[1:] != df.state.values[:-1]).sum()
        # correction sign changes
        s = np.sign(df.correction.values)
        sgn = (s[1:] * s[:-1] < 0).sum()
        # correction step volatility (RMS of |Δcorrection|)
        dc = np.diff(df.correction.values)
        vol = np.sqrt((dc ** 2).mean()) if len(dc) > 0 else 0
        return sc, sgn, vol

    def kpi(log, label):
        sc, sgn, vol = stability_metrics(log)
        return {
            'label': label, 'n_hours': len(log),
            'shortfall_MWh': log.shortfall_mw.sum(),
            'over_commit_MWh': log.over_commit_mw.sum(),
            'shortage_hours': (log.shortfall_mw > 0.5).sum(),
            'over_commit_hours': (log.over_commit_mw > 0.5).sum(),
            'peaker_starts': log.peaker_start.sum(),
            'peaker_hours': log.peaker_on.sum(),
            'state_changes': sc,
            'correction_sign_flips': sgn,
            'correction_volatility_MW': vol,
            'mean_correction_abs': log.correction.abs().mean(),
        }

    k1, k2 = kpi(log1, 'Phase 1 only (reactive)'), kpi(log2, 'Phase 1+2 (predictive)')
    summary = pd.DataFrame([k1, k2])
    print("\n" + "=" * 70)
    print("Overall KPIs (test 2025, daytime 9~17)")
    print("=" * 70)
    print(summary.round(2).to_string(index=False))

    print("\nΔ (Phase 1+2 − Phase 1):")
    cols = ['shortfall_MWh','over_commit_MWh','shortage_hours','over_commit_hours',
             'peaker_starts','state_changes','correction_sign_flips','correction_volatility_MW',
             'mean_correction_abs']
    for col in cols:
        d = k2[col] - k1[col]
        pct = d / k1[col] * 100 if k1[col] != 0 else float('nan')
        print(f"  {col:<28} {k1[col]:>10.2f} → {k2[col]:>10.2f}   Δ {d:>+10.2f} ({pct:>+6.1f}%)")

    # Top events
    print("\n[Top cloud-pass events 정오~15시]")
    events = pd.to_datetime(['2025-03-23','2025-04-26','2025-05-04'])
    print(f"  {'date':<12} {'mode':<24} {'shortfall':>10} {'over_com':>9} {'peakers':>8} {'state_chg':>10} {'mean|corr|':>11}")
    for ev in events:
        for log, lbl in [(log1, 'phase1 reactive'), (log2, 'phase1+2 predictive')]:
            sub = log[(log.datetime_kst.dt.normalize() == ev) & log.hour.between(12, 15)]
            sc = (sub.state.values[1:] != sub.state.values[:-1]).sum() if len(sub) > 1 else 0
            print(f"  {str(ev.date()):<12} {lbl:<24} {sub.shortfall_mw.sum():>9.2f} "
                  f"{sub.over_commit_mw.sum():>8.2f} {int(sub.peaker_start.sum()):>8} {sc:>10} "
                  f"{sub.correction.abs().mean():>10.2f}")

    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
