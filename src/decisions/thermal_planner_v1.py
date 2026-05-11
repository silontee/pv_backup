"""Thermal planner v1 — Phase 1 only vs Phase 1+2 비교.

설계:
  - 입력: Phase 1 (mu, sigma) day-ahead forecast + Phase 2 intraday update +
          stylized demand profile + thermal unit constraints
  - 매 hour h (9~17), latest Phase 2 forecast for h = issue_hour=h-1, lead=1h
    (h=9는 update 없으니 Phase 1 그대로)
  - net_load_forecast_h = demand_h - pv_forecast_h  (portfolio MW)
  - reserve_h = reserve_factor * sigma_pv_h * portfolio_cap   (Cov80 buffer)
  - thermal_target_h = net_load + reserve
  - 적용: thermal_min ≤ dispatch ≤ thermal_max + ramp_rate 제약 (online), 초과 시 peaker startup

Phase 1 only vs Phase 1+2:
  - Phase 1 only: 모든 시간에 mu_phase1 사용 (intraday 갱신 없음)
  - Phase 1+2:    매 hour h에서 가장 신선한 Phase 2 forecast 사용 (issue=h-1, lead=1)

KPI:
  - shortfall_MWh   = sum max(0, demand - pv_actual - thermal_dispatched)
  - over_commit_MWh = sum max(0, thermal_dispatched - (demand - pv_actual))
  - peaker_starts   = peaker가 0→1 전이된 횟수
  - reserve_shortfall = shortfall에서 reserve_factor*sigma 가 *부족했던* 비율
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

# ===== stylized config =====
DEMAND_BASE = 80.0      # MW (낮 시간 baseload)
DEMAND_AMP  = 25.0      # MW (정오 피크 amplitude)
THERMAL_MIN = 30.0      # MW (online baseload, 항상 가동)
ONLINE_MAX  = 80.0      # MW (online thermal capacity, peaker 제외)
PEAKER_MAX  = 30.0      # MW (peaker capacity, cold-start 필요)
THERMAL_MAX = ONLINE_MAX + PEAKER_MAX   # 110 MW
RAMP_RATE   = 10.0      # MW/h (online ramp)
RESERVE_K   = 1.282     # Cov80 buffer (1.282 σ)
PEAKER_TRIGGER = 75.0   # net_load > 이 값이면 peaker 사전 가동
PEAKER_STARTUP_DELAY = 1  # hours (cold start)


def demand_profile(hour):
    """h ∈ [7,19] 부드러운 sin demand 곡선. 정오 14h 피크."""
    return DEMAND_BASE + DEMAND_AMP * np.sin(np.pi * (hour - 7) / 12)


def load_phase1():
    """Phase 1 ensemble parquet → portfolio (datetime_kst → mu, sigma_kw, actual)."""
    p = pd.read_parquet(ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/ensemble_test.parquet")
    p['datetime_kst'] = pd.to_datetime(p['datetime_kst'])
    # portfolio: site sum (mu*cap), sigma RSS
    p['mu_kw'] = p['mu_mean'] * p['site_capacity_kw']
    p['act_kw'] = p['cf'] * p['site_capacity_kw']
    p['var_kw2'] = (p['sigma_total'] * p['site_capacity_kw']) ** 2
    port = p.groupby('datetime_kst', as_index=False).agg(
        pv_p1=('mu_kw','sum'), pv_actual=('act_kw','sum'),
        pv_var=('var_kw2','sum'), cap=('site_capacity_kw','sum'))
    port['pv_p1_mw'] = port['pv_p1'] / 1000.0
    port['pv_actual_mw'] = port['pv_actual'] / 1000.0
    port['pv_sigma_mw'] = np.sqrt(port['pv_var']) / 1000.0
    return port[['datetime_kst','pv_p1_mw','pv_actual_mw','pv_sigma_mw','cap']]


def load_phase2():
    """Phase 2 ensemble (3-seed mean delta) → portfolio rolling forecast.
    For each target_dt, use latest issue_hour (largest), lead=1.
    """
    p = pd.read_parquet(ROOT / "pv/experiments/phase2_intraday_tcn/ensemble_test.parquet")
    p['target_dt'] = pd.to_datetime(p['target_dt'])
    # latest issue per target: lead=1 means issued at target-1h (most fresh)
    p1 = p[p.lead == 1].copy()
    # site-level mu_phase2 → portfolio
    p1['mu_p2_kw'] = p1['mu_phase2'] * p1['cap']
    port = p1.groupby('target_dt', as_index=False).agg(pv_p2=('mu_p2_kw','sum'))
    port['pv_p2_mw'] = port['pv_p2'] / 1000.0
    return port[['target_dt','pv_p2_mw']]


def planner_step(prev_dispatch, prev_peaker, net_load_target, sigma_mw):
    """rule-based: online ramp + reserve + peaker decision.
    Returns (dispatch_mw, peaker_on).
    """
    reserve = RESERVE_K * sigma_mw
    target = net_load_target + reserve
    # online: clamp by ramp from prev
    online_target = min(target, ONLINE_MAX)
    online = max(THERMAL_MIN, min(prev_dispatch + RAMP_RATE, online_target))
    online = max(prev_dispatch - RAMP_RATE, online)
    # peaker decision (rule)
    peaker_needed = target > ONLINE_MAX
    if peaker_needed:
        peaker = 1.0  # peaker dispatched (assume 1h startup OK in stylized)
        peaker_mw = min(PEAKER_MAX, target - online)
    else:
        peaker = 0.0
        peaker_mw = 0.0
    dispatch = online + peaker_mw
    return dispatch, online, peaker_mw, peaker


def run_day(day_df, mode):
    """day_df: hourly portfolio rows for one day, sorted by hour.
    mode ∈ {'phase1','phase1+2'}.
    """
    rows = []
    prev_disp = THERMAL_MIN; prev_peaker = 0.0
    for _, r in day_df.iterrows():
        h = r.hour
        demand_mw = demand_profile(h)
        if mode == 'phase1':
            pv_forecast = r.pv_p1_mw
        else:
            pv_forecast = r.pv_p2_mw if not pd.isna(r.pv_p2_mw) else r.pv_p1_mw
        sigma_mw = r.pv_sigma_mw
        net_load = demand_mw - pv_forecast
        dispatch, online, peaker_mw, peaker = planner_step(
            prev_disp, prev_peaker, net_load, sigma_mw)
        # actual residual
        residual = demand_mw - r.pv_actual_mw - dispatch
        rows.append({
            'datetime_kst': r.datetime_kst, 'hour': h, 'mode': mode,
            'demand_mw': demand_mw,
            'pv_forecast_mw': pv_forecast, 'pv_actual_mw': r.pv_actual_mw,
            'sigma_mw': sigma_mw, 'reserve_mw': RESERVE_K * sigma_mw,
            'thermal_dispatch_mw': dispatch, 'online_mw': online, 'peaker_mw': peaker_mw,
            'peaker_on': peaker, 'peaker_start': 1.0 if peaker > prev_peaker else 0.0,
            'residual_mw': residual,
            'shortfall_mw': max(0.0, residual),
            'over_commit_mw': max(0.0, -residual),
        })
        prev_disp = dispatch; prev_peaker = peaker
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("Thermal planner v1 — Phase 1 vs Phase 1+2 (stylized demand & thermal)")
    print("=" * 70)
    print(f"  demand: {DEMAND_BASE:.0f} + {DEMAND_AMP:.0f}·sin (peak ~{DEMAND_BASE+DEMAND_AMP:.0f}MW @ 14h)")
    print(f"  thermal: min {THERMAL_MIN}, online_max {ONLINE_MAX}, peaker_max {PEAKER_MAX}, "
          f"ramp {RAMP_RATE} MW/h, reserve {RESERVE_K}σ")
    print()

    p1 = load_phase1()
    p2 = load_phase2()
    print(f"  Phase 1 portfolio rows (test): {len(p1):,}")
    print(f"  Phase 2 lead=1 rows (test):    {len(p2):,}")

    # 통합: Phase 1 base + Phase 2 update만 있는 hour는 update 적용
    merged = p1.merge(p2.rename(columns={'target_dt':'datetime_kst'}),
                      on='datetime_kst', how='left')
    merged['hour'] = merged.datetime_kst.dt.hour
    merged['date'] = merged.datetime_kst.dt.normalize()
    # daytime only (9~17)
    merged = merged[merged.hour.between(9, 17)].copy()
    merged = merged.sort_values('datetime_kst').reset_index(drop=True)
    print(f"  daytime hours (9~17, with Phase 1 forecast): {len(merged):,}")
    print(f"  hours with Phase 2 update available:         {merged.pv_p2_mw.notna().sum():,}")
    print()

    out_dir = ROOT / "pv/experiments/thermal_planner_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running planner for each test day...")
    p1_runs, p2_runs = [], []
    for date, g in merged.groupby('date'):
        g = g.sort_values('hour').reset_index(drop=True)
        p1_runs.append(run_day(g, 'phase1'))
        p2_runs.append(run_day(g, 'phase1+2'))
    p1_log = pd.concat(p1_runs, ignore_index=True)
    p2_log = pd.concat(p2_runs, ignore_index=True)
    p1_log.to_parquet(out_dir / "dispatch_phase1.parquet", index=False)
    p2_log.to_parquet(out_dir / "dispatch_phase1plus2.parquet", index=False)

    def kpis(log, label):
        n = len(log)
        return {
            'label': label, 'n_hours': n,
            'shortfall_MWh': log.shortfall_mw.sum(),
            'over_commit_MWh': log.over_commit_mw.sum(),
            'shortage_hours': (log.shortfall_mw > 0.5).sum(),
            'over_commit_hours': (log.over_commit_mw > 0.5).sum(),
            'peaker_hours_on': (log.peaker_on > 0).sum(),
            'peaker_starts': log.peaker_start.sum(),
            'mean_residual': log.residual_mw.mean(),
            'std_residual': log.residual_mw.std(),
            'mean_dispatch': log.thermal_dispatch_mw.mean(),
        }

    k1 = kpis(p1_log, 'Phase 1 only')
    k2 = kpis(p2_log, 'Phase 1 + 2')
    summary = pd.DataFrame([k1, k2])
    print("=" * 70)
    print("Overall KPIs (test 2025, daytime 9~17)")
    print("=" * 70)
    print(summary.round(2).to_string(index=False))

    print()
    print("Δ (Phase 1+2 - Phase 1):")
    for col in ['shortfall_MWh','over_commit_MWh','shortage_hours','over_commit_hours',
                'peaker_starts','peaker_hours_on']:
        d = k2[col] - k1[col]
        pct = d / k1[col] * 100 if k1[col] != 0 else np.nan
        print(f"  {col:<22} {k1[col]:>10.2f} → {k2[col]:>10.2f}   Δ {d:>+10.2f} ({pct:>+6.1f}%)")

    # Top events
    print("\n[Top cloud-pass events 정오~15시]")
    events = pd.to_datetime(['2025-03-23','2025-04-26','2025-05-04'])
    print(f"  {'date':<12} {'mode':<13} {'shortfall':>11} {'over_commit':>12} {'peaker_h':>10} {'peaker_start':>13}")
    for ev in events:
        for log, lbl in [(p1_log, 'phase1'), (p2_log, 'phase1+2')]:
            sub = log[(log.datetime_kst.dt.normalize() == ev) & log.hour.between(12, 15)]
            print(f"  {str(ev.date()):<12} {lbl:<13} {sub.shortfall_mw.sum():>10.2f} "
                  f"{sub.over_commit_mw.sum():>11.2f} {(sub.peaker_on>0).sum():>10.0f} "
                  f"{sub.peaker_start.sum():>13.0f}")

    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
