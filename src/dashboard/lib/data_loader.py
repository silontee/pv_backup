"""Phase 1 + Phase 2 + balancing planner 결과 로딩.

각 함수는 streamlit cache_data로 한 번만 실행됨.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]

# === Spec 상수 (plan §4.5) ===
DEADBAND_MW = 4.0    # 계통 자체 흡수 cushion. plan §4.5 (보조서비스 정산 LNG ≈50% 담당 근거).
PV_PORTFOLIO_MW = 77.28   # 8 사이트 합계
LNG_FLEET_MW = 920        # 분당화력 10 호기 합계 (KOEN)
GT_UNITS = ['CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8']
ST_UNITS = ['CS1','CS2']

# 호기 색 컨벤션 (모든 차트 통일 — stacked / pie / bar 동일)
GT_COLORS = ['#FFC107', '#FFB300', '#FFA000', '#FF8F00',
             '#FF6F00', '#E65100', '#BF360C', '#8B0000']
ST_COLORS = ['#1976D2', '#0D47A1']
UNIT_COLORS = {**{u: GT_COLORS[i] for i, u in enumerate(GT_UNITS)},
               **{u: ST_COLORS[i] for i, u in enumerate(ST_UNITS)}}


@st.cache_data
def load_phase1():
    """Phase 1 5-seed ensemble (test 2025).
    Columns: datetime_kst, site, site_capacity_kw, cf, mu_mean, sigma_total
    """
    p = pd.read_parquet(ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/ensemble_test.parquet")
    p['datetime_kst'] = pd.to_datetime(p['datetime_kst'])
    p['date'] = p['datetime_kst'].dt.normalize()
    p['hour'] = p['datetime_kst'].dt.hour
    return p


@st.cache_data
def load_phase2():
    """Phase 2 final = 2-branch 일몰까지 truncated full-horizon (λ=2.0, T=1, τ=0, L_max=12)
    + outage override layer (recovery 0.10 / 1h)
    3-seed ensemble (test 2025).
    각 issue time t에서 t+1 ~ 일몰까지 reforecast, 단 outage 감지 시 mu_phase2 := 0 override.

    Columns: date, site, issue_hour, lead, target_dt, cf, cap, dc10Tca_target,
             mu_phase1, sigma_phase1, delta_mean, delta_std, gate_mean,
             mu_phase2 (raw), mu_p2_ovr (override 적용), blackout (0/1), sigma_total
    """
    p = pd.read_parquet(ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet")
    p['date'] = pd.to_datetime(p['date'])
    p['target_dt'] = pd.to_datetime(p['target_dt'])
    p['target_hour'] = p['target_dt'].dt.hour
    # mu_phase2 = override 적용본 (blackout 시 0). mu_phase2_raw = override 전 원본.
    return p


@st.cache_data
def load_planner(mode):
    """mode ∈ {'phase1', 'phase1plus2'}. Planner v4 (multi-unit fleet) hourly log.
    Columns include per-unit dP_CG1..dP_CS2 + correction (= total fleet alloc)."""
    fname = "log_phase1.parquet" if mode == 'phase1' else "log_phase1plus2.parquet"
    p = pd.read_parquet(ROOT / f"pv/experiments/thermal_planner_v4/{fname}")
    p['datetime_kst'] = pd.to_datetime(p['datetime_kst'])
    p['date'] = p['datetime_kst'].dt.normalize()
    return p


@st.cache_data
def load_planner_summary():
    return pd.read_csv(ROOT / "pv/experiments/thermal_planner_v4/summary.csv")


LNG_UNITS = ['CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8','CS1','CS2']

@st.cache_data
def load_unit_breakdown(mode):
    """호기별 누적 ΔE (MWh)."""
    log = load_planner(mode)
    return {u: float(log[f'dP_{u}'].sum()) for u in LNG_UNITS if f'dP_{u}' in log.columns}


def portfolio_phase1(df_p1):
    """site sum → portfolio per datetime_kst. mu/actual in MW."""
    p = df_p1.copy()
    p['mu_kw'] = p['mu_mean'] * p['site_capacity_kw']
    p['act_kw'] = p['cf'] * p['site_capacity_kw']
    p['var_kw2'] = (p['sigma_total'] * p['site_capacity_kw']) ** 2
    g = p.groupby('datetime_kst', as_index=False).agg(
        pv_p1_kw=('mu_kw','sum'), pv_actual_kw=('act_kw','sum'),
        pv_var_kw2=('var_kw2','sum'), cap=('site_capacity_kw','sum'))
    g['pv_p1_mw'] = g['pv_p1_kw'] / 1000
    g['pv_actual_mw'] = g['pv_actual_kw'] / 1000
    g['pv_sigma_mw'] = np.sqrt(g['pv_var_kw2']) / 1000
    g['hour'] = g['datetime_kst'].dt.hour
    g['date'] = g['datetime_kst'].dt.normalize()
    return g[['datetime_kst','date','hour','pv_p1_mw','pv_actual_mw','pv_sigma_mw','cap']]


def portfolio_phase2_at_issue(df_p2, date, issue_hour):
    """(date, issue_hour) → portfolio-level Phase 2 forecast for lead 1~3 (no fallback)."""
    p = df_p2.copy()
    p['mu_p2_kw'] = p['mu_phase2'] * p['cap']
    p['mu_p1_kw'] = p['mu_phase1'] * p['cap']
    sub = p[(p.date == date) & (p.issue_hour == issue_hour)]
    g = sub.groupby(['target_dt','lead'], as_index=False).agg(
        pv_p2_kw=('mu_p2_kw','sum'), pv_p1_kw=('mu_p1_kw','sum'))
    g['pv_p2_mw'] = g['pv_p2_kw'] / 1000
    g['pv_p1_mw'] = g['pv_p1_kw'] / 1000
    return g[['target_dt','lead','pv_p1_mw','pv_p2_mw']]


ALL_SITES = ['경상대','고흥만수상','광양항세방','구미','삼천포','영흥','예천','창원']

def portfolio_phase2_at_issue_with_fallback(p2_raw, p1_raw, date, issue_hour):
    """Issue=h에서 Phase 2 sample이 없는 site는 Phase 1 baseline 사용해 portfolio 합산.
    → 항상 8 사이트 일관된 portfolio 값."""
    sub_p2 = p2_raw[(p2_raw.date == date) & (p2_raw.issue_hour == issue_hour)].copy()
    if len(sub_p2) == 0:
        return pd.DataFrame(columns=['target_dt','lead','pv_p1_mw','pv_p2_mw'])
    target_dts = sub_p2['target_dt'].unique()
    p2_keys = set(zip(sub_p2['target_dt'], sub_p2['site']))
    sub_p2['p2_kw'] = sub_p2['mu_phase2'] * sub_p2['cap']
    sub_p2['p1_kw'] = sub_p2['mu_phase1'] * sub_p2['cap']
    p1_use = p1_raw[(p1_raw.date == date) & (p1_raw.datetime_kst.isin(target_dts))].copy()
    p1_use['p1_kw'] = p1_use['mu_mean'] * p1_use['site_capacity_kw']
    rows_extra = []
    for tdt in target_dts:
        for site in ALL_SITES:
            if (tdt, site) not in p2_keys:
                p1_row = p1_use[(p1_use.datetime_kst == tdt) & (p1_use.site == site)]
                if len(p1_row) == 0: continue
                p1_kw = float(p1_row.iloc[0]['p1_kw'])
                lead = int(pd.Timestamp(tdt).hour - issue_hour)
                rows_extra.append({'target_dt': tdt, 'site': site, 'lead': lead,
                                    'p2_kw': p1_kw, 'p1_kw': p1_kw})
    if rows_extra:
        all_df = pd.concat([sub_p2[['target_dt','site','lead','p2_kw','p1_kw']],
                            pd.DataFrame(rows_extra)], ignore_index=True)
    else:
        all_df = sub_p2[['target_dt','site','lead','p2_kw','p1_kw']]
    g = all_df.groupby(['target_dt','lead'], as_index=False).agg(
        p2=('p2_kw','sum'), p1=('p1_kw','sum'))
    g['pv_p2_mw'] = g['p2'] / 1000
    g['pv_p1_mw'] = g['p1'] / 1000
    return g[['target_dt','lead','pv_p1_mw','pv_p2_mw']].sort_values(['target_dt','lead'])


def find_normal_day(df_p1):
    """평균 오차 가장 작고 PV 발전이 정상이었던 날 (Phase 1 stability 시연용)."""
    port = portfolio_phase1(df_p1)
    daytime = port[port.hour.between(9, 17)].copy()
    daytime['err'] = (daytime.pv_p1_mw - daytime.pv_actual_mw).abs()
    by_day = daytime.groupby('date').agg(mae_mw=('err','mean'), pv_max=('pv_actual_mw','max'))
    cand = by_day[by_day.pv_max > 30].sort_values('mae_mw')
    return cand.index[0]


# 대표 날짜 (Phase 1이 잘 맞는 시연용 — 모든 페이지 공용)
# 4계절 안정 발전일 (Phase 1 portfolio NMAE 1% 내외).
PRESET_DAYS = {
    "봄 — 2025-03-20":   "2025-03-20",   # NMAE 0.81%
    "여름 — 2025-08-22": "2025-08-22",   # NMAE 1.18%
    "가을 — 2025-10-20": "2025-10-20",   # NMAE 1.10%
    "겨울 — 2025-12-22": "2025-12-22",   # NMAE 1.08%
}
# Event days — Phase 2 가치 다양성 (3계절 + 강도 large/medium/extreme)
# 선정 기준: planner v4 일별 KPI 분석 (2026-05-09)
EVENT_DAYS = {
    "★ 2025-09-06": "2025-09-06",
    "2025-08-20":   "2025-08-20",
    "2025-03-23":   "2025-03-23",
}


def get_kpis():
    """legacy. summary CSV 반환."""
    return load_planner_summary()


# === v4 신규 헬퍼 (2026-05-09) ===

@st.cache_data
def load_planner_both():
    """P1, P1+2 두 모드 한 번에 반환."""
    return load_planner('phase1'), load_planner('phase1plus2')


def kpi_year(log):
    """1년 누적 핵심 KPI dict."""
    total_dE = sum(log[f'dP_{u}'].sum() for u in LNG_UNITS if f'dP_{u}' in log.columns)
    return dict(
        shortfall=float(log.shortfall_mw.sum()),
        over_commit=float(log.over_commit_mw.sum()),
        total_dE=float(total_dE),
        startup_count=int(log.startup_count.sum()) if 'startup_count' in log.columns else 0,
        warmup_count=int((log.warm_up_unit != '').sum()) if 'warm_up_unit' in log.columns else 0,
        state_keep=int((log.state == 'KEEP').sum()),
        state_increase=int((log.state == 'INCREASE').sum()),
        state_release=int((log.state == 'DELAYED_RELEASE').sum()),
        n_hours=len(log),
    )


def kpi_monthly(log):
    """월별 KPI 집계."""
    df = log.copy()
    df['month'] = df.datetime_kst.dt.to_period('M').astype(str)
    g = df.groupby('month').agg(
        shortfall=('shortfall_mw', 'sum'),
        over_commit=('over_commit_mw', 'sum'),
        instant_gap=('instant_gap_mw', lambda s: s[s>0].sum()),
        effective_gap=('effective_gap_mw', 'sum'),
    ).reset_index()
    # 호기별 ΔE
    for u in LNG_UNITS:
        col = f'dP_{u}'
        if col in df.columns:
            g[f'dE_{u}'] = df.groupby('month')[col].sum().values
    g['total_dE'] = g[[f'dE_{u}' for u in LNG_UNITS if f'dE_{u}' in g.columns]].sum(axis=1)
    return g


def warmup_distribution(log):
    """warm-up 호기별 발동 횟수."""
    if 'warm_up_unit' not in log.columns:
        return pd.DataFrame(columns=['unit','count'])
    s = log[log.warm_up_unit != '']['warm_up_unit'].value_counts().reset_index()
    s.columns = ['unit','count']
    s['type'] = s.unit.apply(lambda u: 'GT' if u.startswith('CG') else 'ST')
    return s


def daily_unit_dE(log, date_sel):
    """특정 날짜의 호기별 ΔE 누적 (시간별)."""
    day = log[log.date == date_sel].sort_values('hour').copy()
    if len(day) == 0:
        return None
    cols = [f'dP_{u}' for u in LNG_UNITS if f'dP_{u}' in day.columns]
    return day[['datetime_kst','hour','state','warm_up_unit','startup_count'] + cols]


def fleet_share_year(log):
    """호기별 1년 누적 ΔE + 비중."""
    rows = []
    total = 0.0
    for u in LNG_UNITS:
        col = f'dP_{u}'
        if col in log.columns:
            v = float(log[col].sum())
            rows.append({'unit': u, 'type': 'GT' if u.startswith('CG') else 'ST', 'dE_MWh': v})
            total += v
    df = pd.DataFrame(rows)
    df['share'] = df['dE_MWh'] / total if total > 0 else 0
    return df.sort_values('dE_MWh', ascending=False)


def monthly_unit_dE(log):
    """호기별 월별 ΔE (long format: month × unit × dE_MWh)."""
    df = log.copy()
    df['month'] = df.datetime_kst.dt.to_period('M').astype(str)
    rows = []
    for m, g in df.groupby('month'):
        for u in LNG_UNITS:
            col = f'dP_{u}'
            if col in g.columns:
                rows.append({'month': m, 'unit': u,
                              'type': 'GT' if u.startswith('CG') else 'ST',
                              'dE_MWh': float(g[col].sum())})
    return pd.DataFrame(rows)


def unit_year_stats(log):
    """호기별 1년 운전 통계 — 활성 시간, peak ΔP, avg ΔP (운전 중)."""
    rows = []
    for u in LNG_UNITS:
        col = f'dP_{u}'
        if col not in log.columns:
            continue
        s = log[col]
        active = s > 0.1
        rows.append({
            'unit': u, 'type': 'GT' if u.startswith('CG') else 'ST',
            'dE_MWh':   float(s.sum()),
            'peak_MW':  float(s.max()),
            'avg_MW (운전중)': float(s[active].mean()) if active.any() else 0.0,
            '활성_h':   int(active.sum()),
            '운전율_%': round(active.mean() * 100, 1),
        })
    return pd.DataFrame(rows).sort_values('dE_MWh', ascending=False)
