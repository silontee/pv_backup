"""Forecast Replay — Phase 1 D-1 baseline vs Phase 2 reissued forecast vs actual.

issue time selector를 움직이면 그 시점의 Phase 2 reissued forecast가 표시.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data_loader import (
    load_phase1, load_phase2, portfolio_phase1, portfolio_phase2_at_issue,
    find_normal_day, PRESET_DAYS, EVENT_DAYS,
)

st.set_page_config(page_title="Forecast Replay", page_icon="📊", layout="wide")
# sidebar 에서 'app' 항목 숨김
st.markdown("""
<style>
[data-testid="stSidebarNav"] ul li:first-child {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Forecast Replay — Phase 1 vs Phase 2 vs Actual")

p1_raw = load_phase1()
p2_raw = load_phase2()
port_p1 = portfolio_phase1(p1_raw)

ALL_SITES = ['경상대','고흥만수상','광양항세방','구미','삼천포','영흥','예천','창원']

# === Sidebar ===
st.sidebar.header("📅 날짜")
preset_keys = ["— 안정적 평일 —"] + list(PRESET_DAYS.keys()) + \
              ["— 큰 변동일 —"] + list(EVENT_DAYS.keys()) + \
              ["📋 직접 선택"]
choice = st.sidebar.selectbox("날짜 선택", preset_keys, index=1)

if choice == "📋 직접 선택":
    available_dates = sorted(p2_raw['date'].dt.strftime("%Y-%m-%d").unique())
    date_str = st.sidebar.selectbox("날짜 선택", available_dates, index=0)
elif choice in PRESET_DAYS:
    date_str = PRESET_DAYS[choice]
elif choice in EVENT_DAYS:
    date_str = EVENT_DAYS[choice]
else:
    # 헤더 라인 선택된 경우 → 첫 preset으로 fallback
    date_str = list(PRESET_DAYS.values())[0]
date_sel = pd.Timestamp(date_str)

st.sidebar.divider()
st.sidebar.header("🏭 범위")
scope = st.sidebar.selectbox("발전소", ["Portfolio (전체 8 사이트)"] + ALL_SITES, index=0)

st.sidebar.divider()
st.sidebar.header("⏰ Issue time")
issue_hour = st.sidebar.slider("issue hour", 7, 18, 12)
st.sidebar.caption(f"이 시각 t={issue_hour}에 발행된 Phase 2 forecast가 t+1, t+2, t+3에 적용됩니다. "
                   f"(7~8시는 일출 후 PV가 시작된 날만 forecast 생성)")

scope_label = "전체 포트폴리오" if scope.startswith("Portfolio") else scope
st.markdown(f"### 📅 {date_str} ・ 🏭 {scope_label} ・ ⏰ {issue_hour}:00 발행 시점")
st.caption("🔵 현재 발행 Phase 2 (issue 시점부터 일몰까지 reforecast) ・ 🟠 직전 발행 Phase 2 (1시간 전, 점선)")

# === main chart 데이터 준비 (portfolio or single site) ===
def site_phase1_day(df_raw, site, date):
    """단일 site의 cf×cap=MW로 변환된 일별 시계열."""
    sub = df_raw[(df_raw.site == site) & (df_raw.date == date)].sort_values('hour').copy()
    sub['pv_actual_mw'] = sub['cf'] * sub['site_capacity_kw'] / 1000
    sub['pv_p1_mw']     = sub['mu_mean'] * sub['site_capacity_kw'] / 1000
    sub['pv_sigma_mw']  = sub['sigma_total'] * sub['site_capacity_kw'] / 1000
    return sub[['datetime_kst','hour','pv_p1_mw','pv_actual_mw','pv_sigma_mw']]


def site_phase2_at_issue(df_raw, site, date, issue_hour):
    """단일 site의 Phase 2 forecast at issue. cf 단위 → MW."""
    sub = df_raw[(df_raw.site == site) & (df_raw.date == date) &
                  (df_raw.issue_hour == issue_hour)].copy()
    if len(sub) == 0:
        return pd.DataFrame(columns=['target_dt','lead','pv_p1_mw','pv_p2_mw'])
    sub['pv_p1_mw'] = sub['mu_phase1'] * sub['cap'] / 1000
    sub['pv_p2_mw'] = sub['mu_phase2'] * sub['cap'] / 1000
    return sub[['target_dt','lead','pv_p1_mw','pv_p2_mw']].sort_values('lead')


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

    # missing (target_dt, site) → Phase 1 baseline에서 가져오기
    p1_use = p1_raw[(p1_raw.date == date) & (p1_raw.datetime_kst.isin(target_dts))].copy()
    p1_use['p1_kw'] = p1_use['mu_mean'] * p1_use['site_capacity_kw']
    rows_extra = []
    for tdt in target_dts:
        for site in ALL_SITES:
            if (tdt, site) not in p2_keys:
                p1_row = p1_use[(p1_use.datetime_kst == tdt) & (p1_use.site == site)]
                if len(p1_row) == 0: continue
                p1_kw = float(p1_row.iloc[0]['p1_kw'])
                lead = int((pd.Timestamp(tdt).hour - issue_hour))
                rows_extra.append({'target_dt': tdt, 'site': site, 'lead': lead,
                                    'p2_kw': p1_kw, 'p1_kw': p1_kw})  # fallback = P1
    if rows_extra:
        extra_df = pd.DataFrame(rows_extra)
        all_df = pd.concat([sub_p2[['target_dt','site','lead','p2_kw','p1_kw']], extra_df], ignore_index=True)
    else:
        all_df = sub_p2[['target_dt','site','lead','p2_kw','p1_kw']]

    g = all_df.groupby(['target_dt','lead'], as_index=False).agg(p2=('p2_kw','sum'), p1=('p1_kw','sum'))
    g['pv_p2_mw'] = g['p2'] / 1000
    g['pv_p1_mw'] = g['p1'] / 1000
    return g[['target_dt','lead','pv_p1_mw','pv_p2_mw']].sort_values(['target_dt','lead'])


if scope.startswith("Portfolio"):
    day_p1 = port_p1[port_p1.date == date_sel].sort_values('hour')
    p2_at_issue = portfolio_phase2_at_issue_with_fallback(p2_raw, p1_raw, date_sel, issue_hour)
    p2_prev = portfolio_phase2_at_issue_with_fallback(p2_raw, p1_raw, date_sel, issue_hour - 1) \
              if issue_hour > 7 else pd.DataFrame(columns=['target_dt','lead','pv_p1_mw','pv_p2_mw'])
else:
    day_p1 = site_phase1_day(p1_raw, scope, date_sel)
    p2_at_issue = site_phase2_at_issue(p2_raw, scope, date_sel, issue_hour)
    p2_prev = site_phase2_at_issue(p2_raw, scope, date_sel, issue_hour - 1) if issue_hour > 7 else \
              pd.DataFrame(columns=['target_dt','lead','pv_p1_mw','pv_p2_mw'])

if len(day_p1) == 0:
    st.warning(f"{date_str}에 Phase 1 데이터 없음.")
    st.stop()

fig = go.Figure()
# σ band (먼저 그림 → 뒤로 깔림)
fig.add_trace(go.Scatter(
    x=day_p1['datetime_kst'], y=day_p1['pv_p1_mw'] + 1.282*day_p1['pv_sigma_mw'],
    mode='lines', name='Phase 1 ±80%', line=dict(color='gray', width=1, dash='dot'),
    showlegend=True
))
fig.add_trace(go.Scatter(
    x=day_p1['datetime_kst'], y=(day_p1['pv_p1_mw'] - 1.282*day_p1['pv_sigma_mw']).clip(lower=0),
    mode='lines', name='Phase 1 -80%', line=dict(color='gray', width=1, dash='dot'),
    showlegend=False, fill='tonexty', fillcolor='rgba(150,150,150,0.15)'
))
# Phase 1 line
fig.add_trace(go.Scatter(
    x=day_p1['datetime_kst'], y=day_p1['pv_p1_mw'],
    mode='lines+markers', name='Phase 1 (D-1 baseline)',
    line=dict(color='#888888', width=2, dash='solid'), marker=dict(size=6)
))
# Phase 2 prior issue — issue_hour-1에서 발행된 forecast
# x축 -12분 jitter로 현재 발행 markers와 겹침 방지, 색은 주황(현재 파랑과 구별)
if len(p2_prev) > 0:
    prev_x = p2_prev['target_dt'] - pd.Timedelta(minutes=2)
    fig.add_trace(go.Scatter(
        x=prev_x, y=p2_prev['pv_p2_mw'],
        mode='lines+markers',
        name=f'Phase 2 직전 발행 @ {issue_hour-1}:00',
        line=dict(color='#ff7f0e', width=2.5, dash='dash'),
        marker=dict(size=10, symbol='circle', color='#ff7f0e',
                    line=dict(width=1, color='white')),
    ))

# Phase 2 reissued forecast (current issue)
if len(p2_at_issue) > 0:
    fig.add_trace(go.Scatter(
        x=p2_at_issue['target_dt'], y=p2_at_issue['pv_p2_mw'],
        mode='lines+markers', name=f'Phase 2 현재 발행 @ {issue_hour}:00',
        line=dict(color='royalblue', width=3),
        marker=dict(size=10, symbol='diamond'),
    ))
# actual 마지막 (가장 위에) — 발행 시점까지만 (운영 시점 시뮬레이션)
day_p1_realized = day_p1[day_p1['hour'] <= issue_hour] if 'hour' in day_p1.columns else day_p1
fig.add_trace(go.Scatter(
    x=day_p1_realized['datetime_kst'], y=day_p1_realized['pv_actual_mw'],
    mode='lines+markers', name=f'실측 (~{issue_hour}:00)',
    line=dict(color='black', width=3),
    marker=dict(size=9, color='black', line=dict(width=1, color='white'))
))
if len(p2_at_issue) > 0:
    # vertical line at issue time (annotation 분리 — Timestamp + annotation 호환 문제 회피)
    issue_ts = pd.Timestamp(f"{date_str} {issue_hour}:00:00")
    fig.add_shape(type="line", xref="x", yref="paper",
                   x0=issue_ts, x1=issue_ts, y0=0, y1=1,
                   line=dict(color="royalblue", width=2, dash="dash"), opacity=0.5)
    fig.add_annotation(x=issue_ts, y=1.02, xref="x", yref="paper",
                        text=f"issue {issue_hour}:00",
                        showarrow=False, font=dict(color="royalblue"))
else:
    st.info(f"{date_str} {issue_hour}:00에 Phase 2 reissue 없음 (해당 issue time daytime/data 부족).")

fig.update_layout(
    template='plotly_white',
    height=480,
    xaxis_title="시간 (KST)", yaxis_title=f"{scope_label} PV (MW)",
    hovermode='x unified',
    legend=dict(orientation='h', y=-0.15),
    margin=dict(t=20, b=20),
    plot_bgcolor='white', paper_bgcolor='white',
)
st.plotly_chart(fig, use_container_width=True)

# === 추가 차트: 매 시각 "실시간 forecasting 지표" (lead-1h rolling) vs 실측 ===
st.subheader("실시간 forecasting 지표")
st.caption("매 시각 가장 최근에 발행된 Phase 2 1시간-앞 예측을 이어 그린 운영용 forecast (일몰까지).")

# 실시간 forecasting 지표 (rolling lead-1h Phase 2) 생성 — scope에 따라
# 마지막 issue 이후 시간(t > max_issue+1)은 마지막 issue의 lead 2, 3로 채움
# Phase 2가 없는 site는 Phase 1 baseline로 fallback (issue=7 시 고흥만 등 일부 site 누락 보정)
def build_rolling_with_tail(scope_choice, p2_raw, p1_raw, date):
    """매 target hour h, 사이트별로:
       - Phase 2 lead=1 issue=h-1 있으면 사용
       - 없으면 마지막 issue의 lead=(h - last_issue) 사용 (tail 채움)
       - 그래도 없으면 Phase 1 baseline 사용 (fallback — site 일부 missing 보정)
       그 다음 portfolio = sum across 8 sites (cap-weighted, kW → MW).
    """
    p2_day = p2_raw[p2_raw.date == date].copy()
    p1_day = p1_raw[p1_raw.date == date].copy()
    if not scope_choice.startswith("Portfolio"):
        p2_day = p2_day[p2_day.site == scope_choice]
        p1_day = p1_day[p1_day.site == scope_choice]
    if len(p1_day) == 0:
        return pd.DataFrame(columns=['target_dt','pv_p2_mw','source'])

    # Phase 1 baseline per (datetime_kst, site) → kW
    p1_day = p1_day.copy()
    p1_day['p1_kw'] = p1_day['mu_mean'] * p1_day['site_capacity_kw']
    p1_day['p1_kw_per_site'] = p1_day['p1_kw']
    p1_lookup = p1_day.set_index(['datetime_kst','site'])['p1_kw'].to_dict()
    cap_lookup = p1_day.set_index(['datetime_kst','site'])['site_capacity_kw'].to_dict()

    # build per-(target_dt, site) best Phase 2 (lead=1 first, then last issue lead 2/3)
    if len(p2_day) > 0:
        max_iss = int(p2_day.issue_hour.max())
        p2_day['p2_kw'] = p2_day['mu_phase2'] * p2_day['cap']
        # priority: lead=1 (latest) > last issue lead=2 > last issue lead=3
        p2_day['priority'] = np.where(p2_day.lead == 1, 0,
                              np.where((p2_day.issue_hour == max_iss) & (p2_day.lead == 2), 1,
                              np.where((p2_day.issue_hour == max_iss) & (p2_day.lead == 3), 2, 99)))
        p2_day = p2_day.sort_values('priority')
        p2_best = p2_day.drop_duplicates(subset=['target_dt','site'], keep='first')
        p2_lookup = p2_best.set_index(['target_dt','site'])['p2_kw'].to_dict()
        priority_lookup = p2_best.set_index(['target_dt','site'])['priority'].to_dict()
    else:
        p2_lookup = {}; priority_lookup = {}

    # build target_dt × site grid using Phase 1's union of (datetime, site)
    rows = []
    for (dt, site), p1_kw in p1_lookup.items():
        cap_v = cap_lookup.get((dt, site), 0)
        if (dt, site) in p2_lookup:
            kw = p2_lookup[(dt, site)]
            pri = priority_lookup[(dt, site)]
            src = 'lead1 (latest)' if pri == 0 else 'last issue lead 2/3'
        else:
            kw = p1_kw
            src = 'Phase 1 fallback (no P2)'
        rows.append({'target_dt': dt, 'site': site, 'kw': kw, 'cap': cap_v, 'source': src})
    df = pd.DataFrame(rows)

    # aggregate to portfolio (or single site)
    if scope_choice.startswith("Portfolio"):
        # source priority: lead1 > tail > fallback (most-frequent or first)
        g = df.groupby('target_dt', as_index=False).agg(
            pv_p2_kw=('kw','sum'),
            n_p2=('source', lambda s: (s != 'Phase 1 fallback (no P2)').sum()),
            sources=('source', lambda s: s.value_counts().index[0]),
        ).rename(columns={'sources':'source'})
    else:
        g = df.rename(columns={'kw':'pv_p2_kw'})[['target_dt','pv_p2_kw','source']]
        g['n_p2'] = (g['source'] != 'Phase 1 fallback (no P2)').astype(int)
    g['pv_p2_mw'] = g['pv_p2_kw'] / 1000
    # 제거: Phase 2가 전혀 없는 hour (전부 P1 fallback) → 의미 없는 시작점 제거
    g = g[g['n_p2'] > 0]
    return g.sort_values('target_dt')[['target_dt','pv_p2_mw','source']]


roll_p2 = build_rolling_with_tail(scope, p2_raw, p1_raw, date_sel)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=day_p1.datetime_kst, y=day_p1.pv_p1_mw,
    mode='lines', name='Phase 1 (D-1 baseline)',
    line=dict(color='#888888', width=2)
))
if len(roll_p2) > 0:
    fig2.add_trace(go.Scatter(
        x=roll_p2.target_dt, y=roll_p2.pv_p2_mw,
        mode='lines+markers',
        name='Phase 2 rolling (latest, lead 1h)',
        line=dict(color='royalblue', width=3),
        marker=dict(size=9, symbol='diamond'),
    ))
fig2.add_trace(go.Scatter(
    x=day_p1.datetime_kst, y=day_p1.pv_actual_mw,
    mode='lines+markers', name='실측 (full day)',
    line=dict(color='black', width=3),
    marker=dict(size=9, color='black', line=dict(width=1, color='white'))
))
fig2.update_layout(
    template='plotly_white', height=400,
    xaxis_title="시간 (KST)", yaxis_title=f"{scope_label} PV (MW)",
    hovermode='x unified',
    legend=dict(orientation='h', y=-0.15),
    margin=dict(t=20, b=20),
    plot_bgcolor='white', paper_bgcolor='white',
)
st.plotly_chart(fig2, use_container_width=True)

# === issue별 forecast 행렬 (small multiples느낌) ===
st.subheader(f"발행 시각별 재발행 예측 — 일몰까지 ({date_str})")
st.caption("같은 날에도 발행 시각이 달라지면 Phase 2가 *일몰까지의 곡선*을 어떻게 다시 그리는지.")

cols = st.columns(5)
issues_to_show = [7, 10, 13, 15, 17]
for col, ih in zip(cols, issues_to_show):
    if scope.startswith("Portfolio"):
        p2_ih = portfolio_phase2_at_issue_with_fallback(p2_raw, p1_raw, date_sel, ih)
    else:
        p2_ih = site_phase2_at_issue(p2_raw, scope, date_sel, ih)
    with col:
        st.markdown(f"**Issue {ih}:00**")
        if len(p2_ih) == 0:
            st.caption("(데이터 없음)")
            continue
        # subset Phase 1 around issue
        targets = p2_ih['target_dt'].tolist()
        sub_p1 = day_p1[day_p1['datetime_kst'].isin(targets)]
        figm = go.Figure()
        figm.add_trace(go.Scatter(x=sub_p1.datetime_kst, y=sub_p1.pv_actual_mw,
                                   name='actual', line=dict(color='black')))
        figm.add_trace(go.Scatter(x=sub_p1.datetime_kst, y=sub_p1.pv_p1_mw,
                                   name='Phase 1', line=dict(color='gray')))
        figm.add_trace(go.Scatter(x=p2_ih.target_dt, y=p2_ih.pv_p2_mw,
                                   name='Phase 2', line=dict(color='royalblue', width=3)))
        figm.update_layout(template='plotly_white', height=200,
                            margin=dict(t=10, b=20, l=10, r=10),
                            showlegend=False, xaxis_title=None, yaxis_title="MW",
                            plot_bgcolor='white', paper_bgcolor='white')
        st.plotly_chart(figm, use_container_width=True)

# === 메트릭 카드 ===
st.subheader(f"이 날 {scope_label} 예측 성능")
day_p1_dt = day_p1[day_p1['hour'].between(9, 17)] if 'hour' in day_p1.columns else day_p1
mae_p1 = (day_p1_dt['pv_p1_mw'] - day_p1_dt['pv_actual_mw']).abs().mean()

# Phase 2 — 이 날의 모든 (issue, lead) 풀링 NMAE (scope 반영)
p2_day = p2_raw[p2_raw.date == date_sel].copy()
if not scope.startswith("Portfolio"):
    p2_day = p2_day[p2_day.site == scope]
p2_day['err_p1'] = (p2_day.cf - p2_day.mu_phase1).abs() * p2_day.cap
p2_day['err_p2'] = (p2_day.cf - p2_day.mu_phase2).abs() * p2_day.cap

if len(p2_day) > 0:
    cap_sum = p2_day.cap.sum()
    nmae_p1 = p2_day.err_p1.sum() / cap_sum * 100
    nmae_p2 = p2_day.err_p2.sum() / cap_sum * 100
    a, b, c = st.columns(3)
    a.metric("Phase 1 NMAE (D-1 baseline)", f"{nmae_p1:.2f}%")
    b.metric("Phase 1+2 NMAE (Phase 2 반영)", f"{nmae_p2:.2f}%",
             delta=f"{nmae_p2-nmae_p1:+.2f}%p", delta_color="inverse")
    c.metric("시간별 평균오차 (MW)", f"{mae_p1:.2f} MW")
