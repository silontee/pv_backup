"""Gap & DEADBAND — instant_gap (raw) + DEADBAND band + effective_gap (LNG 책임) + forward gap."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data_loader import (
    load_phase1, load_phase2, load_planner,
    portfolio_phase1, portfolio_phase2_at_issue_with_fallback,
    DEADBAND_MW, PRESET_DAYS, EVENT_DAYS,
)

st.set_page_config(page_title="실시간 차이 분석", page_icon="📉", layout="wide")
# sidebar 에서 'app' 항목 숨김
st.markdown("""
<style>
[data-testid="stSidebarNav"] ul li:first-child {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("📉 실시간 차이 — 어디까지가 LNG 책임인가")

st.markdown(f"""
<div style='font-size:1.05rem; line-height:1.7;'>
<b>실시간 차이</b> — PV 변동에 따른 원본 차이.<br>
<b>자체 흡수폭 ±{DEADBAND_MW:.0f} MW</b> — 계통 1차/2차 예비력·AGC가 자체 흡수한다고 판단하여 설정.<br>
<b>LNG 대응 차이</b> — LNG가 실제 책임지는 부분.<br>
<b>향후 예상 차이</b> — Phase 2 갱신값 기반 forward signal.<br>
<b>운영 모드</b> — forward signal 기반 호기 응답 상태.
</div>
""", unsafe_allow_html=True)

p1 = load_phase1()
p2 = load_phase2()
log_p2 = load_planner('phase1plus2')
port_p1 = portfolio_phase1(p1)

# Sidebar
st.sidebar.header("📅 날짜")
preset_keys = ["— 안정적 평일 —"] + list(PRESET_DAYS.keys()) + \
              ["— 큰 변동일 —"] + list(EVENT_DAYS.keys()) + \
              ["📋 직접 선택"]
choice = st.sidebar.selectbox("날짜 선택", preset_keys, index=1)
if choice == "📋 직접 선택":
    avail = sorted(p2['date'].dt.strftime("%Y-%m-%d").unique())
    date_str = st.sidebar.selectbox("날짜", avail)
elif choice in PRESET_DAYS:
    date_str = PRESET_DAYS[choice]
elif choice in EVENT_DAYS:
    date_str = EVENT_DAYS[choice]
else:
    date_str = list(PRESET_DAYS.values())[0]
date_sel = pd.Timestamp(date_str)

st.sidebar.divider()
st.sidebar.header("⏰ 현 시각")
issue_hour = st.sidebar.slider("Phase 2 발행 시각 (h)", 7, 17, 12)

st.markdown(f"### 📅 {date_str} ・ ⏰ 현 시각 {issue_hour}:00")

# === 데이터 ===
day_p1 = port_p1[port_p1.date == date_sel].sort_values('hour')
day_p1['instant_gap'] = day_p1.pv_p1_mw - day_p1.pv_actual_mw
day_p1['effective_gap'] = (day_p1.instant_gap - DEADBAND_MW).clip(lower=0)
day_p1_realized = day_p1[day_p1['hour'] <= issue_hour]

p2_at_issue = portfolio_phase2_at_issue_with_fallback(p2, p1, date_sel, issue_hour)
if len(p2_at_issue) > 0:
    p2_at_issue['forward_gap_mw'] = p2_at_issue.pv_p1_mw - p2_at_issue.pv_p2_mw

# planner log (state, FG1, slope)
day_log = log_p2[log_p2.date == date_sel].sort_values('hour')

# === ① Portfolio PV (standalone) ===
st.subheader("① 포트폴리오 PV — Phase 1 (D-1 예측) / Phase 2 (현 시각 갱신) / 실측")

fig_pv = go.Figure()
# σ band (gray dotted, 위아래)
fig_pv.add_trace(go.Scatter(
    x=day_p1.datetime_kst, y=day_p1.pv_p1_mw + 1.282*day_p1.pv_sigma_mw,
    mode='lines', name='Phase 1 ±80% 신뢰구간', line=dict(color='gray', width=1, dash='dot'),
    showlegend=True
))
fig_pv.add_trace(go.Scatter(
    x=day_p1.datetime_kst, y=(day_p1.pv_p1_mw - 1.282*day_p1.pv_sigma_mw).clip(lower=0),
    mode='lines', name='Phase 1 -80%', line=dict(color='gray', width=1, dash='dot'),
    showlegend=False, fill='tonexty', fillcolor='rgba(150,150,150,0.15)',
))
# Phase 1
fig_pv.add_trace(go.Scatter(
    x=day_p1.datetime_kst, y=day_p1.pv_p1_mw,
    mode='lines+markers', name='Phase 1 (D-1 예측)',
    line=dict(color='#888888', width=2), marker=dict(size=6),
))
# Phase 2 현재 발행 (파랑 다이아몬드) — 직전 발행(주황)은 표시 안 함
if len(p2_at_issue) > 0:
    fig_pv.add_trace(go.Scatter(
        x=p2_at_issue['target_dt'], y=p2_at_issue['pv_p2_mw'],
        mode='lines+markers', name=f'Phase 2 (현 시각 {issue_hour}:00 갱신)',
        line=dict(color='royalblue', width=3),
        marker=dict(size=10, symbol='diamond'),
    ))
# 실측 (검정, 가장 위)
fig_pv.add_trace(go.Scatter(
    x=day_p1_realized.datetime_kst, y=day_p1_realized.pv_actual_mw,
    mode='lines+markers', name=f'실측 (~{issue_hour}:00)',
    line=dict(color='black', width=3),
    marker=dict(size=9, color='black', line=dict(width=1, color='white')),
))
# issue time vline
issue_ts = pd.Timestamp(f"{date_str} {issue_hour}:00:00")
fig_pv.add_shape(type="line", xref="x", yref="paper",
                  x0=issue_ts, x1=issue_ts, y0=0, y1=1,
                  line=dict(color="royalblue", width=2, dash="dash"), opacity=0.5)
fig_pv.add_annotation(x=issue_ts, y=1.02, xref="x", yref="paper",
                       text=f"현 시각 {issue_hour}:00", showarrow=False,
                       font=dict(color="royalblue"))

fig_pv.update_layout(
    template='plotly_white', height=480,
    xaxis_title="시간 (KST)", yaxis_title="포트폴리오 PV (MW)",
    hovermode='x unified',
    legend=dict(orientation='h', y=-0.15),
    margin=dict(t=20, b=20),
    plot_bgcolor='white', paper_bgcolor='white',
)
st.plotly_chart(fig_pv, use_container_width=True)

# === 아래 3-row chart: 차이 흐름 ===
st.subheader("② 실시간 차이 → ③ LNG 대응 차이 → ④ 향후 예상 차이 + 운영 모드")

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=(
        f"<b>② 실시간 차이</b> — 자체 흡수폭 ±{DEADBAND_MW:.0f} MW",
        "<b>③ LNG 대응 차이</b> — 자체 흡수폭 차감 후 양수 부분만",
        "<b>④ 향후 예상 차이</b> — 일몰까지 Phase 2 갱신값 + 운영 모드",
    ),
    row_heights=[0.33, 0.33, 0.34], vertical_spacing=0.10,
)

# Row 1: instant_gap + DEADBAND band
fig.add_hrect(y0=-DEADBAND_MW, y1=DEADBAND_MW,
              fillcolor='lightgray', opacity=0.30, line_width=0,
              annotation_text=f'자체 흡수폭 ±{DEADBAND_MW:.0f} MW',
              annotation_position='top right',
              annotation_font_color='dimgray', annotation_font_size=11,
              row=1, col=1)
ig_colors = ['crimson' if v > DEADBAND_MW else
             ('royalblue' if v < -DEADBAND_MW else '#9E9E9E')
             for v in day_p1_realized.instant_gap]
fig.add_trace(go.Bar(x=day_p1_realized.datetime_kst, y=day_p1_realized.instant_gap,
                      name='실시간 차이', marker_color=ig_colors,
                      text=[f'{v:+.1f}' for v in day_p1_realized.instant_gap],
                      textposition='outside', cliponaxis=False,
                      showlegend=False), row=1, col=1)
fig.add_hline(y=0, line=dict(color='black', width=0.5), row=1, col=1)
fig.add_hline(y=DEADBAND_MW, line=dict(color='red', width=1, dash='dash'),
              opacity=0.4, row=1, col=1)
fig.add_hline(y=-DEADBAND_MW, line=dict(color='blue', width=1, dash='dash'),
              opacity=0.4, row=1, col=1)

# Row 2: effective_gap (LNG 책임분)
eg_colors = ['crimson' if v > 0 else '#E0E0E0' for v in day_p1_realized.effective_gap]
fig.add_trace(go.Bar(x=day_p1_realized.datetime_kst, y=day_p1_realized.effective_gap,
                      name='LNG 대응 차이', marker_color=eg_colors,
                      text=[f'{v:.1f}' if v > 0 else '' for v in day_p1_realized.effective_gap],
                      textposition='outside', cliponaxis=False,
                      showlegend=False), row=2, col=1)
fig.add_hline(y=0, line=dict(color='black', width=0.5), row=2, col=1)

# Row 3: forward gap + state
if len(p2_at_issue) > 0:
    fwd_colors = ['crimson' if v > 0 else 'royalblue' for v in p2_at_issue.forward_gap_mw]
    fig.add_trace(go.Bar(x=p2_at_issue.target_dt, y=p2_at_issue.forward_gap_mw,
                          name='향후 예상 차이', marker_color=fwd_colors, opacity=0.7,
                          text=[f'{v:+.1f}' for v in p2_at_issue.forward_gap_mw],
                          textposition='outside', cliponaxis=False,
                          showlegend=False), row=3, col=1)
fig.add_hline(y=0, line=dict(color='black', width=0.5), row=3, col=1)

# state markers (planner log 기반)
state_label_kr = {'KEEP': '유지', 'INCREASE': '추가 대응', 'DELAYED_RELEASE': '점진 해제'}
state_color = {'KEEP': '#9E9E9E', 'INCREASE': '#D32F2F', 'DELAYED_RELEASE': '#1976D2'}
if len(day_log) > 0:
    for st_name, color in state_color.items():
        sub = day_log[day_log.state == st_name]
        if len(sub) > 0:
            kr = state_label_kr[st_name]
            fig.add_trace(go.Scatter(x=sub.datetime_kst, y=[0]*len(sub),
                                      mode='markers', name=f'운영 모드: {kr}',
                                      marker=dict(color=color, size=12, symbol='square',
                                                  line=dict(color='white', width=1)),
                                      hovertemplate=f'운영 모드: {kr}<extra></extra>'),
                          row=3, col=1)

# issue time vline (subplot domain 좌표 사용 → y축 강제 안 함)
for r in [1, 2, 3]:
    fig.add_vline(x=issue_ts, line=dict(color="#1976D2", width=1.5, dash="dash"),
                   opacity=0.4, row=r, col=1)

# y축은 plotly autorange 에 맡김 (rangemode 강제 X)
fig.update_yaxes(title_text="MW", row=1, col=1, autorange=True)
fig.update_yaxes(title_text="MW", row=2, col=1, autorange=True)
fig.update_yaxes(title_text="MW", row=3, col=1, autorange=True)
fig.update_xaxes(title_text="시간 (KST)", row=3, col=1)
fig.update_layout(template='plotly_white', height=720, hovermode='x unified',
                   margin=dict(t=60, b=30),
                   legend=dict(orientation='h', y=-0.06, font=dict(size=12)),
                   plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)

# === 범례 / 색 가이드 ===
st.markdown(f"""
<div style='font-size:0.95rem; line-height:1.8; margin-top:6px;'>
<b>색 가이드</b><br>
<span style='display:inline-block; width:14px; height:14px; background:#D32F2F; vertical-align:middle;'></span>
&nbsp;<b>빨강</b> — PV 부족 (LNG 추가 출력 필요)<br>
<span style='display:inline-block; width:14px; height:14px; background:#1976D2; vertical-align:middle;'></span>
&nbsp;<b>파랑</b> — PV 초과 (LNG 그대로 유지)<br>
<span style='display:inline-block; width:14px; height:14px; background:#9E9E9E; vertical-align:middle;'></span>
&nbsp;<b>회색</b> — 자체 흡수폭 이내 (계통 자체 흡수, LNG 무동원)<br>
<br>
<b>운영 모드</b> (향후 예상 차이 기반)<br>
<span style='display:inline-block; width:14px; height:14px; background:#9E9E9E; vertical-align:middle;'></span>
&nbsp;<b>유지</b> — 평상시, 호기 변경 없음<br>
<span style='display:inline-block; width:14px; height:14px; background:#D32F2F; vertical-align:middle;'></span>
&nbsp;<b>추가 대응</b> — 큰 변동 예상, 호기 추가 가동 또는 예열 검토<br>
<span style='display:inline-block; width:14px; height:14px; background:#1976D2; vertical-align:middle;'></span>
&nbsp;<b>점진 해제</b> — 회복 신호, 켜놓은 호기 천천히 해제
</div>
""", unsafe_allow_html=True)

# === 일별 KPI ===
st.divider()
st.subheader(f"📊 {date_str} 일별 요약")

if len(day_log) > 0:
    sum_instant = float(day_log.instant_gap_mw[day_log.instant_gap_mw > 0].sum())
    sum_effective = float(day_log.effective_gap_mw.sum())
    sum_alloc = float(day_log.total_alloc_mw.sum())
    sum_short = float(day_log.shortfall_mw.sum())
    sum_over = float(day_log.over_commit_mw.sum())
    absorbed = sum_instant - sum_effective

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("실시간 차이 합 (양수)", f"{sum_instant:.1f} MWh",
              help="원본 부족 시간 누적")
    c2.metric("계통 자체 흡수", f"{absorbed:.1f} MWh",
              help="자체 흡수폭 이내 부분")
    c3.metric("LNG 대응 차이 (책임분)", f"{sum_effective:.1f} MWh",
              help="LNG가 메워야 할 양")
    c4.metric("LNG 실제 응답", f"{sum_alloc:.1f} MWh",
              help="호기별 ΔP 합계")
    c5.metric("부족 / 과대보충",
              f"{sum_short:.1f} / {sum_over:.1f}",
              help="LNG 대응 차이 기준")

    # 시간별 표
    with st.expander("📋 시간별 상세 (현 시각까지)"):
        past = day_log[day_log.hour <= issue_hour].copy()
        # 운영 모드 한국어 매핑
        past['운영 모드'] = past['state'].map(
            {'KEEP': '유지', 'INCREASE': '추가 대응', 'DELAYED_RELEASE': '점진 해제'})
        show = past[['hour','instant_gap_mw','effective_gap_mw','total_alloc_mw',
                     'shortfall_mw','over_commit_mw',
                     '운영 모드','FG1','slope','warm_up_unit']].copy()
        show['hour'] = show['hour'].astype(int).astype(str) + ":00"
        show.columns = ['시각','실시간 차이','LNG 대응 차이','LNG 응답',
                        '부족','과대보충','운영 모드','향후 1h','변화율','예열 호기']
        st.dataframe(show.round(2), hide_index=True, use_container_width=True)
else:
    st.info("이 날 운영 log 없음.")
