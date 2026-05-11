"""Phase 2 Value — Phase 1 only vs Phase 1+2 정량 비교."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data_loader import (
    load_planner_both, kpi_year, kpi_monthly, warmup_distribution,
    DEADBAND_MW, EVENT_DAYS, LNG_UNITS,
)

st.set_page_config(page_title="Phase 2 가치", page_icon="📈", layout="wide")
# sidebar 에서 'app' 항목 숨김
st.markdown("""
<style>
[data-testid="stSidebarNav"] ul li:first-child {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Phase 2 가치 — 당일 갱신 예측의 정량 효과")

st.markdown(f"""
**핵심 비교**:
- **Phase 1 단독 (사후 대응)**: D-1 baseline 만 사용. 예열 명령 비활성. 이미 일어난 차이만 사후 대응.
- **Phase 1+2 (선제 대응)**: D-1 + 당일 갱신 예측. 향후 신호 기반 GT 예열 명령 활성.

두 모드 동일 운영 로직, 동일 호기, 동일 자체 흡수폭 {DEADBAND_MW:.0f} MW. 입력 신호만 다름.
""")

log_p1, log_p1p2 = load_planner_both()
k1 = kpi_year(log_p1)
k2 = kpi_year(log_p1p2)

# === KPI 비교 카드 ===
st.subheader("📊 1년 KPI 비교 (test 2025, daytime 9-17)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("부족분 (MWh)",
          f"{k2['shortfall']:.0f}",
          delta=f"{k2['shortfall']-k1['shortfall']:+.0f} ({(k2['shortfall']/k1['shortfall']-1)*100:+.1f}%)",
          delta_color="inverse",
          help=f"Phase 1 단독: {k1['shortfall']:.0f} → Phase 1+2: {k2['shortfall']:.0f}")
c2.metric("과대보충 (MWh)",
          f"{k2['over_commit']:.0f}",
          delta=f"{k2['over_commit']-k1['over_commit']:+.0f} ({(k2['over_commit']/k1['over_commit']-1)*100:+.1f}%)",
          delta_color="inverse",
          help=f"Phase 1 단독: {k1['over_commit']:.0f} → Phase 1+2: {k2['over_commit']:.0f}")
c3.metric("총 추가 발전 (MWh)",
          f"{k2['total_dE']:.0f}",
          delta=f"{k2['total_dE']-k1['total_dE']:+.0f}",
          help="LNG 호기들이 추가로 낸 발전량")
c4.metric("예열 명령 횟수",
          f"{k2['warmup_count']}회",
          delta=f"+{k2['warmup_count']-k1['warmup_count']} (Phase 1 단독=0)",
          help="GT 예열 명령 — 당일 갱신 예측 기반")

# === 비교 표 ===
st.markdown("##### 상세 비교")
comp = pd.DataFrame([
    {'지표': '부족분 (MWh)', 'Phase 1 단독': k1['shortfall'], 'Phase 1+2': k2['shortfall'],
     'Δ': k2['shortfall']-k1['shortfall'], '변화율': f"{(k2['shortfall']/k1['shortfall']-1)*100:+.1f}%"},
    {'지표': '과대보충 (MWh)', 'Phase 1 단독': k1['over_commit'], 'Phase 1+2': k2['over_commit'],
     'Δ': k2['over_commit']-k1['over_commit'], '변화율': f"{(k2['over_commit']/k1['over_commit']-1)*100:+.1f}%"},
    {'지표': '총 추가 발전 (MWh)', 'Phase 1 단독': k1['total_dE'], 'Phase 1+2': k2['total_dE'],
     'Δ': k2['total_dE']-k1['total_dE'], '변화율': f"{(k2['total_dE']/k1['total_dE']-1)*100:+.1f}%"},
    {'지표': '신규 가동 (회)', 'Phase 1 단독': k1['startup_count'], 'Phase 1+2': k2['startup_count'],
     'Δ': k2['startup_count']-k1['startup_count'], '변화율': '-'},
    {'지표': '예열 명령 (회)', 'Phase 1 단독': k1['warmup_count'], 'Phase 1+2': k2['warmup_count'],
     'Δ': k2['warmup_count']-k1['warmup_count'], '변화율': '-'},
    {'지표': '추가 대응 모드 (시간)', 'Phase 1 단독': k1['state_increase'], 'Phase 1+2': k2['state_increase'],
     'Δ': k2['state_increase']-k1['state_increase'], '변화율': '-'},
])
st.dataframe(comp.round(1), hide_index=True, use_container_width=True)

# === 월별 비교 ===
st.divider()
st.subheader("📅 월별 비교")

m1 = kpi_monthly(log_p1)
m2 = kpi_monthly(log_p1p2)

fig_m = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=("월별 부족분 (MWh)", "월별 과대보충 (MWh)"),
                       vertical_spacing=0.10)
fig_m.add_trace(go.Bar(x=m1.month, y=m1.shortfall, name='Phase 1 단독', marker_color='#D32F2F', opacity=0.7),
                row=1, col=1)
fig_m.add_trace(go.Bar(x=m2.month, y=m2.shortfall, name='Phase 1+2', marker_color='#1976D2', opacity=0.85),
                row=1, col=1)
fig_m.add_trace(go.Bar(x=m1.month, y=m1.over_commit, name='Phase 1 단독 (과대)',
                        marker_color='#FFA000', opacity=0.7, showlegend=False),
                row=2, col=1)
fig_m.add_trace(go.Bar(x=m2.month, y=m2.over_commit, name='Phase 1+2 (과대)',
                        marker_color='#0D47A1', opacity=0.85, showlegend=False),
                row=2, col=1)
fig_m.update_yaxes(title_text="MWh", row=1, col=1)
fig_m.update_yaxes(title_text="MWh", row=2, col=1)
fig_m.update_layout(template='plotly_white', height=520, barmode='group',
                     margin=dict(t=60, b=20),
                     legend=dict(orientation='h', y=-0.07),
                     plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig_m, use_container_width=True)

# === 예열 명령 분석 ===
st.divider()
st.subheader("🔥 GT 예열 명령 분석 (Phase 1+2 모드)")

wu = warmup_distribution(log_p1p2)
wu_kr = wu.rename(columns={'unit': '호기', 'count': '명령 횟수', 'type': '구분'})

c_left, c_right = st.columns([1, 1])
with c_left:
    st.markdown("##### 호기별 예열 명령 횟수")
    st.dataframe(wu_kr, hide_index=True, use_container_width=True)
    st.caption(f"총 {wu['count'].sum()}회 — 모두 GT 호기. ST 호기 0회 (기동시간 1~3h, 1시간 lead time 부족).")

with c_right:
    fig_wu = go.Figure(go.Bar(
        x=wu.unit, y=wu['count'],
        marker_color='#FFA000',
        text=wu['count'], textposition='outside',
    ))
    fig_wu.update_layout(title="호기별 예열 명령 횟수",
                          height=350, xaxis_title="", yaxis_title="횟수",
                          margin=dict(t=40, b=20))
    st.plotly_chart(fig_wu, use_container_width=True)

# warm-up 시각별 hour 분포
st.markdown("##### 시간대별 예열 명령")
warmup_log = log_p1p2[log_p1p2.warm_up_unit != ''].copy()
if len(warmup_log) > 0:
    warmup_log['hour'] = warmup_log.datetime_kst.dt.hour
    by_hour = warmup_log.groupby('hour').size().reset_index(name='count')
    fig_h = go.Figure(go.Bar(x=by_hour.hour, y=by_hour['count'],
                              marker_color='#E65100', text=by_hour['count'], textposition='outside'))
    fig_h.update_layout(template='plotly_white', height=280,
                         xaxis_title="시각 (h)", yaxis_title="명령 횟수",
                         margin=dict(t=20, b=40))
    st.plotly_chart(fig_h, use_container_width=True)

# === Cloud-pass days 비교 ===
st.divider()
st.subheader("☁ 큰 변동일 비교 — Phase 1 단독 vs Phase 1+2")

st.markdown("당일 갱신 예측이 *구름 통과* 같은 큰 변동을 어떻게 잡는가 — 직접 비교.")

event_rows = []
for label, date_str in EVENT_DAYS.items():
    date_sel = pd.Timestamp(date_str)
    d1 = log_p1[log_p1.date == date_sel]
    d2 = log_p1p2[log_p1p2.date == date_sel]
    if len(d1) == 0 or len(d2) == 0:
        continue
    event_rows.append({
        '날짜': date_str,
        'Phase 1 부족': float(d1.shortfall_mw.sum()),
        'Phase 1+2 부족': float(d2.shortfall_mw.sum()),
        'Δ': float(d2.shortfall_mw.sum() - d1.shortfall_mw.sum()),
        'Phase 1 과대': float(d1.over_commit_mw.sum()),
        'Phase 1+2 과대': float(d2.over_commit_mw.sum()),
        '예열 명령 (회)': int((d2.warm_up_unit != '').sum()),
    })
ev_df = pd.DataFrame(event_rows)
st.dataframe(ev_df.round(2), hide_index=True, use_container_width=True)

st.caption("""
이 날들은 Phase 1 (D-1) 이 PV 를 *과대 예측* 했지만 실측이 구름 통과 등으로 급감 → 큰 차이 발생.
Phase 2 향후 신호가 이를 1~3시간 사전 감지 → GT 예열 명령.
""")

# === 한 줄 요약 ===
st.divider()
st.subheader("💡 한 줄 결론")

improve_short = (k1['shortfall'] - k2['shortfall']) / k1['shortfall'] * 100
improve_over = (k1['over_commit'] - k2['over_commit']) / k1['over_commit'] * 100

st.success(f"""
**Phase 2 (당일 갱신 예측 + 정지 호기 보정) 의 정량 가치**

- 부족분 **−{improve_short:.1f}%** ({k1['shortfall']:.0f} → {k2['shortfall']:.0f} MWh)
- 과대보충 **−{improve_over:.1f}%** ({k1['over_commit']:.0f} → {k2['over_commit']:.0f} MWh)
- GT 예열 명령 **+{k2['warmup_count']}회** (Phase 1 단독 0회)
- 추가 대응 모드 **+{k2['state_increase']}시간** (향후 신호 기반 선제 대응)

→ 동일 호기 fleet, 동일 자체 흡수폭 조건에서 *Phase 2 가 향후 신호와 GT 예열 명령* 을 만든 것 만으로
부족분과 과대보충 모두 감소.
""")
