"""호기별 백업 — 호기별 ΔP, 실 발전 / 예열 명령, 보기 모드(일별/전체)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.data_loader import (
    load_planner, fleet_share_year, monthly_unit_dE, unit_year_stats,
    DEADBAND_MW, GT_UNITS, ST_UNITS, LNG_UNITS,
    UNIT_COLORS, GT_COLORS, ST_COLORS,
    PRESET_DAYS, EVENT_DAYS,
)

st.set_page_config(page_title="호기별 백업", page_icon="⚙", layout="wide")
# sidebar 에서 'app' 항목 숨김
st.markdown("""
<style>
[data-testid="stSidebarNav"] ul li:first-child {display: none;}
</style>
""", unsafe_allow_html=True)

st.title("⚙ 호기별 백업 — 호기별 분담 + 실 발전 / 예열 결정")

# === 호기 구성 표 (분당화력 LNG 복합) ===
st.markdown("**호기 구성** (분당화력 LNG 복합)")
config_df = pd.DataFrame([
    {'종류': '가스터빈 (GT)', '호기': 'CG1 ~ CG8', '기동시간 (추정)': '~30분 (빠름)'},
    {'종류': '증기터빈 (ST)', '호기': 'CS1 ~ CS2', '기동시간 (추정)': '1~3시간 (느림)'},
])
st.dataframe(config_df, hide_index=True, use_container_width=True)

st.markdown(f"""
**2-단계 의사결정**:
- **Layer A (실 발전)**: LNG 대응 차이 (= 실시간 차이 − 자체 흡수폭 {DEADBAND_MW:.0f} MW) 즉시 보정. 운전 중 호기에 (여유 출력 × 상승 한도) 가중 분배. 부족 시 GT 우선 신규 가동.
- **Layer B (사전 대비)**: 향후 예상 차이 강신호 시 *GT 전용 예열 명령*. 즉시 출력 X, 다음 시각 분배 후보.
""")

log = load_planner('phase1plus2')

# === Sidebar: 보기 모드 토글 ===
st.sidebar.header("📊 보기 모드")
view_mode = st.sidebar.radio(
    "모드 선택",
    options=["📅 일별 (선택한 날짜의 09~17시 흐름)", "📈 전체 (1년 누적)"],
    index=0,
)

# ============================================================
# === 일별 모드
# ============================================================
if view_mode.startswith("📅"):
    st.sidebar.header("📅 날짜")
    preset_keys = ["— 안정적 평일 —"] + list(PRESET_DAYS.keys()) + \
                  ["— 큰 변동일 —"] + list(EVENT_DAYS.keys()) + \
                  ["📋 직접 선택"]
    choice = st.sidebar.selectbox("날짜 선택", preset_keys, index=1)
    if choice == "📋 직접 선택":
        avail = sorted(log['date'].dt.strftime("%Y-%m-%d").unique())
        date_str = st.sidebar.selectbox("날짜", avail)
    elif choice in PRESET_DAYS:
        date_str = PRESET_DAYS[choice]
    elif choice in EVENT_DAYS:
        date_str = EVENT_DAYS[choice]
    else:
        date_str = list(PRESET_DAYS.values())[0]
    date_sel = pd.Timestamp(date_str)

    day = log[log.date == date_sel].sort_values('hour').copy()
    if len(day) == 0:
        st.warning("해당 날짜 운영 log 없음.")
        st.stop()

    st.markdown(f"### 📅 {date_str} — 일별 시간별 보기")

    # === 일별 KPI ===
    day_short = day.shortfall_mw.sum()
    day_over = day.over_commit_mw.sum()
    day_dE = sum(day[f'dP_{u}'].sum() for u in LNG_UNITS)
    day_warmup = (day.warm_up_unit != '').sum()
    day_real = int(day.startup_real.sum())
    day_ramp = int(day.startup_ramp.sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("부족분", f"{day_short:.1f} MWh")
    c2.metric("과대보충", f"{day_over:.1f} MWh")
    c3.metric("총 추가 발전", f"{day_dE:.1f} MWh")
    c4.metric("예열 명령", f"{int(day_warmup)}회")
    c5.metric("신규 가동 / 추가 발전 전환",
              f"{day_real} / {day_ramp}",
              help="진짜 신규 가동 (offline → 발전) / 이미 운전 중인 호기가 추가 ΔP 시작")

    # === 메인 차트: stacked area + state + 이벤트 ===
    st.subheader("호기별 추가 출력 (시간별, 누적)")

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=(
            "<b>호기별 추가 출력 (MW, 누적)</b> — ST(파랑) + GT(주황)",
            "<b>실시간 차이 vs LNG 응답</b>",
            "<b>운영 모드 + 향후 예상 차이</b>",
            "<b>호기 이벤트</b> — 추가 발전 전환 / 신규 가동 / 예열 명령",
        ),
        row_heights=[0.40, 0.30, 0.15, 0.15], vertical_spacing=0.06,
    )

    # Row 1: stacked area
    for u in ST_UNITS:
        col = f'dP_{u}'
        if col in day.columns:
            fig.add_trace(go.Scatter(
                x=day.datetime_kst, y=day[col],
                name=f'{u} (ST)', stackgroup='lng',
                mode='none', fillcolor=UNIT_COLORS[u],
                hovertemplate=f'{u}: %{{y:.1f}} MW<extra></extra>',
            ), row=1, col=1)
    for u in GT_UNITS:
        col = f'dP_{u}'
        if col in day.columns:
            fig.add_trace(go.Scatter(
                x=day.datetime_kst, y=day[col],
                name=f'{u} (GT)', stackgroup='lng',
                mode='none', fillcolor=UNIT_COLORS[u],
                hovertemplate=f'{u}: %{{y:.1f}} MW<extra></extra>',
            ), row=1, col=1)

    # Row 2: gap 비교
    fig.add_trace(go.Scatter(x=day.datetime_kst, y=day.instant_gap_mw,
                             name='실시간 차이', line=dict(color='gray', width=1, dash='dot')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=day.datetime_kst, y=day.effective_gap_mw,
                             name='LNG 대응 차이', line=dict(color='#D32F2F', width=2.5)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=day.datetime_kst, y=day.total_alloc_mw,
                             name='LNG 실제 응답', line=dict(color='#1976D2', width=2.5, dash='dash')),
                  row=2, col=1)
    fig.add_hrect(y0=-DEADBAND_MW, y1=DEADBAND_MW,
                  fillcolor='lightgray', opacity=0.25, line_width=0,
                  annotation_text=f'자체 흡수폭 ±{DEADBAND_MW:.0f} MW', annotation_position='top left',
                  row=2, col=1)
    fig.add_hline(y=0, line=dict(color='black', width=0.5), row=2, col=1)

    # Row 3: 운영 모드 + forward
    state_label_kr = {'KEEP': '유지', 'INCREASE': '추가 대응', 'DELAYED_RELEASE': '점진 해제'}
    state_color = {'KEEP': '#9E9E9E', 'INCREASE': '#D32F2F', 'DELAYED_RELEASE': '#1976D2'}
    for st_name, color in state_color.items():
        sub = day[day.state == st_name]
        if len(sub) > 0:
            kr = state_label_kr[st_name]
            fig.add_trace(go.Scatter(x=sub.datetime_kst, y=[1]*len(sub),
                                      name=kr, mode='markers',
                                      marker=dict(color=color, size=14, symbol='square'),
                                      hovertemplate=f'{kr}<extra></extra>'),
                          row=3, col=1)
    fig.add_trace(go.Scatter(x=day.datetime_kst, y=day.FG1,
                             name='향후 1h 예상 차이', line=dict(color='purple', width=1.5)),
                  row=3, col=1)

    # Row 4: 이벤트 (3 종류 분리)
    real_pts = day[day.startup_real > 0]
    if len(real_pts) > 0:
        fig.add_trace(go.Scatter(x=real_pts.datetime_kst, y=[1]*len(real_pts),
                                  name='신규 가동 (offline→발전)', mode='markers',
                                  marker=dict(color='red', size=14, symbol='circle'),
                                  hovertemplate='신규 가동<extra></extra>'),
                      row=4, col=1)
    ramp_pts = day[day.startup_ramp > 0]
    if len(ramp_pts) > 0:
        fig.add_trace(go.Scatter(x=ramp_pts.datetime_kst, y=[2]*len(ramp_pts),
                                  name='추가 발전 전환', mode='markers',
                                  marker=dict(color='orange', size=10, symbol='triangle-up'),
                                  hovertemplate='추가 발전 전환<extra></extra>'),
                      row=4, col=1)
    warmup_pts = day[day.warm_up_unit != '']
    if len(warmup_pts) > 0:
        fig.add_trace(go.Scatter(x=warmup_pts.datetime_kst, y=[3]*len(warmup_pts),
                                  name='예열 명령 (GT)', mode='markers+text',
                                  marker=dict(color='#E65100', size=14, symbol='star'),
                                  text=warmup_pts.warm_up_unit,
                                  textposition='top center',
                                  hovertemplate='예열: %{text}<extra></extra>'),
                      row=4, col=1)

    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="모드", row=3, col=1, showticklabels=False)
    fig.update_yaxes(title_text="이벤트", row=4, col=1, showticklabels=False, range=[0, 4])
    fig.update_xaxes(title_text="시간 (KST)", row=4, col=1)
    fig.update_layout(template='plotly_white', height=940, hovermode='x unified',
                       legend=dict(orientation='h', y=-0.18, yanchor='top'),
                       margin=dict(t=70, b=110),
                       plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

    # === 일별 호기별 누적 ===
    st.subheader(f"{date_str} — 호기별 일별 추가 발전량 (MWh)")

    day_unit = []
    for u in LNG_UNITS:
        col = f'dP_{u}'
        if col in day.columns:
            day_unit.append({
                '호기': u,
                '종류': 'GT' if u.startswith('CG') else 'ST',
                '추가 발전 (MWh)': float(day[col].sum()),
                '최대 출력 (MW)': float(day[col].max()),
                '운전 시간': int((day[col] > 0).sum()),
            })
    day_unit_df = pd.DataFrame(day_unit).sort_values('추가 발전 (MWh)', ascending=False)

    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.dataframe(day_unit_df.round(2), hide_index=True, use_container_width=True)
    with c_right:
        fig_pie = go.Figure(go.Pie(
            labels=day_unit_df['호기'], values=day_unit_df['추가 발전 (MWh)'].clip(lower=0.01),
            marker=dict(colors=[UNIT_COLORS[u] for u in day_unit_df['호기']]),
            hole=0.4, sort=False,
        ))
        fig_pie.update_layout(title="일별 호기별 분담 비중", height=350,
                              margin=dict(t=40, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    # === 시간별 상세 표 ===
    with st.expander("🔎 시간별 상세 (운영 모드, 향후 신호, 호기별 출력)"):
        show_cols = ['hour', 'instant_gap_mw', 'effective_gap_mw', 'total_alloc_mw',
                     'shortfall_mw', 'over_commit_mw',
                     'state', 'FG1', 'slope', 'warm_up_unit',
                     'startup_real', 'startup_ramp']
        show_cols += [f'dP_{u}' for u in LNG_UNITS]
        table = day[show_cols].copy()
        table['hour'] = table['hour'].astype(int).astype(str) + ":00"
        table['state'] = table['state'].map(state_label_kr)
        rename_map = {
            'hour': '시각', 'instant_gap_mw': '실시간 차이', 'effective_gap_mw': 'LNG 대응 차이',
            'total_alloc_mw': 'LNG 응답', 'shortfall_mw': '부족', 'over_commit_mw': '과대보충',
            'state': '운영 모드', 'FG1': '향후 1h', 'slope': '변화율',
            'warm_up_unit': '예열 호기',
            'startup_real': '신규 가동', 'startup_ramp': '추가 발전 전환',
        }
        table = table.rename(columns=rename_map)
        st.dataframe(table.round(2), hide_index=True, use_container_width=True)


# ============================================================
# === 전체 모드 (1년 누적)
# ============================================================
else:
    st.markdown("### 📈 전체 보기 — 1년 누적 (2025년, Phase 1+2 모드)")

    total_short = log.shortfall_mw.sum()
    total_over = log.over_commit_mw.sum()
    total_dE = sum(log[f'dP_{u}'].sum() for u in LNG_UNITS)
    total_warmup = (log.warm_up_unit != '').sum()
    total_real = int(log.startup_real.sum())
    total_ramp = int(log.startup_ramp.sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("1년 부족분", f"{total_short:.0f} MWh")
    c2.metric("1년 과대보충", f"{total_over:.0f} MWh")
    c3.metric("1년 총 추가 발전", f"{total_dE:.0f} MWh")
    c4.metric("예열 명령 (전부 GT)", f"{int(total_warmup)}회")
    c5.metric("신규 가동 / 추가 발전 전환",
              f"{total_real} / {total_ramp}",
              help="진짜 신규 가동은 거의 발생 X — 대부분이 *이미 운전 중인 호기의 추가 발전 시작*")

    # === 호기별 1년 누적 ===
    st.subheader("호기별 1년 누적 추가 발전량")

    share = fleet_share_year(log)
    c_left, c_right = st.columns([3, 2])
    with c_left:
        show = share.copy()
        show.columns = ['호기', '종류', '추가 발전 (MWh)', '비중']
        show['추가 발전 (MWh)'] = show['추가 발전 (MWh)'].round(1)
        show['비중'] = (show['비중'] * 100).round(1).astype(str) + "%"
        st.dataframe(show, hide_index=True, use_container_width=True)
        st.caption(f"총 추가 발전량 = {share.dE_MWh.sum():.0f} MWh "
                    f"(ST 호기 {share[share.type=='ST'].dE_MWh.sum():.0f} / "
                    f"GT 호기 {share[share.type=='GT'].dE_MWh.sum():.0f})")
    with c_right:
        fig_bar = go.Figure(go.Bar(
            x=share.unit, y=share.dE_MWh,
            marker_color=[UNIT_COLORS[u] for u in share.unit],
            text=share.dE_MWh.round(0), textposition='outside',
        ))
        fig_bar.update_layout(title="호기별 1년 누적", height=400,
                               xaxis_title="", yaxis_title="MWh",
                               margin=dict(t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # === 월별 호기 stacked ===
    st.subheader("월별 호기별 추가 발전량 (stacked)")

    mu = monthly_unit_dE(log)
    fig_m = go.Figure()
    # ST 먼저 (아래)
    for u in ST_UNITS:
        sub = mu[mu.unit == u]
        if len(sub):
            fig_m.add_trace(go.Bar(x=sub.month, y=sub.dE_MWh, name=f'{u} (ST)',
                                    marker_color=UNIT_COLORS[u]))
    for u in GT_UNITS:
        sub = mu[mu.unit == u]
        if len(sub):
            fig_m.add_trace(go.Bar(x=sub.month, y=sub.dE_MWh, name=f'{u} (GT)',
                                    marker_color=UNIT_COLORS[u]))
    fig_m.update_layout(template='plotly_white', barmode='stack', height=460,
                         xaxis_title="월", yaxis_title="추가 발전 (MWh)",
                         legend=dict(orientation='h', y=-0.15),
                         margin=dict(t=20, b=20),
                         plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig_m, use_container_width=True)

    # === 호기별 운전 통계 ===
    st.subheader("호기별 1년 운전 통계")

    stats = unit_year_stats(log)
    stats_show = stats.copy()
    stats_show.columns = ['호기', '종류', '추가 발전 (MWh)', '최대 출력 (MW)',
                          '평균 출력 (운전중, MW)', '활성 시간 (h)', '운전율 (%)']
    stats_show = stats_show.round({'추가 발전 (MWh)': 1, '최대 출력 (MW)': 1,
                                    '평균 출력 (운전중, MW)': 2})
    st.dataframe(stats_show, hide_index=True, use_container_width=True)

    st.caption("""
**해석 가이드**
- *활성 시간* = ΔP > 0.1 MW 인 시간 수 (운전 중 + 추가 발전 시작 시각)
- *운전율* = 활성 시간 / 전체 daytime 3,280시간
- ST 두 호기 (CS1, CS2) 가 운전율 압도적 (헤드룸 큼) — backbone 역할
- GT 호기들은 보조적으로 동원 (특히 큰 변동일에 GT 우선 신규 가동)
""")
