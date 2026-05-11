"""Dashboard 진입점 — 자동으로 첫 페이지로 redirect.

Home 페이지의 PoC scope / KPI summary 등은 *제거됨*.
이전 Home 내용은 `_app_old_backup.py` 에 보관.

Run:
  streamlit run src/dashboard/app.py
"""
import streamlit as st

st.set_page_config(
    page_title="태양광 예측 + LNG 백업 PoC",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# sidebar 에서 'app' 항목 숨김 (CSS)
st.markdown("""
<style>
[data-testid="stSidebarNav"] ul li:first-child {display: none;}
</style>
""", unsafe_allow_html=True)

# 자동 redirect — 첫 페이지로
st.switch_page("pages/1_📊_예측_비교.py")
