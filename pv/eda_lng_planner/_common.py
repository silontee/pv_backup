"""EDA 공통 설정 — 한글 폰트, 색 컨벤션, 경로."""
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# Windows 한글 폰트
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
EDA_DIR = ROOT / "pv/eda_v4"
EDA_DIR.mkdir(exist_ok=True)

# 호기 색 (dashboard 와 통일)
GT_UNITS = ['CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8']
ST_UNITS = ['CS1','CS2']
LNG_UNITS = GT_UNITS + ST_UNITS
GT_COLORS = ['#FFC107','#FFB300','#FFA000','#FF8F00',
             '#FF6F00','#E65100','#BF360C','#8B0000']
ST_COLORS = ['#1976D2','#0D47A1']
UNIT_COLORS = {**{u: GT_COLORS[i] for i,u in enumerate(GT_UNITS)},
               **{u: ST_COLORS[i] for i,u in enumerate(ST_UNITS)}}

# KPI 색
COL_SHORT = '#D32F2F'    # 부족분 (빨강)
COL_OVER = '#1976D2'     # 과대보충 (파랑)
COL_DE = '#FB8C00'       # 추가 발전 (주황)
COL_KEEP = '#9E9E9E'
COL_INC = '#D32F2F'
COL_REL = '#1976D2'
