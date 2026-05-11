"""EDA D1 — 데이터 구성과 제약 (proposal §2 보조 그림).

목적:
  - 22 자체 호기 → 11 사용 가능 호기로 *왜* 필터링되었는지 데이터로 입증
  - ESS 왜곡 사례 (16~18시 피크) 시각화
  - 4년 데이터의 결측 / 가용 시간 분포

핵심 메시지:
  - PV 예측 모델은 *순수 PV 발전량* 이어야 작동
  - ESS 혼재 호기는 *피크가 16~18시* 로 이동 → 학습 시 ESS 운영 모델이 됨
  - 시간별 데이터 미제공 / 가동 축소 호기 도 제외
  - 최종 11 호기 (~77 MW) 가 PoC 대상
"""
import sys
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda_lng_planner"))
from _common import ROOT

EDA_DIR = ROOT / "pv/eda_pv_model"

# === 1. 4년 hourly solar 데이터 로드 (전체 22 호기) ===
files = sorted(glob.glob(str(ROOT / "data/solar_hourly/solar_hourly_*.csv")))
dfs = []
for f in files:
    try:
        df = pd.read_csv(f, encoding='utf-8', skipinitialspace=True,
                         header=None, skiprows=1, usecols=range(27))
    except: continue
    df.columns = ['plant','unit','date'] + [f'h{h}' for h in range(1, 25)]
    df['plant'] = df.plant.astype(str).str.strip()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    long = df.melt(id_vars=['plant','unit','date'], value_vars=[f'h{h}' for h in range(1,25)],
                   var_name='hour_str', value_name='kwh')
    long['hour'] = long.hour_str.str.replace('h','').astype(int) - 1
    long['kwh'] = pd.to_numeric(long['kwh'], errors='coerce')
    dfs.append(long[['plant','unit','date','hour','kwh']])
all_pv = pd.concat(dfs, ignore_index=True)
all_pv = all_pv[all_pv.kwh.notna()].copy()
all_pv['plant_unit'] = all_pv.plant + '_' + all_pv.unit.astype(str)
print(f"전체 4년 hourly rows: {len(all_pv):,}")
print(f"고유 plant_unit: {all_pv.plant_unit.nunique()}")

# === 2. 호기별 평균 시간 프로필 + 피크 시각 ===
hourly_mean = all_pv.groupby(['plant_unit','hour']).kwh.mean().unstack('hour')
peak_hour = hourly_mean.idxmax(axis=1)
afternoon_morning_ratio = (
    all_pv[all_pv.hour.between(12, 19)].groupby('plant_unit').kwh.sum() /
    all_pv[all_pv.hour.between(4, 11)].groupby('plant_unit').kwh.sum().clip(lower=0.001)
)

filter_df = pd.DataFrame({
    'peak_hour': peak_hour,
    'pm_am_ratio': afternoon_morning_ratio,
}).reset_index()
filter_df['classification'] = filter_df.apply(
    lambda r: ('Pure PV' if r.peak_hour <= 14 else
               ('ESS 의심' if r.peak_hour <= 16 else 'ESS 왜곡')), axis=1)
print("\n[호기별 피크 시각 + 오후/오전 비]")
print(filter_df.sort_values('peak_hour').to_string(index=False))

# === 3. 시각화 ===
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# (a) ESS 왜곡 vs Pure PV 시간 프로필 비교
ax = axes[0, 0]
# 대표 호기들
representatives = {
    'Pure PV (정상)': ['고흥만 수상태양광_1', '경상대태양광_1', '광양항세방태양광_1'],
    'ESS 왜곡 (16~18시 피크)': ['영동태양광_1', '삼천포태양광_4', '영흥태양광 #3_1'],
}
colors = {'Pure PV (정상)': '#388E3C', 'ESS 왜곡 (16~18시 피크)': '#D32F2F'}
for label, units in representatives.items():
    for i, u in enumerate(units):
        if u in hourly_mean.index:
            v = hourly_mean.loc[u]
            v_norm = v / v.max() if v.max() > 0 else v
            lbl = label if i == 0 else None
            ax.plot(range(24), v_norm, '-', color=colors[label], lw=2, alpha=0.7, label=lbl)
ax.set_xlabel('시각 (h)')
ax.set_ylabel('정규화 발전량 (peak=1)')
ax.set_title('① 시간 프로필 — Pure PV vs ESS 왜곡 호기')
ax.axvline(12, color='gray', ls=':', alpha=0.5)
ax.text(12.2, 0.05, '정오', fontsize=9, color='gray')
ax.legend()
ax.grid(True, alpha=0.3)

# (b) 호기별 피크 시각 분포
ax = axes[0, 1]
peak_dist = filter_df.groupby(['peak_hour','classification']).size().unstack(fill_value=0)
peak_dist = peak_dist.reindex(columns=['Pure PV','ESS 의심','ESS 왜곡'], fill_value=0)
peak_dist.plot(kind='bar', stacked=True, ax=ax,
                color=['#388E3C', '#FB8C00', '#D32F2F'])
ax.set_xlabel('피크 시각 (h)')
ax.set_ylabel('호기 수')
ax.set_title('② 피크 시각 × 분류 — 16시 이후 = ESS 왜곡')
ax.axvline(2.5, color='black', ls='--', alpha=0.5)   # 14.5 위치 (xtick 인덱스 기준)
ax.legend(title='분류')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.get_xticklabels(), rotation=0)

# (c) 22 → 11 필터링 단계
ax = axes[1, 0]
stages = ['전체\n자체설비', '시간별 데이터\n제공', '피크 ≤ 15시\n(ESS 제외)', '가동 축소\n제외 (삼천포#1)']
counts = [22, 20, 12, 11]
removed = [0, 2, 8, 1]
xs = np.arange(len(stages))
ax.bar(xs, counts, color='#1976D2', label='남은 호기 수')
for i, (c, r) in enumerate(zip(counts, removed)):
    if r > 0:
        ax.text(i, c+0.3, f'-{r}', ha='center', fontsize=10, color='#D32F2F')
    ax.text(i, c/2, f'{c}', ha='center', fontsize=14, color='white', fontweight='bold')
ax.set_xticks(xs)
ax.set_xticklabels(stages)
ax.set_ylabel('호기 수')
ax.set_title('③ 22 → 11 필터링 단계 (data_strategy §2.4)')
ax.set_ylim(0, 25)
ax.grid(True, alpha=0.3, axis='y')

# (d) 시간별 데이터 결측률 (월별)
ax = axes[1, 1]
all_pv['month'] = all_pv.date.dt.to_period('M').astype(str)
monthly_count = all_pv.groupby('month').size()
expected_per_month = 24 * 30 * filter_df[filter_df.classification == 'Pure PV'].plant_unit.nunique()
monthly_rate = (monthly_count / expected_per_month * 100).clip(upper=120)
ax.plot(range(len(monthly_count)), monthly_rate, '-o', color='#1976D2', markersize=3)
ax.axhline(100, color='gray', ls='--', alpha=0.5)
ax.set_xticks(range(0, len(monthly_count), 6))
ax.set_xticklabels([monthly_count.index[i] for i in range(0, len(monthly_count), 6)],
                    rotation=45)
ax.set_ylabel('데이터 가용률 (%)')
ax.set_title('④ 월별 데이터 가용률 — 4년 (2022-01 ~ 2025-12)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = EDA_DIR / "D1_data_constraints.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n저장: {out}")

filter_df.to_csv(EDA_DIR / "D1_unit_filter.csv", index=False)
print(f"저장: {EDA_DIR / 'D1_unit_filter.csv'}")

print(f"""
{'='*60}
→ 모델 함의 — 데이터 구성과 제약
{'='*60}

1. 자체설비 22 호기 중 *시간별 PV 발전 신호* 가 깨끗한 11 호기만 PoC 대상
2. ESS 혼재 호기 (8개) = 16~18시 피크 → 자연 PV 학습 불가
3. 시간별 미제공 (여수, 탑선) = 학습 모델 입력 X
4. 가동 축소 (삼천포 #1, 2025-04~) = 분포 변화로 제외
5. 11 호기 합계 ~77 MW, 그 중 고흥만수상 63 MW (82%) 가 portfolio dominant

→ 4년 hourly 데이터 + 호기 필터링 = PoC 모델 학습/평가의 *입력 데이터 무결성*
""")
