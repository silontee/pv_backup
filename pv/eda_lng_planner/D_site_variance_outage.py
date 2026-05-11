"""EDA D — PV 8 사이트 variance + outage 패턴 시각화.

목적:
  proposal §5 의 "고흥만수상 점심 outage / 광양항세방 10월 shutdown / 구미 6월 outage"
  를 *그림 한 장* 으로 정량 시각화.

방법:
  test 2025 Phase 1 결과 (cf, mu_p1) 사이트별 분석 + outage rule 적용 결과.

→ 모델 함의:
  - 사이트별 outage 패턴이 다름 (점심 / 10월 cluster / 6월 등)
  - portfolio 합 (cap-weighted) 시 noise 일부 상쇄, 그러나 큰 outage 는 portfolio 에도 영향
  - rule-based override (Phase 2 후처리) 가 적합한 이유 설명
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, EDA_DIR

# === 1. test 2025 Phase 1 결과 ===
p1 = pd.read_parquet(ROOT / 'pv/experiments/resmlp_adaln_v2_ensemble/ensemble_test.parquet')
p1['datetime_kst'] = pd.to_datetime(p1['datetime_kst'])
p1['date'] = p1.datetime_kst.dt.normalize()
p1['hour'] = p1.datetime_kst.dt.hour

# === 2. 사이트별 cf 분포 (daytime 9-17만) ===
day = p1[p1.hour.between(9, 17)].copy()

sites = sorted(day.site.unique())
print(f"sites: {sites}")
print(f"\n[사이트별 cf 분포 (daytime 9-17, test 2025)]")
for s in sites:
    sub = day[day.site == s]
    print(f"  {s:12s}  cap={sub.site_capacity_kw.iloc[0]/1000:6.2f}MW  cf mean={sub.cf.mean():.3f}  std={sub.cf.std():.3f}  outage(cf<0.03)= {(sub.cf<0.03).sum():4d}h ({(sub.cf<0.03).mean()*100:.1f}%)")

# === 3. 사이트별 outage 패턴 시각화 ===
fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=False)
axes = axes.flatten()

for ax, s in zip(axes, sites):
    sub = day[day.site == s].copy()
    sub['outage'] = ((sub.cf < 0.03) & (sub.mu_mean > 0.20)).astype(float)
    sub['day_of_year'] = sub.datetime_kst.dt.dayofyear
    pivot = sub.pivot_table(index='hour', columns='day_of_year',
                             values='outage', aggfunc='max', fill_value=0.0)
    pivot = pivot.astype(float)
    im = ax.imshow(pivot.values, aspect='auto', cmap='Reds', interpolation='nearest',
                   extent=[1, 365, 17.5, 8.5], vmin=0, vmax=1)
    cap_mw = sub.site_capacity_kw.iloc[0] / 1000
    n_out = sub.outage.sum()
    ax.set_title(f'{s} (cap {cap_mw:.1f} MW)  —  outage rule 발동 {n_out}h',
                  fontsize=11)
    ax.set_xlabel('day of year')
    ax.set_ylabel('hour')
    ax.set_xticks([1, 60, 121, 182, 244, 305])
    ax.set_xticklabels(['1/1','3/1','5/1','7/1','9/1','11/1'])

plt.tight_layout()
out = EDA_DIR / "D_site_outage_heatmap.png"
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f"\n저장: {out}")

# === 4. 사이트별 cf 분포 violin plot ===
fig2, ax = plt.subplots(figsize=(12, 5))
data = [day[day.site == s].cf.values for s in sites]
parts = ax.violinplot(data, showmeans=True, showmedians=True)
ax.set_xticks(range(1, len(sites)+1))
ax.set_xticklabels(sites, rotation=15)
ax.set_ylabel('cf (capacity factor, daytime 9-17)')
ax.set_title('사이트별 cf 분포 — outage 영향 + 일사 변동성')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
out2 = EDA_DIR / "D_site_cf_violin.png"
plt.savefig(out2, dpi=130, bbox_inches='tight')
print(f"저장: {out2}")

# === 5. 사이트별 outage 통계 csv ===
stats = []
for s in sites:
    sub = day[day.site == s]
    cap_mw = sub.site_capacity_kw.iloc[0] / 1000
    outage = (sub.cf < 0.03) & (sub.mu_mean > 0.20)
    stats.append({
        'site': s,
        'cap_MW': cap_mw,
        'cf_mean': float(sub.cf.mean()),
        'cf_std': float(sub.cf.std()),
        'outage_h': int(outage.sum()),
        'outage_pct': float(outage.mean() * 100),
    })
sdf = pd.DataFrame(stats).sort_values('cap_MW', ascending=False)
sdf.to_csv(EDA_DIR / "D_site_stats.csv", index=False)
print(f"저장: {EDA_DIR / 'D_site_stats.csv'}")

# === 6. 모델 함의 ===
print(f"""
{'='*60}
→ 모델 함의
{'='*60}

1. 사이트별 outage 패턴 다양:
   - 고흥만수상: 점심시간대 반복
   - 광양항세방: 가을 (10월 cluster)
   - 구미: 여름 (6월 등)
   - 작은 사이트들 (창원, 구미): 절대값 영향 작음 (cap 1 MW 미만)
2. 큰 사이트 (고흥만수상 63 MW) 의 outage = portfolio 영향 큼
3. cf 분포 사이트별로 다른 모양 (수상 vs 영농형 vs 항만 등)

→ outage 가 *site-level non-meteorological* 사건임을 데이터로 입증
→ Phase 2 학습으로 일관 흡수 어려움 → rule-based override 정당
""")
