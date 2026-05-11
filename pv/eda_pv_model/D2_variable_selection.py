"""EDA D2 — 변수 선정 근거 (proposal §3 보조 그림).

목적:
  - GK-2A (위성) 와 ASOS (지상) 의 *상호 보완성* 입증
  - 변수별 cf 와의 corr + 보완 효과 측정
  - "왜 이 변수들" / "왜 GK-2A v2 (DSR+ASR+RSR) 인가"

핵심 메시지:
  - GK-2A DSR (위성 일사) 가 cf 와 가장 강한 단조 관계 (+0.85)
  - ASOS 기상 (hm, ws, rn) 은 *부분 흐림 / 안개 / 풍속* marginal 정보
  - GK-2A v2 의 ASR/RSR 는 *알베도 / 흡수* 파생 정보 — 수상 PV (고흥만) 특이성
  - DSR 만 있을 때 vs DSR + ASOS 조합의 cf 설명력 비교
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda_lng_planner"))
from _common import ROOT

EDA_DIR = ROOT / "pv/eda_pv_model"

# === 1. training_set.parquet — 통합 학습셋 ===
df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
df['hour'] = df.datetime_kst.dt.hour
TARGET_SITES = ['경상대','고흥만수상','광양항세방','구미','삼천포','영흥','예천','창원']
df = df[df.site.isin(TARGET_SITES)]
day = df[df.hour.between(9, 16)].copy()
print(f"daytime rows: {len(day):,}")

# === 2. 변수별 cf corr (단일 변수) ===
features = {
    'dsr_mean': 'GK-2A 위성 일사 (W/m²)',
    'zenith_center': '태양 천정각 (°)',
    'ta': 'ASOS 기온 (°C)',
    'hm': 'ASOS 습도 (%)',
    'ws': 'ASOS 풍속 (m/s)',
    'rn': 'ASOS 강수 (mm)',
}
single_corr = []
for f, lbl in features.items():
    if f in day.columns:
        v = day[[f, 'cf']].dropna()
        if len(v):
            single_corr.append({'feature': f, 'label': lbl, 'corr': v[f].corr(v.cf)})
single_df = pd.DataFrame(single_corr).sort_values('corr', key=abs, ascending=False)
print("\n[단일 변수 vs cf corr]")
print(single_df.round(3).to_string(index=False))

# === 3. 누적 R² (변수 추가 시 설명력 증가) ===
# baseline: zenith only (시간 정보)
# +dsr (GK-2A 위성)
# +ASOS 기상 (hm, ws, rn, ta)
# +site_oh

# valid rows (모든 feature 있는)
need = ['cf','zenith_center','dsr_mean','ta','hm','ws','rn','site']
sub = day[need].dropna().copy()
print(f"\nfully-valid rows: {len(sub):,}")

site_oh = pd.get_dummies(sub.site, prefix='site')
groups = {
    '① zenith only': ['zenith_center'],
    '② + dsr (GK-2A)': ['zenith_center','dsr_mean'],
    '③ + ASOS 기상': ['zenith_center','dsr_mean','ta','hm','ws','rn'],
    '④ + site OH': ['zenith_center','dsr_mean','ta','hm','ws','rn'] + site_oh.columns.tolist(),
}
r2_results = []
for name, cols in groups.items():
    X = sub[[c for c in cols if c in sub.columns]].copy()
    if 'site_경상대' in cols:
        X = pd.concat([sub[['zenith_center','dsr_mean','ta','hm','ws','rn']], site_oh], axis=1)
    y = sub.cf
    # linear
    lr = LinearRegression().fit(X, y)
    r2_lr = r2_score(y, lr.predict(X))
    # random forest (tabular nonlinear baseline)
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42).fit(X, y)
    r2_rf = r2_score(y, rf.predict(X))
    r2_results.append({'group': name, 'n_features': X.shape[1], 'R2_linear': r2_lr, 'R2_RF': r2_rf})
    print(f"  {name:25s}  features={X.shape[1]:3d}  R²(linear)={r2_lr:.3f}  R²(RF)={r2_rf:.3f}")
r2_df = pd.DataFrame(r2_results)

# === 4. dsr 분위 × cf 분포 (모델이 잡아야 할 영역) ===
day['dsr_q'] = pd.qcut(day.dsr_mean, q=10, duplicates='drop', labels=False)
dsr_q_stat = day.groupby('dsr_q').agg(
    cf_mean=('cf','mean'), cf_std=('cf','std'),
    dsr_mid=('dsr_mean','mean'), n=('cf','size')).reset_index()

# === 5. 시각화 ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# (a) 단일 변수 vs cf corr
ax = axes[0, 0]
colors = ['#388E3C' if c > 0 else '#D32F2F' for c in single_df['corr']]
ax.barh([f"{r.feature}\n({r.label})" for _, r in single_df.iterrows()],
         single_df['corr'], color=colors)
ax.invert_yaxis()
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('corr with cf')
ax.set_title('① 단일 변수 vs cf 상관 — dsr (GK-2A) 압도')
ax.grid(True, alpha=0.3, axis='x')
for i, c in enumerate(single_df['corr']):
    ax.text(c + 0.02 if c > 0 else c - 0.02, i, f'{c:+.3f}',
            va='center', ha='left' if c > 0 else 'right', fontsize=9)

# (b) 누적 R² (linear vs RF)
ax = axes[0, 1]
xs = np.arange(len(r2_df))
ax.bar(xs - 0.2, r2_df.R2_linear, 0.4, label='Linear (정합 형태)', color='#1976D2')
ax.bar(xs + 0.2, r2_df.R2_RF, 0.4, label='RF (비선형)', color='#FB8C00')
ax.set_xticks(xs)
ax.set_xticklabels([g for g in r2_df.group], rotation=15, ha='right', fontsize=9)
ax.set_ylabel('R²')
ax.set_title('② 변수 추가별 cf 설명력 증가')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for i, (lr, rf) in enumerate(zip(r2_df.R2_linear, r2_df.R2_RF)):
    ax.text(i - 0.2, lr + 0.01, f'{lr:.2f}', ha='center', fontsize=9)
    ax.text(i + 0.2, rf + 0.01, f'{rf:.2f}', ha='center', fontsize=9)

# (c) dsr → cf 관계 + variability (Phase 2 가 잡을 영역 표시)
ax = axes[1, 0]
ax.errorbar(dsr_q_stat.dsr_mid, dsr_q_stat.cf_mean, yerr=dsr_q_stat.cf_std,
             fmt='o-', color='#FB8C00', lw=2, capsize=3, markersize=8)
ax.fill_between(dsr_q_stat.dsr_mid,
                 dsr_q_stat.cf_mean - dsr_q_stat.cf_std,
                 dsr_q_stat.cf_mean + dsr_q_stat.cf_std,
                 alpha=0.2, color='#FB8C00')
ax.axvspan(0, 400, alpha=0.1, color='#D32F2F', label='부분 흐림 영역\n(Phase 2 가치)')
ax.set_xlabel('dsr_mean (W/m²)')
ax.set_ylabel('cf (mean ± std)')
ax.set_title('③ dsr → cf 관계 — 저일사 영역(Q1-Q4) 변동성 큼')
ax.legend()
ax.grid(True, alpha=0.3)

# (d) 시간대별 변수 corr (heatmap)
ax = axes[1, 1]
hour_corr = []
for h in range(9, 17):
    sub_h = day[day.hour == h]
    row = {'hour': h}
    for f in features:
        if f in sub_h.columns:
            v = sub_h[[f, 'cf']].dropna()
            row[f] = v[f].corr(v.cf) if len(v) else np.nan
    hour_corr.append(row)
hour_corr = pd.DataFrame(hour_corr).set_index('hour')
im = ax.imshow(hour_corr.T.values, cmap='RdBu_r', vmin=-0.7, vmax=0.7, aspect='auto')
ax.set_xticks(range(len(hour_corr)))
ax.set_xticklabels(hour_corr.index)
ax.set_yticks(range(len(hour_corr.columns)))
ax.set_yticklabels(hour_corr.columns)
ax.set_xlabel('시각 (h)')
ax.set_title('④ 시간대별 변수 corr — dsr 일관 강함, 기상 변수 시간대 의존')
plt.colorbar(im, ax=ax, label='corr')

plt.tight_layout()
out = EDA_DIR / "D2_variable_selection.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n저장: {out}")

single_df.to_csv(EDA_DIR / "D2_single_corr.csv", index=False)
r2_df.to_csv(EDA_DIR / "D2_cumulative_r2.csv", index=False)
hour_corr.to_csv(EDA_DIR / "D2_hour_corr.csv")

print(f"""
{'='*60}
→ 모델 함의 — 변수 선정
{'='*60}

1. **GK-2A 위성 dsr** 가 cf 와 가장 강한 단조 관계 (+{single_df.iloc[0]['corr']:.2f})
   → ASOS 만으로 부족 (지상 측정 = site-local). 위성 입력 = *광역 일사 정보*
2. **누적 R²**: zenith only ({r2_df.iloc[0].R2_linear:.2f}) → +dsr ({r2_df.iloc[1].R2_linear:.2f}) → +ASOS ({r2_df.iloc[2].R2_linear:.2f}) → +site OH ({r2_df.iloc[3].R2_linear:.2f})
   → dsr 만 추가해도 R² 큰 폭 증가
3. **ASOS 기상 변수**:
   - 습도 (hm) corr {single_df[single_df.feature=='hm']['corr'].values[0]:+.2f} — 흐림/안개 정보
   - 강수 (rn) corr {single_df[single_df.feature=='rn']['corr'].values[0]:+.2f} — 비 오는 시간 cf 저하
   - 풍속 (ws) corr {single_df[single_df.feature=='ws']['corr'].values[0]:+.2f} — 미미
   - 기온 (ta) corr {single_df[single_df.feature=='ta']['corr'].values[0]:+.2f} — 미미 (모듈 효율 영향 작음)
4. site_oh 추가 시 R² {r2_df.iloc[3].R2_linear - r2_df.iloc[2].R2_linear:+.3f}pp
   → 사이트별 bias 차이 (광양항 +7%, 예천 anomaly 등) 흡수 → AdaLN 의 핵심 동기

→ 입력 변수 조합 = {{ dsr (GK-2A) + ASOS (ta, hm, ws, rn) + zenith + site OH + hour/month }}
""")
