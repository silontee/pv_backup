"""10_yecheon_isolated.ipynb 빌더 — 예천 단독 진단."""
import json
from pathlib import Path

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True),
                   "execution_count": None, "outputs": []})


md("""# 10. 예천 단독 진단 — global 문제에서 site-specific 문제로 재정의

**전제**: 노트북 09에서 예천 zero-cluster의 75%가 강수 무관(=clear weather에서 발생), 동일 시각 다른 사이트는 정상 발전. → outage-like 패턴.

**이 노트북의 4 부분**:
1. **near-zero 정의 sweep** (cf<0.03/0.05/0.08/0.10) → 정의에 따라 비율이 어떻게 변하나
2. **WEATHER_ZERO 조건 매트릭스** (dc10Tca / dsr_mean / rn 임계 sweep) → 어디까지가 날씨로 정당화 가능
3. **Residual zero (날씨로도 설명 안 되는 부분)** 의 구조 — partial cloud / hour-zenith / date cluster
4. **NMAE 기여 분해** — 예천 NMAE 8.12% 중 high-output / mid / near-zero 각각 얼마

**목표**: 예천 = global ML 문제가 아니라 **이 사이트의 사건 패턴**이므로 처치(masking, weighting, exclusion)도 사이트 단위로.
""")

code("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import matplotlib as mpl, matplotlib.font_manager as fm
for f in ['Malgun Gothic', 'NanumGothic', 'AppleGothic']:
    if any(f.lower() in n.name.lower() for n in fm.fontManager.ttflist):
        mpl.rcParams['font.family'] = f; break
mpl.rcParams['axes.unicode_minus'] = False

ROOT = Path('../..').resolve()
df = pd.read_parquet(ROOT / 'data/processed/training_set.parquet')
df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
df['rain'] = df['rn'].fillna(0)

ye = df[df.site == '예천'].copy().sort_values('datetime_kst').reset_index(drop=True)
print(f'예천 행: {len(ye):,}  | 기간: {ye.datetime_kst.min().date()} ~ {ye.datetime_kst.max().date()}')
print(f'capacity: {ye.site_capacity_kw.iloc[0]:.0f} kW')
""")

md("""## 1. near-zero threshold sweep

cf threshold를 0.03 / 0.05 / 0.08 / 0.10으로 바꿨을 때 daytime(zenith<60) zero 비율이 어떻게 변하나.

**판독**: threshold가 올라가면 비율이 단조 증가하지만, *증가 폭*이 크면 그 영역에 mass가 있다는 뜻. 단순 outage라면 0~0.03 구간에 mass 집중 (계측 stuck/zero), partial outage·shading이라면 0.03~0.10 구간에 분포.
""")

code("""day = ye[ye.zenith_center < 60].copy()
n = len(day)
print(f'예천 daytime(zenith<60) 행: {n:,}')
print()
print(f\"{'threshold':<12} {'count':>7} {'rate':>8}\")
for thr in [0.03, 0.05, 0.08, 0.10]:
    cnt = (day.cf < thr).sum()
    print(f'  cf < {thr:.2f}    {cnt:>7} {cnt/n*100:>7.2f}%')
print()
# 점진적 mass — 각 구간(0~0.03, 0.03~0.05, 0.05~0.08, 0.08~0.10)
print('  bin               count    cum%')
edges = [0, 0.03, 0.05, 0.08, 0.10, 1.5]
labels = ['[0,0.03)', '[0.03,0.05)', '[0.05,0.08)', '[0.08,0.10)', '[0.10, ∞)']
day['cf_bin'] = pd.cut(day.cf, edges, labels=labels, right=False)
counts = day.cf_bin.value_counts().reindex(labels)
cum = counts.cumsum() / n * 100
for lbl, c, cu in zip(labels, counts, cum):
    print(f'  {lbl:<16} {c:>7} {cu:>7.2f}%')
""")

md("""→ **모델 함의**: threshold 변동 곡선이 가파르게 올라가면 (예: 0.03→0.05에서 +5%p) 그 구간이 진짜 partial output. 평탄하면 (mass 없음) outage 정의를 그 임계값으로 잡아도 무리 없음.
""")

md("""## 2. WEATHER_ZERO 조건 매트릭스

세 가지 임계값(dc10Tca, dsr_mean, rn)의 조합을 매트릭스로 sweep해서 *예천 zero 중 얼마까지 weather로 정당화되나* 측정.

각 셀 = `% of 예천 near-zero (cf<0.05, zenith<60) explained by that weather rule`.

**판독 기준**:
- 어떤 조합도 70% 이상 설명 못 하면 → 진짜 outage 신호 강함
- 강한 임계(rn≥0.5만)에서도 30%+ 설명 → 강수 단독으로도 일부는 설명
""")

code("""near0 = day[day.cf < 0.05].copy()
n_n0 = len(near0)
print(f'예천 daytime near-zero (cf<0.05): {n_n0} 행')
print()
print('각 조건이 단독으로 설명하는 비율:')
print(f\"{'rule':<28} {'covered':>9} {'rate':>8}\")
for rule, mask in [
    ('dc10Tca >= 6',          near0.dc10Tca >= 6),
    ('dc10Tca >= 7',          near0.dc10Tca >= 7),
    ('dc10Tca >= 8',          near0.dc10Tca >= 8),
    ('dc10Tca >= 9',          near0.dc10Tca >= 9),
    ('dsr_mean < 100',        near0.dsr_mean < 100),
    ('dsr_mean < 200',        near0.dsr_mean < 200),
    ('dsr_mean < 300',        near0.dsr_mean < 300),
    ('rn >= 0.1',             near0.rain >= 0.1),
    ('rn >= 0.5',             near0.rain >= 0.5),
    ('rn >= 1.0',             near0.rain >= 1.0),
]:
    c = mask.sum()
    print(f'  {rule:<28} {c:>9} {c/n_n0*100:>7.2f}%')
print()
print('조합 (OR) 으로 설명하는 비율:')
combos = [
    ('dc≥8 OR dsr<200 OR rn≥0.5  (09 노트북)', (near0.dc10Tca>=8) | (near0.dsr_mean<200) | (near0.rain>=0.5)),
    ('dc≥7 OR dsr<300 OR rn≥0.5',              (near0.dc10Tca>=7) | (near0.dsr_mean<300) | (near0.rain>=0.5)),
    ('dc≥7 OR dsr<300 OR rn>0',                (near0.dc10Tca>=7) | (near0.dsr_mean<300) | (near0.rain>0)),
    ('dc≥6 OR dsr<300 OR rn>0',                (near0.dc10Tca>=6) | (near0.dsr_mean<300) | (near0.rain>0)),
    ('dc≥6 OR dsr<400 OR rn>0',                (near0.dc10Tca>=6) | (near0.dsr_mean<400) | (near0.rain>0)),
    ('dc≥5 OR dsr<400 OR rn>0',                (near0.dc10Tca>=5) | (near0.dsr_mean<400) | (near0.rain>0)),
]
print(f\"{'combo':<48} {'covered':>9} {'rate':>8}\")
for name, mask in combos:
    c = mask.sum()
    print(f'  {name:<48} {c:>9} {c/n_n0*100:>7.2f}%')
""")

md("""→ **모델 함의**: 가장 관대한 OR 조합(dc≥5 OR dsr<400 OR rn>0)으로도 100%에 도달 못 하면 그 잔차가 모델 입력으로 잡을 수 없는 **순수 outage**. 도달하면 임계만 더 느슨하게 해서 안전하게 mask 가능.
""")

md("""## 3. 가장 관대한 weather 조건으로도 남는 RESIDUAL ZERO 의 구조

위 매트릭스에서 가장 넓게 잡는 조건을 weather mask로 두고, 그래도 남는 zero 행만 분석. 이 잔차가 **순수 outage / shading / metering** 후보.

**3a. Partial-cloud (dc 3~7) 영역에서 잔차 zero 비율** — 평소 normal한 부분 흐림인데 zero라면 outage에 가까움.

**3b. Hour / zenith 패턴** — 일출·일몰 근처(zenith>50)에 몰리면 산 그림자, 정오 영역(zenith<40)에도 분포하면 시스템 outage.

**3c. 날짜 클러스터** — 한 날짜에 9시간 통째로 zero인 사례가 많으면 정비/계측 오류.
""")

code("""# 가장 관대한 weather 조건
weather_rule = lambda d: ((d.dc10Tca>=5) | (d.dsr_mean<400) | (d.rain>0))

residual = near0[~weather_rule(near0)].copy()
print(f'RESIDUAL ZERO (relaxed weather rule로도 설명 안 됨): {len(residual)} 행')
print(f'  → 예천 daytime의 {len(residual)/len(day)*100:.2f}% / near-zero의 {len(residual)/n_n0*100:.2f}%')

print('\\n=== 3a. RESIDUAL ZERO의 dc10Tca 분포 ===')
print(residual.dc10Tca.value_counts().sort_index().to_string())

print('\\n=== 3b. hour 분포 ===')
print(residual.datetime_kst.dt.hour.value_counts().sort_index().to_string())

print('\\n=== 3b. zenith 분포 (5° bin) ===')
zb = pd.cut(residual.zenith_center, np.arange(0, 65, 5))
print(zb.value_counts().sort_index().to_string())

print('\\n=== 3c. 연도/월 분포 ===')
print('\\n  연도:')
print(residual.datetime_kst.dt.year.value_counts().sort_index().to_string())
print('\\n  월:')
print(residual.datetime_kst.dt.month.value_counts().sort_index().to_string())

print('\\n=== 3c. 날짜 클러스터 (한 날에 N시간 zero) — top 20 ===')
date_cnt = residual.datetime_kst.dt.date.value_counts().head(20)
print(date_cnt.to_string())

print('\\n  분포: 하루에 zero 행 1개 / 2~4개 / 5~9개 가 각각 몇 날?')
days_groups = residual.datetime_kst.dt.date.value_counts()
print(f'  1시간만 zero인 날: {(days_groups==1).sum()} 일')
print(f'  2~4 시간 zero인 날: {((days_groups>=2)&(days_groups<=4)).sum()} 일')
print(f'  5~9 시간 zero인 날: {(days_groups>=5).sum()} 일  ← 통째 outage 의심')
""")

md("""→ **모델 함의**:
- hour/zenith 분포가 평탄(9~17시 고르게) → 시간 의존 shading 아님
- 한 날에 5~9시간 통째 zero 사례가 다수 → **운영/정비 outage 신호**
- 특정 연도(2023, 2025)에 집중 → **연도 단위 외부 사건** (장비 교체, 계측 변경, 정비 시즌)
""")

md("""## 4. 예천 NMAE 기여 분해

예천 ensemble NMAE = 8.12% 중 어느 cf 영역이 가장 많이 기여하는지.

**분해**: error = |cf - cf_pred|, contribution = (error * cap) / (sum_all_cap) * 100. cf 구간별로 합산.

**판독**:
- near-zero 영역(cf<0.05) 기여가 절반 이상 → 처치는 **outage masking / 평가 분리** 1순위
- high-output(cf>0.5) 기여가 큼 → 모델 자체의 점수 oversmoothing 문제
- mid 영역이 큼 → partial-cloud 변동성 대응 필요
""")

code("""# 예천 ensemble 예측 불러오기 (Step1 baseline 5-seed ensemble)
ens = pd.read_parquet(ROOT / 'pv/experiments/resmlp_adaln_v2_ensemble/ensemble_test.parquet')
ens_ye = ens[ens.site == '예천'].copy()
ens_ye['err'] = (ens_ye.cf - ens_ye.mu_mean).abs()
ens_ye['err_kwh'] = ens_ye.err * ens_ye.site_capacity_kw

cap_total_test = ens['site_capacity_kw'].sum()  # all sites total cap (분모 일관성)
cap_ye_total = ens_ye['site_capacity_kw'].sum()
print(f'예천 test rows: {len(ens_ye)}')
print(f'예천 site NMAE: {ens_ye.err_kwh.sum()/cap_ye_total*100:.3f}%')
print(f'예천이 전체 NMAE에 기여: {ens_ye.err_kwh.sum()/cap_total_test*100:.3f}% (전체 분모 기준)')

print()
print('=== cf 구간별 분해 (예천만, 분모는 예천 cap 합계) ===')
edges = [0, 0.05, 0.10, 0.20, 0.40, 0.60, 1.5]
labels = ['near-zero[0,0.05)', 'low[0.05,0.10)', 'low-mid[0.10,0.20)',
           'mid[0.20,0.40)', 'high-mid[0.40,0.60)', 'high[0.60+)']
ens_ye['cf_bin'] = pd.cut(ens_ye.cf, edges, labels=labels, right=False)
print(f\"{'bin':<22} {'n':>5} {'mean_cf':>8} {'mean_err':>9} {'sum_err_kwh':>13} {'NMAE_share':>11}\")
for lbl in labels:
    sub = ens_ye[ens_ye.cf_bin == lbl]
    if len(sub) == 0: continue
    sum_err = sub.err_kwh.sum()
    share = sum_err / ens_ye.err_kwh.sum() * 100
    print(f'  {lbl:<22} {len(sub):>5} {sub.cf.mean():>7.3f} {sub.err.mean():>8.3f} {sum_err:>12.0f} {share:>10.2f}%')

# Daytime + near-zero 분리
print()
print('=== daytime(zenith<60)에서 near-zero vs nonzero ===')
import pvlib  # not necessary but easy if zenith already in df
# 직접 zenith 안 보이면 pvlib 대신 hour proxy 쓰자
ens_ye['hour'] = pd.to_datetime(ens_ye['datetime_kst']).dt.hour
# zenith는 ens parquet에 없으니 source df에서 가져와 join
src = ye[['datetime_kst','zenith_center']].copy()
src['datetime_kst'] = pd.to_datetime(src['datetime_kst'])
ens_ye['datetime_kst'] = pd.to_datetime(ens_ye['datetime_kst'])
ens_ye = ens_ye.merge(src, on='datetime_kst', how='left')
day_ye = ens_ye[ens_ye.zenith_center < 60].copy()
near_z = day_ye[day_ye.cf < 0.05]
nonzero = day_ye[day_ye.cf >= 0.05]

# 예천 자체 NMAE 분모
denom = day_ye.site_capacity_kw.sum()
print(f'  daytime rows: {len(day_ye)}')
print(f'  near-zero subset: n={len(near_z)} ({len(near_z)/len(day_ye)*100:.1f}%)')
print(f'    err mean: {near_z.err.mean():.3f}  contribution: {near_z.err_kwh.sum()/denom*100:.3f}%p')
print(f'  non-zero subset:  n={len(nonzero)}')
print(f'    err mean: {nonzero.err.mean():.3f}  contribution: {nonzero.err_kwh.sum()/denom*100:.3f}%p')

# anomaly_zero subset
anom_mask = (day_ye.cf < 0.05) & ~((day_ye.dsr_mean if False else 1) < 0)  # placeholder
# 실제 anomaly mask 재계산 위해 source merge
src2 = ye[['datetime_kst','dc10Tca','dsr_mean','rain']].copy()
src2['datetime_kst'] = pd.to_datetime(src2['datetime_kst'])
day_ye = day_ye.merge(src2, on='datetime_kst', how='left')
day_ye['anomaly_zero'] = (day_ye.cf < 0.05) & ~(
    (day_ye.dc10Tca >= 8) | (day_ye.dsr_mean < 200) | (day_ye.rain >= 0.5))
day_ye['weather_zero'] = (day_ye.cf < 0.05) & (
    (day_ye.dc10Tca >= 8) | (day_ye.dsr_mean < 200) | (day_ye.rain >= 0.5))

print()
print('=== weather-zero vs anomaly-zero NMAE 기여 (daytime) ===')
for lbl, sub in [
    ('weather_zero', day_ye[day_ye.weather_zero]),
    ('anomaly_zero', day_ye[day_ye.anomaly_zero]),
    ('non_zero',     day_ye[(day_ye.cf >= 0.05)]),
]:
    if len(sub)==0: continue
    print(f'  {lbl:<14} n={len(sub):>5} err_mean={sub.err.mean():.3f}  contribution={sub.err_kwh.sum()/denom*100:.3f}%p')
""")

md("""→ **모델 함의**:
- near-zero가 NMAE의 절반 이상 차지 → 그 영역만 처치해도 큰 폭 개선
- 그 안에서 weather_zero vs anomaly_zero 비율이 핵심:
  - weather_zero 비중 큼: 모델이 흐림/비를 더 잘 학습하면 잡힘 (입력 보강)
  - anomaly_zero 비중 큼: 학습/평가 데이터 mask가 정답 (입력으로는 잡을 수 없음)
""")

md("""## 5. 결론 — 예천 처치 분기

분기 규칙:
- (a) near-zero가 NMAE의 50%+ & residual_zero(=anomaly_zero)가 그 안에서 30%+ → **eval mask** 정당화
- (b) near-zero가 50%- & 잔차가 mid/high에 분포 → 모델 자체 문제, 사이트 단독 처치 어려움 (입력 보강 또는 robust loss)
- (c) residual zero가 특정 날짜 클러스터 (5+ 시간 통째) 다수 → **운영 outage 명시**, plan에 외부 이벤트로 기록

이 노트북 결과로 어느 분기를 선택하고, 그 처치 한 가지(예: 예천 anomaly_zero를 eval에서만 제외)를 적용한 후 baseline NMAE 재보고.
""")

nb = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"}
}, "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).parent / "10_yecheon_isolated.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote:", out, "cells:", len(cells))
