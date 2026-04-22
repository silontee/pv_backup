"""호기별 데이터 품질 점검 — §3 이상치 원인 추적."""
import sys
import pandas as pd
import numpy as np
import glob

sys.stdout.reconfigure(encoding='utf-8')

# 하드코딩 용량 (노트북과 동일)
UNIT_META = {
    '경상대태양광#1':       dict(capacity_kw=905,   stnNm='진주'),
    '고흥만 수상태양광#1':    dict(capacity_kw=63481, stnNm='고흥'),
    '광양항세방태양광#1':    dict(capacity_kw=2993,  stnNm='광양시'),
    '구미태양광#1':         dict(capacity_kw=992,   stnNm='구미'),
    '두산엔진MG태양광#1':    dict(capacity_kw=77,    stnNm='창원'),
    '삼천포태양광#1':       dict(capacity_kw=1097,  stnNm='남해'),
    '삼천포태양광#2':       dict(capacity_kw=1097,  stnNm='남해'),
    '삼천포태양광#3':       dict(capacity_kw=1097,  stnNm='남해'),
    '영흥태양광#1':         dict(capacity_kw=500,   stnNm='인천'),
    '영흥태양광#2':         dict(capacity_kw=500,   stnNm='인천'),
    '영흥태양광#5#1':       dict(capacity_kw=3500,  stnNm='인천'),
    '예천태양광#1':         dict(capacity_kw=2000,  stnNm='안동'),
}

import os
os.chdir(r'D:\pv_backup')
pv = pd.read_csv('data/processed/solar_hourly_long.csv', parse_dates=['date','datetime'])
pv['unit'] = pv['unit'].astype(str)
pv['site_unit'] = pv['site'] + '#' + pv['unit']
pv24 = pv[pv['date'].dt.year == 2024].copy()

print("=" * 100)
print("1. 호기별 2024년 gen_kwh 기본 통계")
print("=" * 100)
print(f"{'site_unit':<25} {'cap_kW':>8} {'n행':>7} {'결측%':>7} "
      f"{'max_h':>10} {'max_CF':>8} {'mean_h':>10} {'mean_CF':>8} "
      f"{'일합 max':>10} {'일합 CF상한':>12}")
print("-" * 100)

for su, meta in UNIT_META.items():
    cap = meta['capacity_kw']
    g = pv24[pv24['site_unit']==su]['gen_kwh']
    n_total = len(g)
    n_nan = g.isna().sum()
    n_zero = (g == 0).sum()
    max_h = g.max()
    max_cf = max_h / cap
    mean_h = g.mean()
    mean_cf = mean_h / cap

    # 일별 합계
    day_total = pv24[pv24['site_unit']==su].groupby('date')['gen_kwh'].sum()
    max_daily = day_total.max() if len(day_total) > 0 else np.nan
    # 이론 일합상한: cap × 24h (절대) = cap*24 kWh. 실제로는 cap×peak_sun ~ cap×4~6h
    # CF 일일 상한 = max_daily / (cap × 24) ... 극한 CF = 1 이면 24시간 최대출력
    daily_cf_ceiling = max_daily / (cap * 24) if not np.isnan(max_daily) else np.nan

    print(f"{su:<25} {cap:>8} {n_total:>7} {100*n_nan/max(n_total,1):>6.1f}% "
          f"{max_h:>10.1f} {max_cf:>8.3f} {mean_h:>10.2f} {mean_cf:>8.3f} "
          f"{max_daily:>10.1f} {daily_cf_ceiling:>12.3f}")

print("\n지침: max_CF > 1.0 이면 용량 오류 or 데이터 오류 (시간당 출력이 정격 초과)")
print("      일합 CF상한 > 0.3 이면 비정상 (일평균 하루 CF가 30% 넘으면 의심)")

# ===== 영흥 #1, #2 상세 =====
print("\n" + "=" * 100)
print("2. 영흥 #1, #2 상세 — CF>1 원인 추적")
print("=" * 100)
for su in ['영흥태양광#1', '영흥태양광#2']:
    sub = pv24[pv24['site_unit']==su].copy()
    print(f"\n--- {su} (하드코딩 용량 {UNIT_META[su]['capacity_kw']} kW) ---")
    print(f"총 행: {len(sub):,}, 0이 아닌 시간: {(sub['gen_kwh']>0).sum():,}")
    print(f"\ngen_kwh 분위수:")
    print(sub['gen_kwh'].describe(percentiles=[0.5, 0.9, 0.95, 0.99, 0.999]).round(2).to_string())

    # 상위 10개 시간 데이터
    print(f"\n상위 5 시간 (gen_kwh 큰 순):")
    top = sub.nlargest(5, 'gen_kwh')[['datetime','hour','gen_kwh']]
    print(top.to_string(index=False))

    # 일합 상위 5일
    daily = sub.groupby('date')['gen_kwh'].sum().sort_values(ascending=False)
    print(f"\n일합 상위 5일 (이론상 용량 {UNIT_META[su]['capacity_kw']} kW × ~5h = ~{UNIT_META[su]['capacity_kw']*5} kWh 상한):")
    print(daily.head().to_string())

# ===== 삼천포 #1, #3 상세 =====
print("\n" + "=" * 100)
print("3. 삼천포 #1, #3 상세 — 저발전/결측 원인 추적")
print("=" * 100)
for su in ['삼천포태양광#1', '삼천포태양광#3']:
    sub = pv24[pv24['site_unit']==su].copy()
    print(f"\n--- {su} (하드코딩 용량 {UNIT_META[su]['capacity_kw']} kW) ---")
    print(f"총 행: {len(sub):,}, 0이 아닌 시간: {(sub['gen_kwh']>0).sum():,}")

    # 월별 총발전량
    sub['month'] = sub['date'].dt.month
    monthly = sub.groupby('month')['gen_kwh'].sum()
    print(f"\n월별 총 gen_kwh:")
    print(monthly.to_string())

    # 발전이 있었던 날 수
    active_days = sub.groupby('date')['gen_kwh'].sum().gt(0).sum()
    total_days = sub['date'].nunique()
    print(f"\n발전 있는 날: {active_days}/{total_days} ({100*active_days/max(total_days,1):.1f}%)")

# ===== ASOS icsr 커버리지 (NaN 호기 확인) =====
print("\n" + "=" * 100)
print("4. ASOS icsr 커버리지 — 남해/구미 미관측 확인")
print("=" * 100)
asos_files = sorted(glob.glob('data/asos_hourly/asos_hourly_2024*.csv'))
asos = pd.concat([pd.read_csv(f) for f in asos_files], ignore_index=True)

for stn in ['진주','고흥','광양시','구미','창원','남해','인천','안동']:
    s = asos[asos['stnNm']==stn]
    if len(s) == 0:
        print(f"  {stn:<6}: 데이터 없음")
        continue
    icsr_valid = s['icsr'].notna().sum()
    print(f"  {stn:<6}: 전체 {len(s):>5}행, icsr 유효 {icsr_valid:>5} ({100*icsr_valid/len(s):.1f}%)")

# solar_sites.csv 참조
print("\n" + "=" * 100)
print("5. solar_sites.csv 용량 vs UNIT_META 하드코딩 비교")
print("=" * 100)
try:
    sites = pd.read_csv('data/solar_sites.csv')
    print(sites[['site_name','capacity_kw','region']].to_string(index=False))
except Exception as e:
    print(f"로드 실패: {e}")
