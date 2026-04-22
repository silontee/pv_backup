"""GK-2A GHI 월별 완성도 점검 (Wide 포맷: datetime_kst + 사이트 컬럼)."""
import sys, os, glob
import pandas as pd
import numpy as np
from calendar import monthrange

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'D:\pv_backup')

files = sorted(glob.glob('data/gk2a_ghi/gk2a_ghi_*.csv'))
print(f"총 파일: {len(files)}개\n")

SITES = ['고흥만수상','삼천포','영흥','광양항세방','예천','영동','탑선','구미','경상대','여수','창원']

# 월별 요약
rows = []
for fp in files:
    fn = os.path.basename(fp)
    ym = fn.replace('gk2a_ghi_','').replace('.csv','')
    yr, mo = int(ym[:4]), int(ym[4:])
    days_in_month = monthrange(yr, mo)[1]

    df = pd.read_csv(fp)
    df['datetime_kst'] = pd.to_datetime(df['datetime_kst'])
    df['date'] = df['datetime_kst'].dt.date
    df['hour'] = df['datetime_kst'].dt.hour

    total_rows = len(df)
    # 유효 값 여부: 사이트 컬럼 중 하나라도 non-NaN + > 0
    site_cols = [c for c in SITES if c in df.columns]

    # 시간별 "유효 hour": 적어도 하나의 사이트에서 non-NaN
    has_any_value = df[site_cols].notna().any(axis=1)
    hours_with_data = has_any_value.sum()

    # 일별: 적어도 하나 시간에서 non-NaN 있음
    days_with_data = df.loc[has_any_value, 'date'].nunique()

    # 사이트별 유효 시간 비율
    site_valid_pct = {c: 100 * df[c].notna().sum() / max(total_rows,1) for c in site_cols}

    rows.append({
        'ym': ym,
        'days_in_month': days_in_month,
        'days_with_data': days_with_data,
        '완성%': 100*days_with_data/days_in_month if days_in_month else 0,
        'total_rows': total_rows,
        'hours_with_any': hours_with_data,
        **{f'{c}%': round(site_valid_pct[c], 1) for c in site_cols},
    })

report = pd.DataFrame(rows)
print("=== 월별 완성도 요약 ===")
print(report[['ym','days_in_month','days_with_data','완성%','total_rows','hours_with_any']].to_string(index=False))
print()

# 완전 월 (모든 날에 데이터 있음)
complete = report[report['days_with_data'] == report['days_in_month']]
print(f"\n=== 일자 기준 완전 월 (days_with_data == days_in_month) ===")
print(f"완전: {len(complete)}/{len(report)} 개월")
if len(complete):
    print(f"목록: {', '.join(complete['ym'].tolist())}")

# 거의 완전 월 (≥95%)
near = report[(report['완성%'] >= 95) & (report['완성%'] < 100)]
print(f"\n=== 거의 완전 월 (95~99%) ===")
print(f"개수: {len(near)}")
if len(near):
    print(near[['ym','days_with_data','days_in_month','완성%']].to_string(index=False))

# 부분 월 (<95%)
low = report[report['완성%'] < 95]
print(f"\n=== 부분/불완전 월 (<95%) ===")
print(f"개수: {len(low)}")
if len(low):
    print(low[['ym','days_with_data','days_in_month','완성%']].to_string(index=False))

# 사이트별 평균 유효 비율
print(f"\n=== 사이트별 평균 유효값 비율 (낮 시간 중) ===")
site_means = report.filter(regex='%$').drop(columns=['완성%']).mean().sort_values(ascending=False)
print(site_means.round(1).to_string())
print("\n주의: 50% 근처가 정상 (낮 15시간 중 절반은 해 뜨기 전/진 뒤의 낮은 값으로 NaN 처리될 수 있음)")

# 연도별 요약
report['year'] = report['ym'].str[:4]
yearly = report.groupby('year').agg(
    months=('ym','count'),
    complete=('완성%', lambda s: (s==100).sum()),
    avg_pct=('완성%','mean'),
)
print("\n=== 연도별 요약 ===")
print(yearly.round(1).to_string())
