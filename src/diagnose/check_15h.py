"""15시 FAIL이 어느 월에 몰렸는지, 전체 기간 퍼졌는지 조사."""
import sys, os, re, glob
import pandas as pd
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'D:\pv_backup')

# 로그에서 15시 FAIL만 추출
logs = ['data/gk2a_ghi/_collect.log', 'data/gk2a_ghi/_retry.log']
pat = re.compile(r'(\d{4})-(\d{2})-(\d{2})\s+15:\d{2}\s+(OK|FAIL)')
fail_15_ym = Counter()
ok_15_ym = Counter()
for lg in logs:
    if not os.path.exists(lg): continue
    with open(lg, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pat.search(line)
            if m:
                y, mo, d, st = m.groups()
                ym = f'{y}{mo}'
                if st == 'FAIL':
                    fail_15_ym[ym] += 1
                else:
                    ok_15_ym[ym] += 1

print("=== 15시 FAIL 월별 분포 (로그 기반) ===")
print(f"{'ym':<8} {'OK':>5} {'FAIL':>5} {'FAIL%':>7}")
all_ym = sorted(set(list(fail_15_ym.keys()) + list(ok_15_ym.keys())))
for ym in all_ym:
    ok = ok_15_ym.get(ym, 0)
    fl = fail_15_ym.get(ym, 0)
    tot = ok + fl
    pct = 100*fl/max(tot,1)
    marker = ' ⚠️' if pct > 30 else ''
    print(f"{ym:<8} {ok:>5} {fl:>5} {pct:>6.1f}%{marker}")
print(f"\n합계: OK {sum(ok_15_ym.values())}, FAIL {sum(fail_15_ym.values())}")

# 집중된 월 식별
concentrated = [(ym, fl) for ym, fl in fail_15_ym.items() if fl > 30]
concentrated.sort(key=lambda x: -x[1])
print(f"\n=== 15시 FAIL 30건 이상 몰린 월 ===")
for ym, fl in concentrated:
    print(f"  {ym}: {fl}건")

# CSV에서 15시 NaN 분포
print("\n\n=== CSV에서 15시 NaN 비율 (월별) ===")
SITES = ['고흥만수상','삼천포','영흥','광양항세방','예천','영동','탑선','구미','경상대','여수','창원']
files = sorted(glob.glob('data/gk2a_ghi/gk2a_ghi_*.csv'))
rows = []
for f in files:
    ym = os.path.basename(f).replace('gk2a_ghi_','').replace('.csv','')
    d = pd.read_csv(f)
    d['datetime_kst'] = pd.to_datetime(d['datetime_kst'])
    d['hour'] = d['datetime_kst'].dt.hour
    d15 = d[d['hour']==15]
    if len(d15) == 0: continue
    n_cells = len(d15) * len(SITES)
    n_nan = d15[SITES].isna().values.sum()
    rows.append({
        'ym': ym,
        'days': len(d15),
        'nan_cells': n_nan,
        'total_cells': n_cells,
        'nan%': round(100*n_nan/max(n_cells,1), 1),
        'log_fail%': round(100*fail_15_ym.get(ym,0)/max(ok_15_ym.get(ym,0)+fail_15_ym.get(ym,0),1), 1),
    })
report = pd.DataFrame(rows)

# 정상 월 vs 문제 월
print(report.to_string(index=False))
print(f"\n15시 NaN% < 20%인 월: {(report['nan%']<20).sum()}/{len(report)}")
print(f"15시 NaN% < 5%인 월: {(report['nan%']<5).sum()}/{len(report)}")

# 2024년 15시 일별 NaN
print("\n\n=== 2024년 15시 NaN 일별 (어떤 날이 문제인지) ===")
files_2024 = sorted(glob.glob('data/gk2a_ghi/gk2a_ghi_2024*.csv'))
d2024 = pd.concat([pd.read_csv(f) for f in files_2024], ignore_index=True)
d2024['datetime_kst'] = pd.to_datetime(d2024['datetime_kst'])
d2024['hour'] = d2024['datetime_kst'].dt.hour
d2024['date'] = d2024['datetime_kst'].dt.date
d2024_15 = d2024[d2024['hour']==15].copy()

# 15시에 전 사이트 NaN인 날 (=완전 결측)
d2024_15['all_nan'] = d2024_15[SITES].isna().all(axis=1)
d2024_15['any_valid'] = ~d2024_15['all_nan']
full_loss_days = d2024_15[d2024_15['all_nan']]['date'].tolist()
print(f"2024년 15시 데이터 전혀 없는 날: {len(full_loss_days)}일 / 366일")
if len(full_loss_days) <= 30:
    print(f"해당 날짜: {full_loss_days}")
else:
    print(f"해당 날짜 (처음 20): {full_loss_days[:20]}")
    print(f"... 외 {len(full_loss_days)-20}개")

# 월별
d2024_15['month'] = d2024_15['datetime_kst'].dt.month
print(f"\n2024년 월별 15시 전체-NaN 날 수:")
print(d2024_15.groupby('month')['all_nan'].sum().to_string())
