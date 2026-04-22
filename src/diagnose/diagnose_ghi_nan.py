"""GK-2A NaN 원인 분해 — 진짜 FAIL vs 저각도 vs 품질플래그."""
import sys, os, re, glob
import pandas as pd
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'D:\pv_backup')

# ===== 1. 로그 분석 (FAIL vs OK per month/hour) =====
logs = ['data/gk2a_ghi/_collect.log', 'data/gk2a_ghi/_retry.log']
fail_by_ym_hour = Counter()
ok_by_ym_hour = Counter()

# 로그 형식:
#   YYYY-MM-DD HH:MM OK  ...
#   YYYY-MM-DD HH:MM FAIL ...
log_pat = re.compile(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})\s+(OK|FAIL)')

for lg in logs:
    if not os.path.exists(lg): continue
    with open(lg, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = log_pat.search(line)
            if m:
                y, mo, d, hr, mn, status = m.groups()
                ym = f'{y}{mo}'
                if status == 'FAIL':
                    fail_by_ym_hour[(ym, int(hr))] += 1
                else:
                    ok_by_ym_hour[(ym, int(hr))] += 1

print("=== 로그 기반 요청 통계 (retry 로그 포함) ===")
total_fail = sum(fail_by_ym_hour.values())
total_ok = sum(ok_by_ym_hour.values())
print(f"OK 총 {total_ok:,} | FAIL 총 {total_fail:,}")
print(f"FAIL 비율: {100*total_fail/max(total_fail+total_ok,1):.2f}%")

# 월별 FAIL 집계
fail_by_ym = Counter()
ok_by_ym = Counter()
for (ym, h), c in fail_by_ym_hour.items(): fail_by_ym[ym] += c
for (ym, h), c in ok_by_ym_hour.items():   ok_by_ym[ym] += c
print(f"\n월별 FAIL 분포 (상위 10):")
for ym, c in sorted(fail_by_ym.items(), key=lambda x: -x[1])[:10]:
    total = c + ok_by_ym.get(ym, 0)
    print(f"  {ym}: FAIL {c:4d} / 요청 {total:5d} ({100*c/max(total,1):.1f}%)")

# 시간대별 FAIL 비율
print(f"\n시간대별 FAIL 비율 (전체 기간):")
for h in range(6, 21):
    f = sum(c for (ym, hh), c in fail_by_ym_hour.items() if hh == h)
    o = sum(c for (ym, hh), c in ok_by_ym_hour.items()   if hh == h)
    total = f + o
    print(f"  {h:02d}시: FAIL {f:5d} / 요청 {total:6d} ({100*f/max(total,1):5.2f}%)")

# ===== 2. CSV NaN 분포 (시간대별) =====
print("\n\n=== CSV NaN 분포 — 시간대별 (2024년 기준) ===")
files = sorted(glob.glob('data/gk2a_ghi/gk2a_ghi_2024*.csv'))
dfs = []
for f in files:
    d = pd.read_csv(f)
    d['datetime_kst'] = pd.to_datetime(d['datetime_kst'])
    dfs.append(d)
ghi = pd.concat(dfs, ignore_index=True)
ghi['hour'] = ghi['datetime_kst'].dt.hour

SITES = ['고흥만수상','삼천포','영흥','광양항세방','예천','영동','탑선','구미','경상대','여수','창원']

# 시간대별 NaN%, zero%, mean
print(f"{'hour':<5} {'전체행':<8} {'NaN비율':<12} {'0이하 비율':<12} {'평균 W/m²':<10}")
for h in sorted(ghi['hour'].unique()):
    sub = ghi[ghi['hour']==h]
    # 전 사이트 flatten
    vals = sub[SITES].values.flatten()
    n = len(vals)
    n_nan = np.isnan(vals).sum()
    n_zero = (vals <= 0).sum() if n - n_nan > 0 else 0
    mean = np.nanmean(vals)
    print(f"  {h:02d}시  {n:>7,}  {100*n_nan/max(n,1):>6.1f}%     "
          f"{100*n_zero/max(n-n_nan,1):>6.1f}%       {mean:>8.1f}")

print("\n해석 기준")
print("  - 6, 7, 19, 20시 등 일출/일몰 경계 NaN% 높음 → 저각도 물리적 한계 (진짜 NaN 아님)")
print("  - 12시 근처 NaN% 높음 → API FAIL 또는 품질플래그 문제 (진짜 NaN)")

# ===== 3. 월별 CSV NaN% vs 로그 FAIL% 비교 =====
print("\n\n=== 월별: CSV NaN% vs 로그 FAIL% (전체 기간) ===")
all_files = sorted(glob.glob('data/gk2a_ghi/gk2a_ghi_*.csv'))
rows = []
for f in all_files:
    ym = os.path.basename(f).replace('gk2a_ghi_','').replace('.csv','')
    d = pd.read_csv(f)
    n_cells = len(d) * len(SITES)
    n_nan = d[SITES].isna().values.sum()
    nan_pct = 100 * n_nan / max(n_cells, 1)

    log_fail = fail_by_ym.get(ym, 0)
    log_ok = ok_by_ym.get(ym, 0)
    log_total = log_fail + log_ok
    fail_pct = 100 * log_fail / max(log_total, 1) if log_total else None

    rows.append({'ym': ym, 'csv_rows': len(d),
                 'csv_NaN%': round(nan_pct, 1),
                 '로그 요청': log_total,
                 '로그 FAIL%': round(fail_pct, 2) if fail_pct is not None else None,
                 'NaN − FAIL': round(nan_pct - (fail_pct or 0), 1)})

report = pd.DataFrame(rows)
print(report.to_string(index=False))
print("\n해석")
print("  - 'NaN − FAIL' 값이 크면 → FAIL 외 추가 NaN (저각도 + DQF 실패)")
print("  - 전 시간대 FAIL%가 0에 가깝고 NaN%만 크면 → 대부분 저각도 NaN (실제로는 0에 가까움)")
print("  - FAIL% 높은 월은 수집 재시도 필요")
