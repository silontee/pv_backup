"""
GK-2A GHI 재수집 — 관대 DQF 기준

기존 `data/gk2a_ghi/gk2a_ghi_YYYYMM.csv` (DQF=1만 수집됨)에서
NaN 있는 시각을 모두 찾아 재요청 → 관대 DQF 기준으로 값 복구.

FAIL 원인 + DQF≠1로 버려진 값 모두 통합 처리.
→ 기존 _retry.log 기반 FAIL-only retry 로직을 대체함.

DQF 기준 (공식 ATBD NMSC-SCI-ATBD-INS_v1.0 Table 4):
  ACCEPTABLE = {1,2,3,4}         excellent (맑음/흐림 100%), acceptable (75%)
             ∪ {9,10}             fog, snow (특수 날씨, 값 유효)
             ∪ {11,12}            오존 결측, sun-glint (부정확하지만 사용 가능)
  ZERO       = {13,14}            night (SZA>80), out-of-view (VZA>80) → 0.0
  제외        = {5,6,7,8,15}       bad / uncertainty / 기후 이상치 / 처리 불가

출력 (Long format, DQF 포함):
  data/gk2a_ghi_retry/gk2a_ghi_YYYYMM_retry.csv
    datetime_kst, site, ghi, dqf
  data/gk2a_ghi_retry/_dqf_stats.csv  (누적 DQF 분포 통계)

주의: DQF 컬럼은 **학습 데이터 품질 필터링용**.
모델 feature로 직접 사용하지 않음 (Open-Meteo 예보에 DQF 없어 학습-예보 일관성 위배).
"""

import os
import sys
import csv
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

# 기존 수집기에서 재사용
from crawl_gk2a import (
    API_KEYS,
    SAVE_DIR,
    SITE_PIXELS,
    KST_HOURS,
    NUM_WORKERS,
    MAX_FILE_SIZE,
    SLEEP_BETWEEN_REQ,
    KeyManager,
    kst_to_utc_str,
    download_nc,
)

# ===== 재수집 설정 =====
# 프로젝트 루트 = src/crawl/gk2a/ 에서 3단계 상위
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RETRY_DIR = os.path.join(PROJECT_ROOT, "data", "gk2a_ghi_retry")
BYTES_LIMIT_PER_KEY = int(4.5 * 1024 * 1024 * 1024)  # 4.5GB (기존과 동일)

# 관대 DQF 기준
ACCEPTABLE_DQF = {1, 2, 3, 4, 9, 10, 11, 12}
ZERO_DQF       = {13, 14}
# 제외 = {5, 6, 7, 8, 15} + Unknown

_PRINT_LOCK = threading.Lock()
def tlog(msg):
    with _PRINT_LOCK:
        print(msg, flush=True)


# ============================================================
# DQF 관대 추출
# ============================================================
def extract_ghi_lenient(nc_path):
    """관대 DQF 기준으로 (value, dqf) 반환.

    Returns:
        dict: {site_name: (value_or_None, dqf_or_None)}
    """
    try:
        ds = xr.open_dataset(nc_path)
        dsr = ds["DSR"].values
        dqf = ds["DSR_DQF1"].values
        ds.close()

        results = {}
        for name, row, col in SITE_PIXELS:
            if not (0 <= row < 900 and 0 <= col < 900):
                results[name] = (None, None)
                continue
            val = float(dsr[row, col])
            qf_raw = dqf[row, col]
            qf = int(qf_raw) if not np.isnan(qf_raw) else None

            if np.isnan(val):
                # DQF 13, 14는 value=NaN이어도 물리적 0 처리
                if qf in ZERO_DQF:
                    results[name] = (0.0, qf)
                else:
                    results[name] = (None, qf)
            elif qf in ACCEPTABLE_DQF:
                results[name] = (round(val, 1), qf)
            elif qf in ZERO_DQF:
                results[name] = (0.0, qf)
            else:
                # 5, 6, 7, 8, 15, Unknown → 제외
                results[name] = (None, qf)
        return results
    except Exception as e:
        tlog(f"    extract_ghi_lenient 예외: {e}")
        return {name: (None, None) for name, _, _ in SITE_PIXELS}


# ============================================================
# 기존 CSV에서 NaN 슬롯 탐색
# ============================================================
def load_nan_slots(year, month):
    """기존 CSV에서 NaN(전 사이트 or 일부 사이트) 있는 (day, hour) 리스트 반환.

    - 파일 없으면 None 반환 (원본 수집 안 된 월)
    - 파일 있어도 NaN 없으면 [] 반환
    """
    filepath = os.path.join(SAVE_DIR, f"gk2a_ghi_{year:04d}{month:02d}.csv")
    if not os.path.exists(filepath):
        return None

    df = pd.read_csv(filepath)
    df['datetime_kst'] = pd.to_datetime(df['datetime_kst'])
    site_names = [name for name, _, _ in SITE_PIXELS]

    # 일부라도 NaN인 시각 모두 재요청 대상
    nan_mask = df[site_names].isna().any(axis=1)
    nan_df = df[nan_mask][['datetime_kst']].copy()
    slots = [(dt.day, dt.hour) for dt in nan_df['datetime_kst']]
    return slots


# ============================================================
# 한 시각 재수집
# ============================================================
def collect_retry_hour(year, month, day, hour, km):
    """한 시각의 NC 파일 재수집 + 관대 DQF 적용.

    Returns:
        (ghi_dqf_dict | None, n_bytes, fail_flag)
    """
    key = km.get_usable_key()
    if key is None:
        return None, 0, False  # 키 소진

    utc_str = kst_to_utc_str(year, month, day, hour, 0)
    t0 = time.time()
    nc_path, nbytes = download_nc(utc_str, key)
    elapsed = time.time() - t0

    if nc_path:
        km.add_usage(key, nbytes)
        ghi_dqf = extract_ghi_lenient(nc_path)
        os.unlink(nc_path)
        tlog(f"    {year:04d}-{month:02d}-{day:02d} {hour:02d}:00 OK  "
             f"{nbytes/1024:.0f}KB / {elapsed:.1f}s [{key[:8]}]")
        time.sleep(SLEEP_BETWEEN_REQ)
        return ghi_dqf, nbytes, False
    else:
        tlog(f"    {year:04d}-{month:02d}-{day:02d} {hour:02d}:00 FAIL "
             f"({elapsed:.1f}s) [{key[:8]}]")
        time.sleep(SLEEP_BETWEEN_REQ)
        return None, 0, True


# ============================================================
# Long format CSV 저장
# ============================================================
def save_long(records, filepath):
    """Long format: datetime_kst, site, ghi, dqf"""
    # 시각·사이트 순 정렬
    records_sorted = sorted(records, key=lambda r: (r['datetime_kst'], r['site']))
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['datetime_kst', 'site', 'ghi', 'dqf'])
        writer.writeheader()
        for r in records_sorted:
            writer.writerow({
                'datetime_kst': r['datetime_kst'],
                'site': r['site'],
                'ghi': '' if r['ghi'] is None else r['ghi'],
                'dqf': '' if r['dqf'] is None else r['dqf'],
            })


def log_dqf_stats(stats_file, ym, dqf_counter):
    """누적 DQF 분포 통계 append."""
    header_needed = not os.path.exists(stats_file)
    with open(stats_file, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(['ym', 'dqf', 'count'])
        for dqf, count in sorted(dqf_counter.items(),
                                  key=lambda x: (x[0] is None, x[0] if x[0] is not None else 999)):
            label = str(dqf) if dqf is not None else 'null'
            writer.writerow([ym, label, count])


# ============================================================
# 월 단위 재수집
# ============================================================
def collect_month_retry(year, month, km):
    os.makedirs(RETRY_DIR, exist_ok=True)
    output_file = os.path.join(RETRY_DIR, f"gk2a_ghi_{year:04d}{month:02d}_retry.csv")
    stats_file = os.path.join(RETRY_DIR, "_dqf_stats.csv")

    if os.path.exists(output_file):
        print(f"  건너뜀: {os.path.basename(output_file)} (이미 재수집 완료)")
        return True

    slots = load_nan_slots(year, month)
    if slots is None:
        print(f"  원본 CSV 없음: {year}-{month:02d} — 스킵 (먼저 crawl_gk2a.py 수집 필요)")
        return True
    if len(slots) == 0:
        print(f"  NaN 없음: {year}-{month:02d} — 스킵")
        # 빈 파일이라도 생성 (재수집 완료 표시)
        save_long([], output_file)
        return True

    site_names = [name for name, _, _ in SITE_PIXELS]
    print(f"  재수집 대상 슬롯: {len(slots)}개 (일부라도 NaN인 시각 모두)")

    records = []
    dqf_counter = Counter()
    fail_count = 0
    total_bytes = 0
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(collect_retry_hour, year, month, d, h, km): (d, h)
                   for d, h in slots}

        for fut in as_completed(futures):
            d, h = futures[fut]
            try:
                ghi_dqf, nbytes, fail = fut.result()
            except Exception as e:
                tlog(f"  !! {year}-{month:02d}-{d:02d} {h:02d}시 예외: {e}")
                fail_count += 1
                continue

            if ghi_dqf is None:
                # 키 소진 or FAIL
                if fail:
                    fail_count += 1
                continue

            # 레코드 생성
            dt_str = f"{year:04d}-{month:02d}-{d:02d} {h:02d}:00"
            for name in site_names:
                val, qf = ghi_dqf.get(name, (None, None))
                records.append({
                    'datetime_kst': dt_str,
                    'site': name,
                    'ghi': val,
                    'dqf': qf,
                })
                dqf_counter[qf] += 1
            total_bytes += nbytes
            completed += 1

            # 100 슬롯마다 증분 저장
            if completed % 100 == 0:
                save_long(records, output_file)
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remain = (len(slots) - completed) / rate if rate > 0 else 0
                tlog(f"  [{completed:4d}/{len(slots)}] "
                     f"{completed*100/len(slots):.1f}% | "
                     f"경과 {elapsed/60:.1f}분 | 남은예상 {remain/60:.1f}분 | "
                     f"FAIL {fail_count} | 사용 {km.total_usage_gb():.2f}GB")

    # 최종 저장
    save_long(records, output_file)
    log_dqf_stats(stats_file, f"{year:04d}{month:02d}", dqf_counter)

    # 월 요약
    print(f"\n  {year}-{month:02d} 완료: {completed}/{len(slots)} 성공, FAIL {fail_count}")
    print(f"  DQF 분포 (전 사이트 기준):")
    for dqf, cnt in sorted(dqf_counter.items(),
                            key=lambda x: (x[0] is None, x[0] if x[0] is not None else 999)):
        if dqf is None:
            label = 'null (bad/unavail/읽기실패)'
        else:
            kind = ('excellent' if dqf in {1,2} else
                    'acceptable' if dqf in {3,4} else
                    'special' if dqf in {9,10,11,12} else
                    'zero(night/outside)' if dqf in {13,14} else
                    'excluded')
            label = f'DQF={dqf} ({kind})'
        print(f"    {label}: {cnt}")
    return True


# ============================================================
# 월 순회
# ============================================================
def iter_months(start_year, start_month, end_year, end_month, reverse=False):
    months = []
    for y in range(start_year, end_year + 1):
        sm = start_month if y == start_year else 1
        em = end_month if y == end_year else 12
        for m in range(sm, em + 1):
            months.append((y, m))
    if reverse:
        months.reverse()
    return months


if __name__ == "__main__":
    km = KeyManager(API_KEYS, BYTES_LIMIT_PER_KEY)

    start_year, start_month = 2022, 1
    end_year, end_month = 2025, 12
    reverse = False

    if len(sys.argv) >= 3:
        start_year = int(sys.argv[1])
        start_month = int(sys.argv[2])
        if len(sys.argv) >= 5:
            end_year = int(sys.argv[3])
            end_month = int(sys.argv[4])
        if len(sys.argv) >= 6:
            reverse = sys.argv[5].lower() in ("reverse", "desc", "true", "1")

    print(f"=== GK-2A GHI 재수집 (관대 DQF) ===")
    print(f"  기간: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d} "
          f"({'최근→과거' if reverse else '과거→최근'})")
    print(f"  대상: 기존 CSV의 NaN 시각 (FAIL + DQF≠1 통합 처리)")
    print(f"  DQF 기준:")
    print(f"    유효 (값 사용): {sorted(ACCEPTABLE_DQF)}")
    print(f"    물리적 0 처리: {sorted(ZERO_DQF)}")
    print(f"    제외 (None): 5, 6, 7, 8, 15, Unknown")
    print(f"  출력: {RETRY_DIR}/gk2a_ghi_YYYYMM_retry.csv (long format)")
    print(f"  통계: {RETRY_DIR}/_dqf_stats.csv (누적)")
    print()

    for y, m in iter_months(start_year, start_month, end_year, end_month, reverse):
        print(f"\n=== {y}-{m:02d} ===")
        ok = collect_month_retry(y, m, km)
        if not ok:
            print(f"\n!! 키 소진으로 중단. 사용량: {km.total_usage_gb():.2f}GB")
            sys.exit(1)

    print(f"\n=== 재수집 완료! 총 사용량: {km.total_usage_gb():.2f}GB ===")
