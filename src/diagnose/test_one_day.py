"""하루치 수집 테스트 — 15시간 × 10분 6장 = 90 파일 평균 1시간/호기

실행: uv run python test_one_day.py [YYYY] [MM] [DD]
기본값: 2025-12-15
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

from crawl_gk2a import (
    KeyManager,
    API_KEYS,
    BYTES_LIMIT_PER_KEY,
    collect_hour,
    SITE_PIXELS,
    KST_HOURS,
    NUM_WORKERS,
    _save_csv,
)
import os

SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "gk2a_ghi")

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
MONTH = int(sys.argv[2]) if len(sys.argv) > 2 else 12
DAY = int(sys.argv[3]) if len(sys.argv) > 3 else 15

km = KeyManager(API_KEYS, BYTES_LIMIT_PER_KEY)
site_names = [n for n, _, _ in SITE_PIXELS]

print(f"=== 하루 테스트: {YEAR}-{MONTH:02d}-{DAY:02d} ===")
print(f"  시간 {KST_HOURS[0]:02d}~{KST_HOURS[-1]:02d}시, 10분당 1장 × 6 = {len(KST_HOURS)*6} 파일")
print(f"  병렬 워커 {NUM_WORKERS}개")
print()

start = time.time()
results = {}

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    futures = {ex.submit(collect_hour, YEAR, MONTH, DAY, h, km, site_names): h for h in KST_HOURS}
    for fut in as_completed(futures):
        h = futures[fut]
        try:
            avg, fc, nb = fut.result()
        except Exception as e:
            print(f"  !! {h:02d}시 예외: {e}")
            continue
        if avg is None:
            print(f"  !! {h:02d}시 키 소진")
            continue
        results[h] = avg
        ok_sites = sum(1 for v in avg.values() if v is not None)
        print(f"  {h:02d}시 완료: 성공 {ok_sites}/{len(site_names)}, 실패 샘플 {fc}/6")

total = time.time() - start
print(f"\n총 소요: {total:.1f}초 ({total/60:.1f}분)")
print(f"성공 시간: {len(results)}/{len(KST_HOURS)}")
print(f"사용 쿼터: {km.total_usage_gb()*1024:.2f} MB")

# CSV 저장
if results:
    filename = f"gk2a_ghi_{YEAR:04d}{MONTH:02d}{DAY:02d}_test.csv"
    filepath = os.path.join(SAVE_DIR, filename)
    rows = []
    for h in KST_HOURS:
        row = {"datetime_kst": f"{YEAR:04d}-{MONTH:02d}-{DAY:02d} {h:02d}:00"}
        avg = results.get(h)
        for name in site_names:
            row[name] = avg.get(name) if avg else None
        rows.append(row)
    _save_csv(filepath, site_names, rows)
    print(f"저장: {filepath}")
