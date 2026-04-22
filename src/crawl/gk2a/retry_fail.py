"""GK-2A FAIL 재수집기

기존 CSV에서 모든 사이트가 **빈 문자열("")** 인 row만 재시도 대상.
"NaN" 문자열은 KMA 실제 no-data → 건너뜀.
원본 CSV는 in-place 업데이트.
"""
import os
import sys
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

from crawl_gk2a import (
    KeyManager,
    API_KEYS,
    BYTES_LIMIT_PER_KEY,
    collect_hour,
    SITE_PIXELS,
    NUM_WORKERS,
    _save_csv,
    _tlog,
)
import glob

SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "gk2a_ghi")
SITE_NAMES = [n for n, _, _ in SITE_PIXELS]


def find_missing(csv_path):
    """CSV에서 모든 사이트가 **빈 문자열("")** 인 row 찾기 (FAIL 만).
    "NaN" 은 KMA 실데이터 → 재시도 대상 아님.
    """
    missing = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # FAIL 만: 모든 사이트 컬럼이 "" (빈 문자열)
            all_fail = all(row.get(name, "") == "" for name in SITE_NAMES)
            if all_fail:
                # datetime_kst: '2022-01-03 12:00'
                dt = datetime.strptime(row["datetime_kst"], "%Y-%m-%d %H:%M")
                missing.append((dt.year, dt.month, dt.day, dt.hour))
    return missing


def update_csv(csv_path, results):
    """CSV에 새로 받은 결과 병합. results: {(y,m,d,h): {site: val}}
    val 타입:
      - float   → 해당 값으로 업데이트
      - "NaN"   → "NaN" 문자열로 업데이트 (KMA 실제 no-data)
      - None    → 업데이트 안 함 (재시도도 FAIL → 기존 "" 유지)
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = datetime.strptime(row["datetime_kst"], "%Y-%m-%d %H:%M")
            key = (dt.year, dt.month, dt.day, dt.hour)
            if key in results and results[key] is not None:
                avg = results[key]
                for name in SITE_NAMES:
                    v = avg.get(name)
                    if v is None:
                        continue  # 재시도도 FAIL — 기존 "" 유지
                    row[name] = str(v)  # float 또는 "NaN" 둘 다 str 처리
            rows.append(row)
    _save_csv(csv_path, SITE_NAMES, rows)


def main():
    files = sorted(glob.glob(os.path.join(SAVE_DIR, "gk2a_ghi_*.csv")))
    print(f"=== GK-2A FAIL 재수집 ===")
    print(f"  CSV 파일: {len(files)}")

    # 누락 row 전체 수집
    all_missing = []
    per_file = {}
    for fp in files:
        miss = find_missing(fp)
        per_file[fp] = miss
        all_missing.extend(miss)
        if miss:
            print(f"  {os.path.basename(fp)}: {len(miss)}건 누락")

    print(f"\n총 누락: {len(all_missing)}건")
    if not all_missing:
        print("재수집할 게 없습니다.")
        return

    # 키 매니저
    km = KeyManager(API_KEYS, BYTES_LIMIT_PER_KEY)
    print(f"키 {len(API_KEYS)}개, 워커 {NUM_WORKERS}개로 재시도\n")

    # 병렬 재수집 (증분 저장)
    results = {}
    completed = 0
    start = time.time()
    SAVE_EVERY = 50  # 50건마다 CSV 저장

    def incremental_save():
        """현재 results를 CSV에 반영."""
        for fp, miss in per_file.items():
            file_results = {k: results[k] for k in miss if k in results}
            if file_results:
                try:
                    update_csv(fp, file_results)
                except Exception as e:
                    _tlog(f"  !! 저장 실패 {os.path.basename(fp)}: {e}")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {
            ex.submit(collect_hour, y, m, d, h, km, SITE_NAMES): (y, m, d, h)
            for y, m, d, h in all_missing
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                avg, fc, nb = fut.result()
            except Exception as e:
                _tlog(f"  !! {key} 예외: {e}")
                continue
            if avg is None:
                _tlog(f"  !! {key} 키 소진")
                continue
            results[key] = avg
            completed += 1
            if completed % 20 == 0:
                el = time.time() - start
                rate = completed / el if el > 0 else 0
                _tlog(f"  [{completed:4d}/{len(all_missing)}] 경과 {el:.0f}s | 속도 {rate*60:.1f} hr/min | 사용 {km.total_usage_gb():.2f}GB")
            # 증분 저장 (중단되어도 결과 보존)
            if completed % SAVE_EVERY == 0:
                incremental_save()
                _tlog(f"    (증분 저장 완료)")

    print(f"\n총 재시도 완료: {len(results)} / {len(all_missing)}")
    print(f"전체 사용 쿼터: {km.total_usage_gb():.2f} GB")

    # 파일별로 업데이트
    print("\nCSV 업데이트 중...")
    for fp, miss in per_file.items():
        if not miss:
            continue
        # 이 파일에 해당하는 결과만 추출
        file_results = {k: results[k] for k in miss if k in results}
        if file_results:
            update_csv(fp, file_results)
            recovered = sum(1 for v in file_results.values() if any(x is not None for x in v.values()))
            print(f"  {os.path.basename(fp)}: {recovered}/{len(miss)} 복구")

    print("\n완료.")


if __name__ == "__main__":
    main()
