"""
GK-2A 위성 GHI(DSR) 수집 — Strict 버전

- 기상청 API허브에서 SWRAD NC 파일 다운로드
- 태양광 사이트 좌표에서 DSR(GHI) 추출
- **매 정시 (1장/시간)** — 10분 단위는 throttle 심해서 정각만
- 주간 07~18 KST 12시간만 (06/19/20시는 KMA 에서 NaN 주는 구간 → 제외)
- 기간: CLI 인자로 지정, 기본 2022-01 ~ 2024-12
- 키 18개 로테이션, 키당 4.5GB **pre-check** (5GB 절대 안 넘음)
- **NaN vs FAIL 구분**: 다운로드 성공 but 값 없음 → "NaN" / 다운 실패 → "" (빈문자열)
"""

import os
import sys
import time
import csv
import calendar
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import requests
import numpy as np
import xarray as xr
from pyproj import Transformer

API_KEYS = [
    # --- 유저 검증 목록 (2026-04-21): 작동 유력 ---
    # 비번 ddol0202@ 계정들
    "p8A4StGqQuSAOErRqtLkrA",      # kimsw5652@naver.com
    "iDZuH9eFTFi2bh_XhfxYWQ",      # won0202@gachon.ac.kr
    "6fbBHSmEQ1a2wR0phENWYQ",      # kimsw5507@gmail.com
    "vLGm-52yQdixpvudsqHYCQ",      # kimsw9771@gmail.com          ← NEW
    "A4HvGcpZQdeB7xnKWVHXyg",      # kimsw130079@gmail.com        ← NEW
    "GOMCAft3Q6mjAgH7dyOpzQ",      # kimsw5652@nate.com           ← NEW
    # 비번 alexander123! 계정들
    "aKqhZx9iScOqoWcfYknD9Q",      # subeen119@gachon.ac.kr
    "TbQVn43xRgi0FZ-N8XYIbQ",      # dltnqls1919@gmail.com
    "ex6T6qmzSliek-qps8pYbg",      # subeen119@gmail.com          ← NEW
    "CVopnVy4RqCaKZ1cuPagNA",      # subeen119@naver.com          ← NEW
    "t8LdExxpRRKC3RMcaaUSBw",      # qwer041109@gmail.com
    "ew6BMTpJSO6OgTE6STjuGQ",      # asdf041109@gmail.com
    # --- 이전 목록, 상태 미확인 ---
    "yJOD_596Q4GTg_-feqOBLQ",
    "7R08Z4yyTDudPGeMsrw72g",
    # --- 2026-04-21 추가 4개, 활용신청 확인 필요 ---
    "S6KnUHqTSQGip1B6kykBOA",
    "QeBwMrh6T2OgcDK4ep9jvA",
    "RTcn_eohQum3J_3qITLpKA",
    "Q4c6XndvRSaHOl53b7Um0g",
]
BYTES_LIMIT_PER_KEY = int(4.5 * 1024 * 1024 * 1024)  # 4.5GB

API_BASE = "https://apihub.kma.go.kr/api/typ05/api/GK2A/LE2/SWRAD/KO/data"

# 프로젝트 루트 = src/crawl/gk2a/ 에서 3단계 상위
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "gk2a_ghi")

# LCC 투영
PROJ_STR = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +x_0=0 +y_0=0 +datum=WGS84"
TRANSFORMER = Transformer.from_crs("EPSG:4326", PROJ_STR, always_xy=True)
PIXEL_SIZE = 2000
UL_EASTING = -899000
UL_NORTHING = 899000

# 태양광 사이트 (중복 좌표 제거 - 고유 위치만)
SITES = [
    ("고흥만수상", 34.5711, 127.2803),
    ("삼천포", 34.9469, 128.0603),
    ("영흥", 37.2572, 126.4975),
    ("광양항세방", 34.9181, 127.7106),
    ("예천", 36.6467, 128.4525),
    ("영동", 37.1839, 128.4614),
    ("탑선", 35.2994, 126.7847),
    ("구미", 36.1195, 128.3444),
    ("경상대", 35.1531, 128.0936),
    ("여수", 34.7604, 127.6622),
    ("창원", 35.2100, 128.5831),
]

# 격자 인덱스 미리 계산
SITE_PIXELS = []
for name, lat, lon in SITES:
    x_lcc, y_lcc = TRANSFORMER.transform(lon, lat)
    col = int((x_lcc - UL_EASTING) / PIXEL_SIZE)
    row = int((UL_NORTHING - y_lcc) / PIXEL_SIZE)
    SITE_PIXELS.append((name, row, col))

KST_HOURS = list(range(7, 19))               # 07~18 KST (12시간) — 06/19/20시는 KMA NaN 구간
KST_MINUTES = [0]                            # 정각만 (1장/시간) — 10분 단위는 throttle 심함
MAX_FILE_SIZE = 10 * 1024 * 1024             # 10MB 여유 (2025-11/12 실측 8MB, 2배 버퍼)
SLEEP_BETWEEN_REQ = 2.0                      # 초 (병렬 시 각 워커 개별)
NUM_WORKERS = 8                              # 병렬 워커 수


class KeyManager:
    """API 키 로테이션 + pre-check 용량 제한 (thread-safe, 5GB 절대 안 넘음)"""

    def __init__(self, keys, limit_per_key):
        self.keys = keys
        self.limit = limit_per_key
        self.usage = {k: 0 for k in keys}
        self.current_idx = 0
        self._lock = threading.Lock()

    def get_usable_key(self, expected_size=MAX_FILE_SIZE):
        """다음 요청을 받을 수 있는 키 반환. 없으면 None. Thread-safe."""
        with self._lock:
            for _ in range(len(self.keys)):
                key = self.keys[self.current_idx]
                if self.usage[key] + expected_size <= self.limit:
                    return key
                self.current_idx = (self.current_idx + 1) % len(self.keys)
            return None

    def add_usage(self, key, nbytes):
        with self._lock:
            self.usage[key] += nbytes

    def total_usage_gb(self):
        with self._lock:
            return sum(self.usage.values()) / (1024 ** 3)

    def remaining_gb(self):
        with self._lock:
            return sum(max(0, self.limit - u) for u in self.usage.values()) / (1024 ** 3)

    def has_quota(self):
        with self._lock:
            return any(u + MAX_FILE_SIZE <= self.limit for u in self.usage.values())


def kst_to_utc_str(year, month, day, kst_hour, kst_minute=0):
    kst = datetime(year, month, day, kst_hour, kst_minute)
    utc = kst - timedelta(hours=9)
    return utc.strftime("%Y%m%d%H%M")


def download_nc(utc_str, key):
    url = f"{API_BASE}?date={utc_str}&authKey={key}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 1000:
                tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
                tmp.write(r.content)
                tmp.close()
                return tmp.name, len(r.content)
            elif r.status_code == 403:
                return None, 0  # 키 제한
            else:
                return None, 0
        except Exception:
            if attempt < 2:
                time.sleep(5)
    return None, 0


def extract_ghi(nc_path):
    try:
        ds = xr.open_dataset(nc_path)
        dsr = ds["DSR"].values
        dqf = ds["DSR_DQF1"].values
        ds.close()

        results = {}
        for name, row, col in SITE_PIXELS:
            if 0 <= row < 900 and 0 <= col < 900:
                val = float(dsr[row, col])
                qf = float(dqf[row, col])
                if np.isnan(val) or qf != 1:
                    results[name] = None
                else:
                    results[name] = round(val, 1)
            else:
                results[name] = None
        return results
    except Exception:
        return {name: None for name, _, _ in SITE_PIXELS}


_PRINT_LOCK = threading.Lock()

def _tlog(msg):
    """thread-safe print"""
    with _PRINT_LOCK:
        print(msg, flush=True)


def collect_hour(year, month, day, kst_hour, km, site_names):
    """한 시간대 다운로드 → 사이트별 값 반환.

    Returns: (result_dict, fail_count, total_bytes)
      result_dict 값 타입:
        - float  : 정상 GHI (avg)
        - "NaN"  : 다운로드 성공 but KMA가 준 값이 NaN (실제 no-data)
        - None   : 다운로드 실패 (재시도 대상)
      result_dict == None: 키 소진
    """
    samples = {name: [] for name in site_names}  # 다운로드 성공한 샘플들 (None 포함 가능)
    fail_count = 0
    total_bytes = 0

    for minute in KST_MINUTES:
        key = km.get_usable_key()
        if key is None:
            return None, fail_count, total_bytes

        utc_str = kst_to_utc_str(year, month, day, kst_hour, minute)
        t0 = time.time()
        nc_path, nbytes = download_nc(utc_str, key)
        elapsed = time.time() - t0

        if nc_path:
            km.add_usage(key, nbytes)
            total_bytes += nbytes
            ghi_vals = extract_ghi(nc_path)
            os.unlink(nc_path)
            for name in site_names:
                samples[name].append(ghi_vals.get(name))  # float 또는 None(KMA NaN) 둘 다 기록
            _tlog(f"    {year:04d}-{month:02d}-{day:02d} {kst_hour:02d}:{minute:02d} OK  {nbytes/1024:.0f}KB / {elapsed:.1f}s ({nbytes/elapsed/1024:.0f}KB/s) [{key[:8]}]")
        else:
            fail_count += 1
            _tlog(f"    {year:04d}-{month:02d}-{day:02d} {kst_hour:02d}:{minute:02d} FAIL ({elapsed:.1f}s) [{key[:8]}]")

        time.sleep(SLEEP_BETWEEN_REQ)

    # 사이트별 결과 계산
    num_ok_downloads = len(KST_MINUTES) - fail_count
    result = {}
    for name in site_names:
        if num_ok_downloads == 0:
            # 모든 다운로드 실패 → FAIL
            result[name] = None
        else:
            valid = [v for v in samples[name] if v is not None]
            if valid:
                result[name] = round(sum(valid) / len(valid), 1)
            else:
                # 다운로드는 성공했지만 KMA 가 준 값이 전부 NaN → 실데이터 NaN
                result[name] = "NaN"
    return result, fail_count, total_bytes


def collect_month(year, month, km):
    """월 단위 수집 — 시간(hour) 단위로 병렬 실행."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = f"gk2a_ghi_{year:04d}{month:02d}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        print(f"  건너뜀: {filename}")
        return True

    last_day = calendar.monthrange(year, month)[1]
    site_names = [name for name, _, _ in SITE_PIXELS]

    # 작업 큐: (day, hour) 모든 조합
    tasks = [(day, h) for day in range(1, last_day + 1) for h in KST_HOURS]
    print(f"  작업 {len(tasks)}개 시간, 워커 {NUM_WORKERS}개")

    results = {}  # (day, hour) -> avg_dict
    completed_count = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(collect_hour, year, month, d, h, km, site_names): (d, h) for d, h in tasks}

        for fut in as_completed(futures):
            d, h = futures[fut]
            try:
                avg, fc, nbytes = fut.result()
            except Exception as e:
                print(f"  !! {year}-{month:02d}-{d:02d} {h:02d}시 예외: {e}")
                continue

            if avg is None:
                # 키 소진
                print(f"  !! {year}-{month:02d}-{d:02d} {h:02d}시 키 소진으로 건너뜀")
                continue

            results[(d, h)] = avg
            completed_count += 1

            # 시간 단위 요약 (한 시간 완료 시 매번)
            elapsed = time.time() - start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            remaining = (len(tasks) - completed_count) / rate if rate > 0 else 0
            _tlog(f"  [{completed_count:4d}/{len(tasks)}] {year}-{month:02d}-{d:02d} {h:02d}시 완료 (실패샘플 {fc}/6, {nbytes/1024:.0f}KB) | 총경과 {elapsed/60:.1f}분 | 남은예상 {remaining/60:.0f}분 | 사용 {km.total_usage_gb():.2f}GB")

            # 50시간마다 증분 CSV 저장 (중도 중단 대비)
            if completed_count % 50 == 0:
                tmp_rows = []
                for d2, h2 in tasks:
                    row = {"datetime_kst": f"{year:04d}-{month:02d}-{d2:02d} {h2:02d}:00"}
                    avg2 = results.get((d2, h2))
                    for name in site_names:
                        row[name] = avg2.get(name) if avg2 else None
                    tmp_rows.append(row)
                _save_csv(filepath, site_names, tmp_rows)
                print(f"    (증분 저장 완료, {filename})")

    # 결과를 정렬해서 CSV로 저장
    rows = []
    for d, h in tasks:
        row = {"datetime_kst": f"{year:04d}-{month:02d}-{d:02d} {h:02d}:00"}
        avg = results.get((d, h))
        for name in site_names:
            row[name] = avg.get(name) if avg else None
        rows.append(row)

    _save_csv(filepath, site_names, rows)
    filled = len(results)
    print(f"  저장: {filename} ({len(rows)}행, 성공 시간 {filled}/{len(tasks)})")
    return filled > 0


def _save_csv(filepath, site_names, rows):
    header = ["datetime_kst"] + site_names
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def iter_months(start_year, start_month, end_year, end_month, reverse=False):
    """월 (year, month) 튜플 순회. reverse=True면 최근 → 과거."""
    months = []
    for y in range(start_year, end_year + 1):
        sm = start_month if y == start_year else 1
        em = end_month if y == end_year else 12
        for m in range(sm, em + 1):
            months.append((y, m))
    if reverse:
        months = months[::-1]
    return months


if __name__ == "__main__":
    km = KeyManager(API_KEYS, BYTES_LIMIT_PER_KEY)

    start_year, start_month = 2022, 1
    end_year, end_month = 2025, 12
    reverse = False  # 기본: 과거 → 최근 (작은 파일 2022부터 확보)

    if len(sys.argv) >= 3:
        start_year = int(sys.argv[1])
        start_month = int(sys.argv[2])
        if len(sys.argv) >= 5:
            end_year = int(sys.argv[3])
            end_month = int(sys.argv[4])
        if len(sys.argv) >= 6:
            reverse = sys.argv[5].lower() in ("reverse", "desc", "true", "1")

    print(f"=== GK-2A GHI 수집 (Strict + Parallel) ===")
    print(f"  기간: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d} ({'최근 → 과거' if reverse else '과거 → 최근'})")
    print(f"  시간: {KST_HOURS[0]:02d}:00 ~ {KST_HOURS[-1]:02d}:00 KST, 매 10분 (시간당 6장 → 평균)")
    print(f"  키 {len(API_KEYS)}개, 키당 {BYTES_LIMIT_PER_KEY / (1024**3):.1f}GB, 총 {len(API_KEYS) * BYTES_LIMIT_PER_KEY / (1024**3):.1f}GB 쿼터")
    print(f"  병렬 워커: {NUM_WORKERS}개, 워커당 Sleep: {SLEEP_BETWEEN_REQ}s")
    print()

    for y, m in iter_months(start_year, start_month, end_year, end_month, reverse=reverse):
        print(f"\n=== {y}-{m:02d} ===")
        ok = collect_month(y, m, km)
        if not ok:
            print(f"\n!! 키 소진으로 중단. 사용량: {km.total_usage_gb():.2f}GB")
            sys.exit(1)

    print(f"\n=== 완료! 총 사용량: {km.total_usage_gb():.2f}GB ===")
