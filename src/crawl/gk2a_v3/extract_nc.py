"""GK-2A v3 — 로컬 NC 파일에서 사이트별 DSR/ASR/RSR 추출.

입력: data/gk2a_raw/YYYYMM/DD/*.nc (10분 단위, 파일시간 UTC)
     (샘플: data/gk2a_raw_sample/DD/*.nc — 드라이런용)
출력: data/gk2a_v3/YYYYMM.csv (long format, 10분 해상도)

핵심:
- 1x1 픽셀 추출 (2km 격자 자체가 충분히 coarse)
- DSR/ASR/RSR 3변수 + 각 DQF + SW_DQF 전부 보존
- UTC ↔ KST 변환 제공 (파일시간=UTC 확정, file_creation_time Z 명시)
- status 플래그로 OK/NaN/DQF거부/grid밖/읽기실패 구분

사용:
    python extract_nc.py 202201                   # 단일 월
    python extract_nc.py 202201 202202 202203     # 복수 월
    python extract_nc.py --sample                 # data/gk2a_raw_sample/ 드라이런
"""

import os
import sys
import re
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta, timezone
# ProcessPool/ThreadPool 둘 다 Windows에서 이슈 (multiprocessing DLL 차단 + HDF5 스레드 안전성) → 순차 처리

# Windows cp949 회피
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from netCDF4 import Dataset as NCDataset  # xarray 대신 직접 접근 (11픽셀 슬라이스 빠름)
from pyproj import Transformer

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "gk2a_raw"
SAMPLE_DIR = PROJECT_ROOT / "data" / "gk2a_raw_sample"
OUT_DIR = PROJECT_ROOT / "data" / "gk2a_v3"

# ============================================================
# 사이트 11 호기 좌표 (태양광, data_strategy.md §2 확정)
# ============================================================
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

# ============================================================
# LCC 투영 — NC 파일 메타데이터와 동일
# ============================================================
PROJ_STR = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +x_0=0 +y_0=0 +datum=WGS84"
TRANSFORMER = Transformer.from_crs("EPSG:4326", PROJ_STR, always_xy=True)
PIXEL_SIZE = 2000
UL_EASTING = -899000
UL_NORTHING = 899000

SITE_PIXELS = []
for name, lat, lon in SITES:
    x, y = TRANSFORMER.transform(lon, lat)
    col = int((x - UL_EASTING) / PIXEL_SIZE)
    row = int((UL_NORTHING - y) / PIXEL_SIZE)
    SITE_PIXELS.append((name, row, col))

UTC = timezone.utc
KST = timezone(timedelta(hours=9))

# ============================================================
# 추출 변수 목록
# ============================================================
VARS = {
    "DSR": "dsr",             # Downward SW Radiation — GHI
    "DSR_DQF1": "dsr_dqf",
    "ASR": "asr",             # Absorbed SW Radiation
    "ASR_DQF1": "asr_dqf",
    "RSR": "rsr",             # Reflected SW Radiation
    "RSR_DQF1": "rsr_dqf",
    "SW_DQF": "sw_dqf",
}


def parse_filename_utc(name):
    """파일명에서 UTC datetime 추출.
    예: gk2a_ami_le2_swrad_ko020lc_202201150300.nc → UTC 2022-01-15 03:00"""
    m = re.search(r"(\d{12})", name)
    if not m:
        return None
    ts = m.group(1)
    return datetime(int(ts[0:4]), int(ts[4:6]), int(ts[6:8]),
                    int(ts[8:10]), int(ts[10:12]), tzinfo=UTC)


def extract_one(nc_path, utc_dt):
    """NC 1개 파일에서 사이트별 변수 뽑기. records 리스트 반환.

    최적화: 전체 배열 로드 대신 픽셀 좌표에서 직접 슬라이스 (약 100배 빠름).
    """
    try:
        ds = NCDataset(str(nc_path), "r")
        # 사이트별 픽셀 11개만 한 번에 슬라이스
        # netCDF4는 팬시 인덱싱 지원 X → 각 사이트 스칼라 읽기
        site_vals = []  # [(name, {nc_var: raw_value or None})]
        for site_name, row, col in SITE_PIXELS:
            if not (0 <= row < 900 and 0 <= col < 900):
                site_vals.append((site_name, None))  # out of grid 표시
                continue
            vals = {}
            for nc_var in VARS:
                if nc_var in ds.variables:
                    raw = ds.variables[nc_var][row, col]
                    # netCDF4는 masked array로 반환; float 변환 시 NaN 관리
                    if np.ma.is_masked(raw):
                        vals[nc_var] = None
                    else:
                        v = float(raw)
                        vals[nc_var] = None if np.isnan(v) else v
                else:
                    vals[nc_var] = None
            site_vals.append((site_name, vals))
        ds.close()
    except Exception as e:
        return [
            (utc_dt, name, *[None] * len(VARS), f"read_error_{type(e).__name__}")
            for name, _, _ in SITE_PIXELS
        ]

    records = []
    for site_name, vals in site_vals:
        if vals is None:
            records.append((utc_dt, site_name, *[None] * len(VARS), "out_of_grid"))
            continue

        # Status 판정
        dsr_v = vals.get("DSR")
        dqf_v = vals.get("DSR_DQF1")
        sw_dqf_v = vals.get("SW_DQF")

        if dsr_v is None:
            status = "nan_value"
        elif dqf_v is not None and int(dqf_v) != 1:
            status = "dsr_dqf_reject"
        elif sw_dqf_v is not None and int(sw_dqf_v) != 1:
            status = "sw_dqf_reject"
        else:
            status = "ok"

        def _fmt(v, is_dqf=False):
            if v is None:
                return None
            if is_dqf:
                return int(v)
            return round(float(v), 2)

        records.append((
            utc_dt, site_name,
            _fmt(vals["DSR"]),
            _fmt(vals["DSR_DQF1"], is_dqf=True),
            _fmt(vals["ASR"]),
            _fmt(vals["ASR_DQF1"], is_dqf=True),
            _fmt(vals["RSR"]),
            _fmt(vals["RSR_DQF1"], is_dqf=True),
            _fmt(vals["SW_DQF"], is_dqf=True),
            status,
        ))
    return records


def _process_file(nc_path):
    """단일 파일 처리 (병렬 실행용)."""
    utc_dt = parse_filename_utc(nc_path.name)
    if utc_dt is None:
        return []
    return extract_one(nc_path, utc_dt)


def process_month(yyyymm, base_dir=None, output_name=None, num_workers=None):
    """월 단위 처리. base_dir 이하의 *.nc 전부 스캔. 순차 처리.

    num_workers 파라미터는 하위 호환용 (무시됨 — Windows HDF5/multiprocessing 이슈로 sequential).
    최적화된 netCDF4 슬라이스로 파일당 ~50ms → 4,400 파일 약 3.7분/월.
    """
    if base_dir is None:
        base_dir = RAW_DIR / yyyymm

    if not base_dir.exists():
        print(f"  !! 디렉토리 없음: {base_dir}")
        return

    nc_files = sorted(base_dir.rglob("*.nc"))
    print(f"  {yyyymm}: {len(nc_files)} NC 파일 스캔 (순차 처리)", flush=True)
    if not nc_files:
        return

    all_recs = []
    import time
    start = time.time()

    for i, nc in enumerate(nc_files):
        try:
            recs = _process_file(nc)
            all_recs.extend(recs)
        except Exception as e:
            print(f"  !! {nc.name}: {e}", flush=True)
        if (i + 1) % 500 == 0:
            pct = (i + 1) / len(nc_files) * 100
            el = time.time() - start
            rate = (i + 1) / el if el > 0 else 0
            remain = (len(nc_files) - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1}/{len(nc_files)} ({pct:.0f}%) | "
                  f"경과 {el/60:.1f}분 | 남은 {remain/60:.1f}분", flush=True)

    cols = [
        "datetime_utc", "site",
        "dsr", "dsr_dqf",
        "asr", "asr_dqf",
        "rsr", "rsr_dqf",
        "sw_dqf",
        "status",
    ]
    df = pd.DataFrame(all_recs, columns=cols)
    df["datetime_kst"] = df["datetime_utc"].apply(
        lambda dt: dt.astimezone(KST).replace(tzinfo=None) if dt else None
    )
    df["datetime_utc"] = df["datetime_utc"].apply(
        lambda dt: dt.replace(tzinfo=None) if dt else None
    )
    df = df[["datetime_utc", "datetime_kst", "site"] + cols[2:]]
    df = df.sort_values(["datetime_utc", "site"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / (output_name or f"{yyyymm}.csv")
    df.to_csv(out_file, index=False, encoding="utf-8")

    # ===== 통계 리포트 =====
    n = len(df)
    print(f"    저장: {out_file}")
    print(f"    총 행: {n:,}")
    if n:
        s = df["status"].value_counts()
        for k, v in s.items():
            print(f"      {k}: {v:,} ({v/n*100:.1f}%)")
        ok = df[df["status"] == "ok"]
        if len(ok):
            print(f"    DSR 유효값 통계: min {ok['dsr'].min():.1f}, "
                  f"max {ok['dsr'].max():.1f}, mean {ok['dsr'].mean():.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("months", nargs="*", help="YYYYMM 목록")
    parser.add_argument("--sample", action="store_true",
                        help="data/gk2a_raw_sample/ 전용 드라이런")
    parser.add_argument("--workers", type=int, default=4,
                        help="병렬 워커 수 (기본 4, CPU 코어 수 참고하여 조정)")
    args = parser.parse_args()

    print("=== GK-2A v3 추출기 ===")
    print(f"  raw base: {RAW_DIR}")
    print(f"  output:   {OUT_DIR}")
    print(f"  사이트 11개 × 10분 해상도 × {len(VARS)} 변수")
    print()

    if args.sample:
        print("=== 샘플 모드 (gk2a_raw_sample) ===")
        # 샘플은 월/일 구조 없이 바로 day-level 폴더일 수 있음
        sample_root = SAMPLE_DIR
        if not sample_root.exists():
            print(f"  !! 샘플 없음: {sample_root}")
            return
        # 드라이런: 샘플 전체를 "sample" 이름으로 처리
        process_month("sample", base_dir=sample_root,
                      output_name="sample.csv", num_workers=args.workers)
        return

    if not args.months:
        parser.print_help()
        return

    for yyyymm in args.months:
        print(f"=== {yyyymm} ===")
        process_month(yyyymm, num_workers=args.workers)
        print()


if __name__ == "__main__":
    main()
