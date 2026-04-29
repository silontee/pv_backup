"""NCDC SFTP에서 GK-2A NC 파일 월별 자동 다운로드 (병렬).

특징:
- paramiko SFTP 클라이언트
- 비밀번호 실행 시 프롬프트 (파일 저장 없음)
- 이미 받은 파일 자동 스킵 (resume)
- 병렬 다운로드 (ThreadPoolExecutor)
- 월 단위 처리, 실패 재시도

사용:
    python download_nc.py 202201                   # 단일 월
    python download_nc.py 202201 202202 202203     # 복수 월
    python download_nc.py --parallel 6 202201      # 병렬 6
    python download_nc.py --dry-run 202201         # 목록만 (실제 다운 X)

환경변수 (선택, 비번 프롬프트 생략용):
    NCDC_PASSWORD — 설정 시 프롬프트 생략
    NCDC_USER     — 기본값: ncdcftp17
"""

import os
import sys
import argparse
import getpass
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Windows cp949 회피
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import paramiko

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "gk2a_raw"

# ============================================================
# 서버 정보 (공개 가능 — 비번은 별도)
# ============================================================
HOST = "203.247.94.245"
PORT = 40021
DEFAULT_USER = "ncdcftp17"
REMOTE_ROOT = "/SAT"  # 서버측 SAT 디렉토리


class DownloadStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.downloaded = 0
        self.skipped = 0
        self.failed = 0
        self.bytes = 0

    def inc(self, field, amount=1):
        with self.lock:
            setattr(self, field, getattr(self, field) + amount)


def make_sftp(host, port, user, password):
    """새 SFTP 클라이언트 생성 (워커마다 독립).

    SSHClient 경로로 가야 password/keyboard-interactive 양쪽 폴백 가능.
    """
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=port,
        username=user,
        password=password,
        look_for_keys=False,     # SSH 키 파일 조회 안 함 (경로에 키 있으면 오인)
        allow_agent=False,       # SSH agent 안 씀
        banner_timeout=30,
        auth_timeout=30,
    )
    sftp = client.open_sftp()
    return sftp, client


def list_remote_files(sftp, remote_month_dir):
    """원격 월 디렉토리 아래 모든 .nc 파일 경로/size 수집.

    구조 예: /SAT/202201/01/*.nc, /SAT/202201/02/*.nc, ...
    """
    files = []
    try:
        day_entries = sftp.listdir_attr(remote_month_dir)
    except IOError as e:
        print(f"  !! 원격 디렉토리 접근 실패: {remote_month_dir} ({e})")
        return files

    for day in sorted(day_entries, key=lambda e: e.filename):
        if not day.filename.isdigit():
            continue  # 일자 폴더(01~31)만
        day_dir = f"{remote_month_dir}/{day.filename}"
        try:
            nc_entries = sftp.listdir_attr(day_dir)
        except IOError:
            continue
        for nc in nc_entries:
            if nc.filename.endswith(".nc"):
                files.append({
                    "remote": f"{day_dir}/{nc.filename}",
                    "day": day.filename,
                    "filename": nc.filename,
                    "size": nc.st_size,
                })
    return files


def download_one(task, local_root, host, port, user, password, stats, skip_if_exists):
    """단일 파일 다운로드 (워커)."""
    local_day = local_root / task["day"]
    local_day.mkdir(parents=True, exist_ok=True)
    local_path = local_day / task["filename"]

    # 이미 있고 크기 맞으면 스킵
    if skip_if_exists and local_path.exists():
        if local_path.stat().st_size == task["size"]:
            stats.inc("skipped")
            return ("skip", task["filename"])

    # 워커별 독립 SFTP 세션
    try:
        sftp, client = make_sftp(host, port, user, password)
    except Exception as e:
        stats.inc("failed")
        return ("fail_connect", task["filename"], str(e))

    try:
        sftp.get(task["remote"], str(local_path))
        local_size = local_path.stat().st_size
        if local_size != task["size"]:
            local_path.unlink(missing_ok=True)
            stats.inc("failed")
            return ("fail_size_mismatch", task["filename"], f"{local_size} vs {task['size']}")
        stats.inc("downloaded")
        stats.inc("bytes", task["size"])
        return ("ok", task["filename"])
    except Exception as e:
        stats.inc("failed")
        # 부분 파일 제거
        if local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass
        return ("fail", task["filename"], str(e))
    finally:
        try:
            sftp.close()
            client.close()
        except Exception:
            pass


def trigger_extraction(yyyymm, extract_workers=4):
    """다운로드 완료 후 추출 subprocess 비동기 시작.

    Returns: subprocess.Popen 인스턴스
    """
    extract_script = Path(__file__).parent / "extract_nc.py"
    log_file = PROJECT_ROOT / "data" / "gk2a_v3" / f"_extract_{yyyymm}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(extract_script), yyyymm,
           "--workers", str(extract_workers)]
    print(f"  🔄 추출 subprocess 시작: {yyyymm} (로그: {log_file.name})")
    # stdout/stderr를 로그파일로 리다이렉트
    log_fh = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    return proc, log_fh


def download_month(yyyymm, user, password, num_workers=4, dry_run=False, skip_if_exists=True):
    """월 단위 다운로드."""
    print(f"\n=== {yyyymm} 다운로드 시작 ===")
    local_root = RAW_DIR / yyyymm
    local_root.mkdir(parents=True, exist_ok=True)

    # 1. 원격 목록 수집
    print(f"  원격 디렉토리 조회: /SAT/{yyyymm}/")
    sftp_list, client_list = make_sftp(HOST, PORT, user, password)
    try:
        files = list_remote_files(sftp_list, f"{REMOTE_ROOT}/{yyyymm}")
    finally:
        sftp_list.close()
        client_list.close()

    if not files:
        print(f"  !! 파일 없음 — {yyyymm} 서버에 없거나 접근 불가")
        return

    total_bytes = sum(f["size"] for f in files)
    print(f"  파일: {len(files)}개, 총 {total_bytes/1024/1024/1024:.2f} GB")

    if dry_run:
        print("  (dry-run) 실제 다운로드 안 함")
        for f in files[:3]:
            print(f"    샘플: {f['remote']} ({f['size']/1024:.0f}KB)")
        return

    # 2. 기존 파일 체크
    existing = 0
    for f in files:
        local = local_root / f["day"] / f["filename"]
        if local.exists() and local.stat().st_size == f["size"]:
            existing += 1
    if existing:
        print(f"  기존 완료: {existing}/{len(files)} — 스킵 예정")

    # 3. 병렬 다운로드
    stats = DownloadStats()
    print(f"  워커 {num_workers}개로 다운로드 시작...")

    import time
    start = time.time()
    done_count = [0]
    progress_lock = threading.Lock()

    def progress(result):
        with progress_lock:
            done_count[0] += 1
            # 50개마다 진행률
            if done_count[0] % 50 == 0:
                el = time.time() - start
                rate = done_count[0] / el if el > 0 else 0
                remain = (len(files) - done_count[0]) / rate if rate > 0 else 0
                mb_per_s = stats.bytes / 1024 / 1024 / el if el > 0 else 0
                print(f"    [{done_count[0]:>5}/{len(files)}] "
                      f"{done_count[0]*100/len(files):.1f}% | "
                      f"경과 {el/60:.1f}분 | 남은 {remain/60:.1f}분 | "
                      f"{mb_per_s:.1f} MB/s | "
                      f"OK {stats.downloaded} SKIP {stats.skipped} FAIL {stats.failed}")

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [
            ex.submit(download_one, f, local_root, HOST, PORT, user, password, stats, skip_if_exists)
            for f in files
        ]
        for fut in as_completed(futures):
            try:
                result = fut.result()
                progress(result)
                if result[0].startswith("fail"):
                    print(f"  !! {result[1]}: {result[0]} {result[2] if len(result) > 2 else ''}")
            except Exception as e:
                print(f"  !! 워커 예외: {e}")

    el = time.time() - start
    print(f"\n  {yyyymm} 완료 — 경과 {el/60:.1f}분")
    print(f"    다운: {stats.downloaded} / 스킵: {stats.skipped} / 실패: {stats.failed}")
    print(f"    총 {stats.bytes/1024/1024/1024:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("months", nargs="+", help="YYYYMM 목록")
    parser.add_argument("--parallel", type=int, default=6,
                        help="병렬 워커 수 (기본 6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="목록만 조회, 실제 다운 안 함")
    parser.add_argument("--user", default=os.environ.get("NCDC_USER", DEFAULT_USER),
                        help=f"SFTP 사용자명 (기본 {DEFAULT_USER})")
    parser.add_argument("--auto-extract", action="store_true", default=True,
                        help="월 다운 완료 즉시 추출 subprocess 자동 실행 (기본 ON)")
    parser.add_argument("--no-auto-extract", dest="auto_extract", action="store_false",
                        help="추출 자동 실행 비활성화")
    parser.add_argument("--extract-workers", type=int, default=4,
                        help="추출 subprocess 워커 수 (기본 4)")
    args = parser.parse_args()

    # 비밀번호: 환경변수 > 프롬프트
    password = os.environ.get("NCDC_PASSWORD")
    if not password:
        password = getpass.getpass(f"SFTP password for {args.user}@{HOST}: ")
    if not password:
        print("!! 비밀번호 필요")
        sys.exit(1)

    print(f"=== NCDC GK-2A 다운로드 ===")
    print(f"  서버: {HOST}:{PORT} (사용자: {args.user})")
    print(f"  로컬: {RAW_DIR}")
    print(f"  월: {args.months}")
    print(f"  병렬: {args.parallel}")
    if args.dry_run:
        print("  모드: DRY RUN (실제 다운 X)")

    # 추출 subprocess 트래킹
    extract_procs = []  # list of (yyyymm, Popen, log_fh)

    for yyyymm in args.months:
        try:
            download_month(yyyymm, args.user, password,
                           num_workers=args.parallel,
                           dry_run=args.dry_run)
            # 다운로드 성공 시 즉시 추출 (병렬)
            if args.auto_extract and not args.dry_run:
                proc, log_fh = trigger_extraction(yyyymm, args.extract_workers)
                extract_procs.append((yyyymm, proc, log_fh))
                # 완료된 추출 정리 (non-blocking poll)
                for ym, p, fh in extract_procs[:]:
                    if p.poll() is not None:
                        fh.close()
                        status = "✅" if p.returncode == 0 else f"❌ ({p.returncode})"
                        print(f"  {status} 추출 완료: {ym} → data/gk2a_v3/{ym}.csv")
                        extract_procs.remove((ym, p, fh))
        except KeyboardInterrupt:
            print(f"\n  !! {yyyymm} 중단")
            break
        except Exception as e:
            print(f"  !! {yyyymm} 예외: {e}")
            continue

    # 남은 추출 대기
    if extract_procs:
        print(f"\n=== 남은 추출 {len(extract_procs)}개 대기 중... ===")
        for ym, proc, log_fh in extract_procs:
            proc.wait()
            log_fh.close()
            status = "✅" if proc.returncode == 0 else f"❌ ({proc.returncode})"
            print(f"  {status} 추출 완료: {ym}")

    print("\n=== 전체 완료 ===")
    print(f"  원본 NC: data/gk2a_raw/")
    print(f"  추출 CSV: data/gk2a_v3/")
    print(f"  추출 로그: data/gk2a_v3/_extract_*.log")


if __name__ == "__main__":
    main()
