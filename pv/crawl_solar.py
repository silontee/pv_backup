"""
남동발전 태양광 시간���별 발전실적 + 기상정보 크롤링
- test_download.py에서 검증된 방식 기반
- 월 단위로 분할 크롤링 (서버 부하 방지)
- 태양광 발전실적 + 발전소 기상정보 동시 수집
"""

import os
import sys
import time
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

# === 설정 ===
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
os.makedirs(DATA_DIR, exist_ok=True)

# 크롤링 대상 페이지
TARGETS = {
    "solar": {
        "name": "태양광_시간대별",
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt21/main.do?menuCd=FN0912020217",
    },
    "weather": {
        "name": "기상정보_일자별",
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt18/main.do?menuCd=FN0912020203",
    },
    "thermal": {
        "name": "화력_시간대별",
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt26/main.do?menuCd=FN0912020221",
    },
}


def generate_monthly_ranges(start_ym: str, end_ym: str):
    """월 단위 날짜 범위 생성.
    start_ym: '202301', end_ym: '202604'
    returns: [('20230101', '20230131'), ('20230201', '20230228'), ...]
    """
    ranges = []
    current = datetime.strptime(start_ym + "01", "%Y%m%d")
    end = datetime.strptime(end_ym + "01", "%Y%m%d")

    while current < end:
        # 해당 월의 마지막 날
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1)
        else:
            next_month = current.replace(month=current.month + 1)
        last_day = next_month - timedelta(days=1)

        ranges.append((current.strftime("%Y%m%d"), last_day.strftime("%Y%m%d")))
        current = next_month

    return ranges


def crawl_one_month(page, url: str, start_date: str, end_date: str, save_path: str):
    """한 달치 데이터를 크롤링하여 CSV로 저장.
    Returns: (success: bool, file_size: int)
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=30000)
        time.sleep(2)

        # 날짜 설정 (JS로 직접 — readonly 필드 우회)
        page.evaluate(f"document.getElementById('strDateS').value = '{start_date}'")
        page.evaluate(f"document.getElementById('strDateE').value = '{end_date}'")

        # 조회 실행
        with page.expect_navigation(timeout=60000):
            page.evaluate("goSubmit()")
        time.sleep(3)

        # CSV 다운로드
        with page.expect_download(timeout=60000) as download_info:
            page.evaluate("goCsvDown()")

        download = download_info.value
        download.save_as(save_path)
        file_size = os.path.getsize(save_path)

        return True, file_size

    except Exception as e:
        print(f"    [ERROR] {e}")
        return False, 0


def main():
    # --- 기간/대상 설정 ---
    START_YM = os.environ.get("START_YM", "202301")  # 기본: 2023년 1월
    END_YM = os.environ.get("END_YM", "202604")      # 기본: 2026년 4월 (미포함)

    # 대상 선택: 인자 없으면 solar만, 인자로 지정 가능 (solar, weather, thermal)
    if len(sys.argv) > 1:
        target_keys = sys.argv[1:]
    else:
        target_keys = ["solar"]

    # 유효성 검사
    for k in target_keys:
        if k not in TARGETS:
            print(f"[ERROR] 알 수 없는 대상: '{k}'. 가능한 값: {list(TARGETS.keys())}")
            sys.exit(1)

    monthly_ranges = generate_monthly_ranges(START_YM, END_YM)

    print(f"=== 남동발전 크롤링 시작 ===")
    print(f"  대상: {target_keys}")
    print(f"  기간: {START_YM} ~ {END_YM} ({len(monthly_ranges)}개월)")
    print(f"  저장: {DATA_DIR}")
    print()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)

        for target_key in target_keys:
            target = TARGETS[target_key]
            print(f"\n{'='*60}")
            print(f"  {target['name']}")
            print(f"{'='*60}")

            success_count = 0
            skip_count = 0
            fail_count = 0

            for i, (d_start, d_end) in enumerate(monthly_ranges):
                ym = d_start[:6]
                filename = f"{target_key}_{ym}.csv"
                save_path = os.path.join(DATA_DIR, filename)

                # 이미 존재하면 스킵
                if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
                    size = os.path.getsize(save_path)
                    print(f"  [{i+1}/{len(monthly_ranges)}] {ym} — SKIP (이미 존재, {size:,} bytes)")
                    skip_count += 1
                    continue

                print(f"  [{i+1}/{len(monthly_ranges)}] {ym} ({d_start}~{d_end}) ...", end=" ", flush=True)

                page = context.new_page()
                ok, size = crawl_one_month(page, target["url"], d_start, d_end, save_path)
                page.close()

                if ok:
                    print(f"OK ({size:,} bytes)")
                    success_count += 1
                else:
                    print(f"FAIL")
                    fail_count += 1

                time.sleep(2)  # 서버 부하 방지

            print(f"\n  결과: 성공 {success_count}, 스킵 {skip_count}, 실패 {fail_count}")

        browser.close()

    # 최종 요약
    print(f"\n{'='*60}")
    print(f"크롤링 완료! 저장된 파일:")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".csv"):
            size = os.path.getsize(os.path.join(DATA_DIR, f))
            print(f"  {f} ({size:,} bytes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
