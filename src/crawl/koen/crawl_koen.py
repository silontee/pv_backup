"""
한국남동발전 링크데이터 크롤링 스크립트
- Playwright를 사용하여 시간대별 발전실적 데이터를 CSV로 다운로드
- 대상: 화력, 태양광, 풍력, 연료전지, 해양소수력 시간대별 발전실적
- 기상정보, 대기오염물질 등도 포함
"""

import os
import sys
import time
import glob
from datetime import datetime, timedelta
from playwright.sync_api import sync_playwright

# 다운로드 경로
DOWNLOAD_DIR = os.path.abspath("C:/Energy_effi/data/raw")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# 남동발전 링크데이터 페이지 목록
PAGES = {
    # 시간대별 발전실적 (핵심)
    "화력_시간대별": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt26/main.do?menuCd=FN0912020221",
        "ipptNam": "화력",
    },
    "태양광_시간대별": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt21/main.do?menuCd=FN0912020217",
        "ipptNam": "태양광",
    },
    "풍력_시간대별": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt23/main.do?menuCd=FN0912020218",
        "ipptNam": "풍력",
    },
    "연료전지_시간대별": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt07/main.do?menuCd=FN0912020215",
        "ipptNam": "연료전지",
    },
    # 기상정보
    "기상정보_일자별": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt18/main.do?menuCd=FN0912020203",
        "ipptNam": None,
    },
    # 대기오염
    "대기오염_배출실적": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt02/main.do?menuCd=FN0912020204",
        "ipptNam": None,
    },
    "대기오염_일자별배출": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt14/main.do?menuCd=FN0912020211",
        "ipptNam": None,
    },
    # 발전소 주변 농도
    "주변농도": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt03/main.do?menuCd=FN0912020201",
        "ipptNam": None,
    },
    "주변농도_일자별": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt17/main.do?menuCd=FN0912020202",
        "ipptNam": None,
    },
}


def generate_date_ranges(start_date, end_date, interval_days=30):
    """날짜 범위를 interval_days 단위로 분할"""
    ranges = []
    current = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    while current < end:
        range_end = min(current + timedelta(days=interval_days - 1), end)
        ranges.append((current.strftime("%Y%m%d"), range_end.strftime("%Y%m%d")))
        current = range_end + timedelta(days=1)
    return ranges


def wait_for_download(download_dir, timeout=60):
    """다운로드 완료 대기"""
    start = time.time()
    while time.time() - start < timeout:
        files = glob.glob(os.path.join(download_dir, "*.csv"))
        crdownload = glob.glob(os.path.join(download_dir, "*.crdownload"))
        tmp = glob.glob(os.path.join(download_dir, "*.tmp"))
        if files and not crdownload and not tmp:
            # 가장 최근 파일 반환
            latest = max(files, key=os.path.getmtime)
            if time.time() - os.path.getmtime(latest) < 5:
                return latest
        time.sleep(1)
    return None


def crawl_page(page_name, page_config, start_date, end_date, browser_context):
    """단일 페이지의 데이터를 크롤링"""
    print(f"\n{'='*60}")
    print(f"크롤링 시작: {page_name}")
    print(f"기간: {start_date} ~ {end_date}")
    print(f"URL: {page_config['url']}")
    print(f"{'='*60}")

    date_ranges = generate_date_ranges(start_date, end_date, interval_days=30)

    for i, (d_start, d_end) in enumerate(date_ranges):
        print(f"\n  [{i+1}/{len(date_ranges)}] {d_start} ~ {d_end}")

        page = browser_context.new_page()
        try:
            # 페이지 로드
            page.goto(page_config["url"], wait_until="networkidle", timeout=30000)
            time.sleep(2)

            # 날짜 입력
            # 시작일
            start_input = page.query_selector('input[name="strDateS"], input[id="strDateS"]')
            if start_input:
                start_input.fill("")
                start_input.fill(f"{d_start[:4]}-{d_start[4:6]}-{d_start[6:8]}")

            # 종료일
            end_input = page.query_selector('input[name="strDateE"], input[id="strDateE"]')
            if end_input:
                end_input.fill("")
                end_input.fill(f"{d_end[:4]}-{d_end[4:6]}-{d_end[6:8]}")

            # 발전소 선택 (전체 선택하기 위해 첫번째 옵션 유지)
            # strOrgNo select가 있으면 첫 번째 옵션(전체) 선택

            # 조회 버튼 클릭
            submit_btn = page.query_selector('a.btn_search, button.btn_search, input[type="submit"]')
            if not submit_btn:
                # goSubmit 함수를 직접 호출
                page.evaluate("goSubmit()")
            else:
                submit_btn.click()

            time.sleep(3)
            page.wait_for_load_state("networkidle", timeout=30000)

            # CSV 다운로드
            with page.expect_download(timeout=60000) as download_info:
                # goCsvDown 함수 호출
                page.evaluate("goCsvDown()")

            download = download_info.value
            filename = f"{page_name}_{d_start}_{d_end}.csv"
            save_path = os.path.join(DOWNLOAD_DIR, filename)
            download.save_as(save_path)

            file_size = os.path.getsize(save_path)
            print(f"    -> 저장: {filename} ({file_size:,} bytes)")

        except Exception as e:
            print(f"    -> 오류: {e}")
        finally:
            page.close()

        time.sleep(1)  # 서버 부하 방지


def main():
    # 크롤링 기간 설정
    START_DATE = "20230101"  # 2023년 1월 1일부터
    END_DATE = "20241231"    # 2024년 12월 31일까지

    # 특정 페이지만 크롤링하려면 여기서 필터링
    target_pages = sys.argv[1:] if len(sys.argv) > 1 else list(PAGES.keys())

    print(f"크롤링 대상: {target_pages}")
    print(f"기간: {START_DATE} ~ {END_DATE}")
    print(f"저장 경로: {DOWNLOAD_DIR}")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
        )
        context = browser.new_context(
            accept_downloads=True,
            ignore_https_errors=True,
        )

        for page_name in target_pages:
            if page_name not in PAGES:
                print(f"[경고] '{page_name}'은 정의되지 않은 페이지입니다. 건너뜁니다.")
                continue

            page_config = PAGES[page_name]
            crawl_page(page_name, page_config, START_DATE, END_DATE, context)

        browser.close()

    print(f"\n{'='*60}")
    print(f"크롤링 완료!")
    print(f"저장된 파일 목록:")
    for f in sorted(glob.glob(os.path.join(DOWNLOAD_DIR, "*.csv"))):
        size = os.path.getsize(f)
        print(f"  {os.path.basename(f)} ({size:,} bytes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
