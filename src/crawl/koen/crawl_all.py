"""
한국남동발전 링크데이터 크롤링 스크립트
- 월 단위(1일~말일)로 수집
- 멀티스레드로 병렬 다운로드
- data/ 하위에 카테고리별 폴더 저장
"""

import os
import sys
import calendar
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.sync_api import sync_playwright

BASE_DIR = os.path.abspath("data")
MAX_WORKERS = 4  # 동시 브라우저 탭 수
MAX_RETRIES = 3

# 크롤링 대상 페이지
PAGES = {
    "solar_hourly": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt21/main.do?menuCd=FN0912020217",
        "folder": "solar_hourly",
        "desc": "태양광 시간대별 발전실적",
    },
    "thermal_hourly": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt26/main.do?menuCd=FN0912020221",
        "folder": "thermal_hourly",
        "desc": "화력(분당) 시간대별 발전실적",
        "org_no": "8731",
        "hokis": ["CS1", "CS2", "CG1", "CG2", "CG3", "CG4", "CG5", "CG6", "CG7", "CG8", "ES1", "ES2"],
    },
    "wind_hourly": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt23/main.do?menuCd=FN0912020218",
        "folder": "wind_hourly",
        "desc": "풍력 시간대별 발전실적",
    },
    "fuelcell_hourly": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt25/main.do?menuCd=FN0912020220",
        "folder": "fuelcell_hourly",
        "desc": "연료전지 시간대별 발전실적",
    },
    "hydro_hourly": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt24/main.do?menuCd=FN0912020219",
        "folder": "hydro_hourly",
        "desc": "해양소수력 시간대별 발전실적",
    },
    "fuel_consumption": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt05/main.do?menuCd=FN0912020212",
        "folder": "fuel_consumption",
        "desc": "연료소비실적",
        "date_mode": "month",
    },
    "fuel_procurement": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt04/main.do?menuCd=FN0912020213",
        "folder": "fuel_procurement",
        "desc": "연료도입실적",
        "date_mode": "month",
    },
    "air_quality_daily": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt17/main.do?menuCd=FN0912020202",
        "folder": "air_quality_daily",
        "desc": "일자별 발전소 주변농도현황",
    },
    "emission_daily": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt16/main.do?menuCd=FN0912020205",
        "folder": "emission_daily",
        "desc": "일자별 대기오염물질배출",
    },
    "water_emission": {
        "url": "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt19/main.do?menuCd=FN0912020206",
        "folder": "water_emission",
        "desc": "수질오염물질배출",
    },
}


def generate_months(start_year, start_month, end_year, end_month):
    """월 단위 (1일~말일) 리스트 생성"""
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        last_day = calendar.monthrange(y, m)[1]
        d_start = f"{y:04d}{m:02d}01"
        d_end = f"{y:04d}{m:02d}{last_day:02d}"
        months.append((d_start, d_end))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def download_one(page_url, d_start, d_end, save_path, org_no=None, hoki=None, date_mode="date"):
    """단일 월 데이터 다운로드 (스레드에서 실행)
    org_no: 발전소 코드 (예: '8731')
    hoki: 호기 코드 (예: 'CG1')
    date_mode: 'date' (strDateS/E, YYYYMMDD) or 'month' (strMonthS/E, YYYYMM)
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()

                page.goto(page_url, wait_until="networkidle", timeout=60000)
                page.wait_for_function("typeof goCsvDown === 'function'", timeout=15000)

                # 발전소 선택 (있으면)
                if org_no:
                    page.select_option("select[name=strOrgNo]", org_no)
                    import time; time.sleep(2)  # 호기 목록 로드 대기

                # 호기 선택 (있으면)
                if hoki:
                    page.select_option("select[name=strHokiS]", hoki)
                    page.select_option("select[name=strHokiE]", hoki)

                # 날짜 설정
                if date_mode == "month":
                    month_s = d_start[:6]  # YYYYMM
                    month_e = d_end[:6]
                    page.evaluate(f"document.getElementById('strMonthS').value = '{month_s}'")
                    page.evaluate(f"document.getElementById('strMonthE').value = '{month_e}'")
                else:
                    page.evaluate(f"document.getElementById('strDateS').value = '{d_start}'")
                    page.evaluate(f"document.getElementById('strDateE').value = '{d_end}'")

                with page.expect_navigation(timeout=120000):
                    page.evaluate("goSubmit()")

                page.wait_for_function("typeof goCsvDown === 'function'", timeout=30000)

                with page.expect_download(timeout=120000) as download_info:
                    page.evaluate("goCsvDown()")

                download = download_info.value
                download.save_as(save_path)
                size = os.path.getsize(save_path)

                browser.close()
                return size
        except Exception as e:
            try:
                browser.close()
            except:
                pass
            if attempt == MAX_RETRIES:
                return f"FAIL: {str(e)[:60]}"
    return "FAIL: unknown"


def crawl_page(page_key, page_config, months, workers=None):
    """한 페이지의 전체 월 데이터를 멀티스레드로 크롤링"""
    folder = os.path.join(BASE_DIR, page_config["folder"])
    os.makedirs(folder, exist_ok=True)

    org_no = page_config.get("org_no")
    hokis = page_config.get("hokis")
    date_mode = page_config.get("date_mode", "date")
    num_workers = workers or MAX_WORKERS

    if hokis:
        print(f"\n{'='*60}")
        print(f"[{page_key}] {page_config['desc']} ({len(months)}개월 x {len(hokis)}호기)")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"[{page_key}] {page_config['desc']} ({len(months)}개월)")
        print(f"{'='*60}")

    # 다운로드할 작업 목록 생성
    tasks = []
    total_expected = 0

    if hokis:
        for hoki in hokis:
            for d_start, d_end in months:
                ym = d_start[:6]
                filename = f"{page_key}_{hoki}_{ym}.csv"
                save_path = os.path.join(folder, filename)
                total_expected += 1

                if os.path.exists(save_path) and os.path.getsize(save_path) > 500:
                    print(f"  SKIP: {filename}")
                    continue

                tasks.append((page_config["url"], d_start, d_end, save_path, filename, org_no, hoki, date_mode))
    else:
        for d_start, d_end in months:
            ym = d_start[:6]
            filename = f"{page_key}_{ym}.csv"
            save_path = os.path.join(folder, filename)
            total_expected += 1

            if os.path.exists(save_path) and os.path.getsize(save_path) > 500:
                print(f"  SKIP: {filename}")
                continue

            tasks.append((page_config["url"], d_start, d_end, save_path, filename, None, None, date_mode))

    if not tasks:
        print("  모두 다운로드 완료 상태")
        return total_expected, 0

    success = total_expected - len(tasks)  # 이미 스킵된 것
    fail = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for url, d_start, d_end, save_path, filename, o, h, dm in tasks:
            f = executor.submit(download_one, url, d_start, d_end, save_path, org_no=o, hoki=h, date_mode=dm)
            futures[f] = (filename, d_start, d_end)

        for future in as_completed(futures):
            filename, d_start, d_end = futures[future]
            result = future.result()
            if isinstance(result, int):
                print(f"  OK: {filename} ({result:,} bytes)")
                success += 1
            else:
                print(f"  {result}: {filename}")
                fail += 1

    print(f"  완료: {success} 성공, {fail} 실패")
    return success, fail


def main():
    # 기본값: 태양광 2022-01 ~ 2025-12
    target_pages = sys.argv[1:] if len(sys.argv) > 1 else ["solar_hourly"]
    start_year, start_month = 2022, 1
    end_year, end_month = 2025, 12

    months = generate_months(start_year, start_month, end_year, end_month)

    print(f"기간: {start_year}-{start_month:02d} ~ {end_year}-{end_month:02d} ({len(months)}개월)")
    print(f"대상: {target_pages}")
    print(f"동시 다운로드: {MAX_WORKERS}개")

    total_success = 0
    total_fail = 0

    for page_key in target_pages:
        if page_key not in PAGES:
            print(f"[경고] '{page_key}' 없는 페이지. 건너뜁니다.")
            continue
        config = PAGES[page_key]
        workers = len(config["hokis"]) if config.get("hokis") else MAX_WORKERS
        s, f = crawl_page(page_key, config, months, workers=workers)
        total_success += s
        total_fail += f

    print(f"\n{'='*60}")
    print(f"전체: {total_success} 성공, {total_fail} 실패")

    # 실패 목록
    if total_fail > 0:
        print("\n실패 목록 (재실행하면 자동으로 재시도):")
        for page_key in target_pages:
            if page_key not in PAGES:
                continue
            folder = os.path.join(BASE_DIR, PAGES[page_key]["folder"])
            for d_start, d_end in months:
                ym = d_start[:6]
                path = os.path.join(folder, f"{page_key}_{ym}.csv")
                if not os.path.exists(path) or os.path.getsize(path) <= 500:
                    print(f"  {page_key}_{ym}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
