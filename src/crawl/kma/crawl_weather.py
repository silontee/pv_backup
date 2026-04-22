"""
한국남동발전 일자별 발전소 기상정보 크롤링
- 발전소: 삼천포(SP), 영흥(YH), 영동(YD), 여수(YS), 분당(BD)
- 월 단위 수집, CSV 다운로드 후 UTF-8 변환
"""

import os
import sys
import calendar
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from playwright.sync_api import sync_playwright

URL = "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt18/main.do?menuCd=FN0912020203"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "weather_station")
MAX_RETRIES = 3

# 발전소 코드
STATIONS = {
    "SP": "삼천포",
    "YH": "영흥",
    "YD": "영동",
    "YS": "여수",
    "BD": "분당",
}


def download_month(year: int, month: int, org_code: str = "") -> str:
    """한 달치 기상 데이터 다운로드. org_code 빈값이면 전체."""
    last_day = calendar.monthrange(year, month)[1]
    d_start = f"{year:04d}{month:02d}01"
    d_end = f"{year:04d}{month:02d}{last_day:02d}"

    org_label = STATIONS.get(org_code, "all")
    filename = f"weather_{org_label}_{year:04d}{month:02d}.csv"
    save_path = os.path.join(SAVE_DIR, filename)

    os.makedirs(SAVE_DIR, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        browser = None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()

                print(f"  [{attempt}/{MAX_RETRIES}] {org_label} 페이지 로딩...")
                page.goto(URL, wait_until="networkidle", timeout=60000)
                page.wait_for_function("typeof goSubmit === 'function'", timeout=15000)

                # 날짜 설정
                for field_id, date_val in [("strDateS", d_start), ("strDateE", d_end)]:
                    page.evaluate(f"""
                        var el = document.getElementById('{field_id}');
                        el.removeAttribute('readonly');
                        el.value = '';
                    """)
                    field = page.locator(f"#{field_id}")
                    field.click()
                    field.fill(date_val)
                    page.evaluate(f"document.getElementById('{field_id}').blur()")
                    time.sleep(0.5)

                # 발전소 선택
                if org_code:
                    page.select_option("select[name=strOrgNo]", org_code)
                    time.sleep(0.5)

                time.sleep(1)

                # 조회
                print(f"  [{attempt}/{MAX_RETRIES}] {org_label} 조회 중 ({d_start}~{d_end})...")
                page.evaluate("goSubmit()")
                page.wait_for_load_state("networkidle", timeout=120000)
                time.sleep(3)

                page.wait_for_function("typeof goCsvDown === 'function'", timeout=30000)

                # CSV 다운로드
                print(f"  [{attempt}/{MAX_RETRIES}] {org_label} CSV 다운로드...")
                with page.expect_download(timeout=120000) as download_info:
                    page.evaluate("goCsvDown()")

                download = download_info.value
                temp_path = save_path + ".tmp"
                download.save_as(temp_path)

                browser.close()
                browser = None

                # UTF-8 변환
                convert_to_utf8(temp_path, save_path)
                os.remove(temp_path)

                size = os.path.getsize(save_path)
                with open(save_path, encoding="utf-8") as f:
                    lines = f.readlines()
                print(f"  완료: {filename} ({size:,} bytes, {len(lines)-1}행)")
                return save_path

        except Exception as e:
            if browser:
                try:
                    browser.close()
                except:
                    pass
            print(f"  [{attempt}/{MAX_RETRIES}] 실패: {str(e)[:80]}")
            if attempt < MAX_RETRIES:
                time.sleep(5)

    print(f"  최종 실패: {org_label} {year}-{month:02d}")
    return None


def convert_to_utf8(src_path: str, dst_path: str):
    with open(src_path, "rb") as f:
        raw = f.read()

    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        text = raw.decode("cp949", errors="replace")

    text = text.lstrip("\ufeff")
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # 단일 테스트: python crawl_weather.py 2024 7
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        org = sys.argv[3] if len(sys.argv) >= 4 else ""
        print(f"=== 발전소 기상 수집: {year}-{month:02d} ({org or '전체'}) ===")
        result = download_month(year, month, org)
        if result:
            with open(result, encoding="utf-8") as f:
                lines = f.readlines()
            print(f"\n헤더: {lines[0].strip()[:120]}")
            if len(lines) > 1:
                print(f"첫 행: {lines[1].strip()[:120]}")
    else:
        # 전체 수집: 전체 발전소, 2022~2025
        print("=== 발전소 기상 전체 수집 (2022-01 ~ 2025-03, 전체 발전소) ===")
        months = []
        for y in range(2022, 2026):
            end_m = 3 if y == 2025 else 12
            for m in range(1, end_m + 1):
                filename = f"weather_all_{y:04d}{m:02d}.csv"
                filepath = os.path.join(SAVE_DIR, filename)
                if os.path.exists(filepath):
                    print(f"  건너뜀: {filename}")
                    continue
                months.append((y, m))

        print(f"  수집 대상: {len(months)}개월")
        for y, m in months:
            download_month(y, m, "")  # 전체 발전소
