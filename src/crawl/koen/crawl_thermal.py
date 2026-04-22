"""
한국남동발전 시간대별 화력발전실적 크롤링
- 발전구분: 전체 (분당화력만 있음)
- 호기: 전체
- 월 단위 수집, CSV 다운로드 후 UTF-8 변환
"""

import os
import sys
import calendar
import time
from playwright.sync_api import sync_playwright

URL = "https://www.koenergy.kr/kosep/gv/nf/dt/nfdt26/main.do?menuCd=FN0912020221"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "thermal_hourly")
MAX_RETRIES = 3


def download_month(year: int, month: int) -> str:
    """한 달치 데이터를 다운로드하고 UTF-8로 변환"""
    last_day = calendar.monthrange(year, month)[1]
    d_start = f"{year:04d}{month:02d}01"
    d_end = f"{year:04d}{month:02d}{last_day:02d}"
    filename = f"thermal_hourly_{year:04d}{month:02d}.csv"
    save_path = os.path.join(SAVE_DIR, filename)

    os.makedirs(SAVE_DIR, exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        browser = None
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(accept_downloads=True)
                page = context.new_page()

                # 1. 페이지 로드
                print(f"  [{attempt}/{MAX_RETRIES}] 페이지 로딩...")
                page.goto(URL, wait_until="networkidle", timeout=60000)

                # 2. goSubmit, goCsvDown 함수가 존재할 때까지 대기
                page.wait_for_function("typeof goSubmit === 'function'", timeout=15000)

                # 3. 날짜 설정
                # readonly 해제 → 필드 클리어 → 타이핑 → blur
                for field_id, date_val in [("strDateS", d_start), ("strDateE", d_end)]:
                    page.evaluate(f"""
                        var el = document.getElementById('{field_id}');
                        el.removeAttribute('readonly');
                        el.value = '';
                    """)
                    field = page.locator(f"#{field_id}")
                    field.click()
                    field.fill(date_val)
                    # blur로 포커스 이탈 (달력 위젯 반영)
                    page.evaluate(f"document.getElementById('{field_id}').blur()")
                    time.sleep(0.5)
                time.sleep(1)

                # 4. 발전구분 = 전체 (첫번째 옵션 또는 빈값)
                # 드롭다운에서 '전체'를 선택 - 보통 value=""가 전체
                try:
                    page.select_option("select[name=strOrgNo]", "")
                except:
                    pass  # 이미 전체일 수 있음

                # 5. 호기도 전체
                try:
                    page.select_option("select[name=strHokiS]", "")
                except:
                    pass
                try:
                    page.select_option("select[name=strHokiE]", "")
                except:
                    pass

                time.sleep(0.5)

                # 6. 조회 실행
                print(f"  [{attempt}/{MAX_RETRIES}] 조회 중 ({d_start}~{d_end})...")
                page.evaluate("goSubmit()")

                # 7. 테이블 로드 대기 (조회 결과가 나올 때까지)
                # networkidle 대신 테이블 데이터가 로드될 때까지 대기
                page.wait_for_load_state("networkidle", timeout=120000)
                time.sleep(3)  # 추가 대기 (데이터 렌더링)

                # 8. goCsvDown 함수 대기
                page.wait_for_function("typeof goCsvDown === 'function'", timeout=30000)

                # 9. CSV 다운로드
                print(f"  [{attempt}/{MAX_RETRIES}] CSV 다운로드...")
                with page.expect_download(timeout=120000) as download_info:
                    page.evaluate("goCsvDown()")

                download = download_info.value
                temp_path = save_path + ".tmp"
                download.save_as(temp_path)

                browser.close()
                browser = None

                # 10. UTF-8 변환
                convert_to_utf8(temp_path, save_path)
                os.remove(temp_path)

                size = os.path.getsize(save_path)
                # 간단한 품질 체크
                with open(save_path, encoding="utf-8") as f:
                    lines = f.readlines()
                data_rows = len(lines) - 1
                print(f"  완료: {filename} ({size:,} bytes, {data_rows}행)")
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

    print(f"  ❌ 최종 실패: {year}-{month:02d}")
    return None


def convert_to_utf8(src_path: str, dst_path: str):
    """CP949/EUC-KR CSV를 UTF-8로 변환"""
    with open(src_path, "rb") as f:
        raw = f.read()

    # 인코딩 감지
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, ValueError):
            continue
    else:
        text = raw.decode("cp949", errors="replace")

    # BOM 제거 후 UTF-8로 저장
    text = text.lstrip("\ufeff")
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def collect_all(start_year=2020, end_year=2024, max_workers=3):
    """전체 기간 수집 (멀티스레드)"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 수집할 월 목록
    months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            filename = f"thermal_hourly_{y:04d}{m:02d}.csv"
            filepath = os.path.join(SAVE_DIR, filename)
            if os.path.exists(filepath):
                print(f"  건너뜀 (이미 존재): {filename}")
                continue
            months.append((y, m))

    print(f"\n수집 대상: {len(months)}개월 ({start_year}-01 ~ {end_year}-12)")
    print(f"스레드: {max_workers}개\n")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_month, y, m): (y, m) for y, m in months}
        for future in as_completed(futures):
            y, m = futures[future]
            key = f"{y}-{m:02d}"
            try:
                result = future.result()
                results[key] = result
            except Exception as e:
                results[key] = None
                print(f"  ❌ {key}: {e}")

    # 품질 리포트
    print("\n" + "=" * 60)
    print("수집 결과 리포트")
    print("=" * 60)

    warnings = []
    for key in sorted(results.keys()):
        path = results[key]
        if not path:
            print(f"  ❌ {key}: 실패")
            continue

        with open(path, encoding="utf-8") as f:
            lines = f.readlines()

        total = len(lines) - 1
        nonzero = 0
        for line in lines[1:]:
            fields = line.strip().split(",")
            if len(fields) > 24:
                try:
                    val = float(fields[24].strip()) if fields[24].strip() else 0
                    if val > 0:
                        nonzero += 1
                except:
                    pass

        pct = nonzero / total * 100 if total > 0 else 0
        flag = " ⚠️" if pct < 10 else ""
        print(f"  {key}: {total}행, 값>0: {nonzero} ({pct:.1f}%){flag}")
        if pct < 10:
            warnings.append((key, total, nonzero, pct))

    if warnings:
        print(f"\n⚠️  값>0 비율이 10% 미만인 달 ({len(warnings)}개):")
        for key, total, nonzero, pct in warnings:
            print(f"  {key}: {nonzero}/{total} ({pct:.1f}%)")

    print(f"\n완료: {sum(1 for v in results.values() if v)}/{len(results)}개월 성공")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # 단일 월 테스트
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        print(f"=== 화력 발전실적 수집: {year}-{month:02d} ===")
        download_month(year, month)
    else:
        # 전체 수집 (2020~2024)
        print("=== 화력 발전실적 전체 수집 (2022-01 ~ 2024-12) ===")
        collect_all(start_year=2022, end_year=2024, max_workers=3)
