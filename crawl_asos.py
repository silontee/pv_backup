"""
ASOS 시간별 기상 데이터 수집 (공공데이터포털 API)
- 11개 관측소 (태양광 사이트 매칭)
- 시간별, 2022-01 ~ 2025-03
- 변수: 기온, 습도, 풍속, 강수, (GHI/운량은 참고용)
"""

import requests
import json
import os
import sys
import time
import csv
from datetime import datetime, timedelta

API_KEY = "4898a4a156fe4581c07e7fab860f82305ddfd94410601d869563222c2393e892"
API_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "asos_hourly")

# 태양광 사이트 매칭 ASOS 관측소
STATIONS = {
    112: "인천",      # → 영흥태양광 (10.7MW)
    136: "안동",      # → 예천태양광 (2MW)
    155: "창원",      # → 두산엔진MG (0.1MW)
    156: "광주",      # → 탑선태양광 (1MW)
    168: "여수",      # → 여수태양광 (0.2MW)
    192: "진주",      # → 경상대태양광 (0.9MW)
    221: "제천",      # → 영동태양광 (1.1MW)
    262: "고흥",      # → 고흥만 수상태양광 (63.5MW) ※261은 해남(65km), 262가 고흥(5.3km)
    266: "광양",      # → 광양항세방태양광 (3MW)
    279: "구미",      # → 구미태양광 (1MW)
    295: "남해",      # → 삼천포태양광 (14.9MW)
}

# 추출할 필드
FIELDS = [
    ("tm", "시각"),
    ("stnId", "지점번호"),
    ("stnNm", "지점명"),
    ("ta", "기온"),
    ("hm", "습도"),
    ("ws", "풍속"),
    ("wd", "풍향"),
    ("rn", "강수량"),
    ("pa", "현지기압"),
    ("ps", "해면기압"),
    ("ss", "일조시간"),
    ("icsr", "일사량"),
    ("dc10Tca", "전운량"),
    ("dc10LmcsCa", "중하층운량"),
    ("ts", "지면온도"),
]


def fetch_station_month(stn_id: int, year: int, month: int) -> list:
    """한 관측소의 한 달치 시간별 데이터를 API로 가져오기"""
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    start_dt = f"{year:04d}{month:02d}01"
    end_dt = f"{year:04d}{month:02d}{last_day:02d}"

    all_items = []
    page = 1

    while True:
        params = {
            "serviceKey": API_KEY,
            "numOfRows": 999,
            "pageNo": page,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": start_dt,
            "startHh": "00",
            "endDt": end_dt,
            "endHh": "23",
            "stnIds": str(stn_id),
        }

        for attempt in range(3):
            try:
                r = requests.get(API_URL, params=params, timeout=30)
                data = r.json()
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    API 실패: {stn_id} {year}-{month:02d} page={page}: {e}")
                    return all_items

        body = data.get("response", {}).get("body", {})
        if not body:
            break

        items = body.get("items", {}).get("item", [])
        if not items:
            break

        all_items.extend(items)

        total_count = int(body.get("totalCount", 0))
        if len(all_items) >= total_count:
            break
        page += 1

    return all_items


def save_month(year: int, month: int):
    """모든 관측소의 한 달치 데이터를 CSV로 저장"""
    os.makedirs(SAVE_DIR, exist_ok=True)
    filename = f"asos_hourly_{year:04d}{month:02d}.csv"
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        print(f"  건너뜀: {filename}")
        return filepath

    all_rows = []
    for stn_id, stn_name in STATIONS.items():
        print(f"  {stn_name}({stn_id}) 수집 중...", end=" ")
        items = fetch_station_month(stn_id, year, month)
        print(f"{len(items)}행")

        for item in items:
            row = {code: item.get(code, "") for code, _ in FIELDS}
            all_rows.append(row)

        time.sleep(0.5)  # API 부하 방지

    # CSV 저장
    header = [code for code, _ in FIELDS]
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  저장: {filename} ({len(all_rows)}행)")
    return filepath


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        # 단일 월: python crawl_asos.py 2024 7
        year = int(sys.argv[1])
        month = int(sys.argv[2])
        print(f"=== ASOS 수집: {year}-{month:02d} ===")
        save_month(year, month)
    else:
        # 전체: 2022-01 ~ 2025-03
        print("=== ASOS 전체 수집 (2022-01 ~ 2025-03) ===")
        for y in range(2022, 2026):
            end_m = 3 if y == 2025 else 12
            for m in range(1, end_m + 1):
                print(f"\n=== {y}-{m:02d} ===")
                save_month(y, m)
