"""Open-Meteo NWP 예보 API 탐색 — 학습-예보 변수 일관성 확인."""
import sys
import requests
import json
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# 대표 사이트 2곳 (고흥만 수상, 영흥)
sites = [
    {'name': '고흥만',  'lat': 34.57, 'lon': 127.30},
    {'name': '영흥',    'lat': 37.26, 'lon': 126.46},
]

# 학습 데이터(ASOS + GK-2A)와 매칭되는 변수 + 참고용
vars_to_request = [
    'shortwave_radiation',       # GHI (학습: GK-2A)
    'direct_radiation',          # DNI (참고)
    'diffuse_radiation',         # DHI (참고)
    'temperature_2m',            # 기온 (학습: ASOS ta)
    'relative_humidity_2m',      # 습도 (학습: ASOS hm)
    'wind_speed_10m',            # 풍속 (학습: ASOS ws)
    'cloud_cover',               # 운량 (plan상 제외 결정, 확인용)
    'precipitation',             # 강수
    'surface_pressure',          # 기압
]

print("=" * 80)
print("Open-Meteo API 탐색 (예보용 NWP)")
print("=" * 80)

for site in sites:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={site['lat']}&longitude={site['lon']}"
        f"&hourly={','.join(vars_to_request)}"
        f"&forecast_days=2"
        f"&timezone=Asia%2FSeoul"
    )
    print(f"\n>>> {site['name']} ({site['lat']}N, {site['lon']}E)")

    try:
        r = requests.get(url, timeout=15)
        data = r.json()
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    if 'error' in data:
        print(f"  API ERROR: {data}")
        continue

    units  = data.get('hourly_units', {})
    hourly = data.get('hourly', {})
    times  = hourly.get('time', [])
    print(f"  기간: {times[0]} ~ {times[-1]} ({len(times)} 시간)")
    print(f"  {'변수':<25} {'단위':<12} {'24h 최대값'}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")
    for var in vars_to_request:
        if var in hourly:
            values = hourly[var]
            unit = units.get(var, '')
            sample = [v for v in values[:24] if v is not None]
            max_val = max(sample) if sample else None
            print(f"  {var:<25} [{unit:<10}] {max_val}")
        else:
            print(f"  {var:<25} {'MISSING':<12}")

# Archive API도 확인 (MOS 학습용 과거 NWP)
print("\n" + "=" * 80)
print("Archive API 확인 (MOS 편향 보정용 과거 NWP)")
print("=" * 80)
archive_url = (
    f"https://archive-api.open-meteo.com/v1/archive"
    f"?latitude=34.57&longitude=127.30"
    f"&start_date=2024-06-01&end_date=2024-06-02"
    f"&hourly=shortwave_radiation,temperature_2m,relative_humidity_2m,wind_speed_10m"
    f"&timezone=Asia%2FSeoul"
)
try:
    r = requests.get(archive_url, timeout=15)
    d = r.json()
    if 'error' in d:
        print(f"ERROR: {d}")
    else:
        hr = d.get('hourly', {})
        print(f"OK. 2024-06-01~02 고흥만 시간별:")
        print(f"  시간 수: {len(hr.get('time', []))}")
        print(f"  첫 12시 shortwave_radiation: {hr.get('shortwave_radiation', [])[:12]}")
        print(f"  → MOS 학습 가능 (NWP 과거 vs GK-2A 관측 비교)")
except Exception as e:
    print(f"ERROR: {e}")

# 요약
print("\n" + "=" * 80)
print("학습-예보 변수 매칭표")
print("=" * 80)
print(f"{'학습 변수':<20} {'Open-Meteo 예보':<35} {'매칭'}")
print(f"{'-'*20} {'-'*35} {'-'*6}")
print(f"{'GK-2A GHI':<20} {'shortwave_radiation':<35} [OK]")
print(f"{'ASOS 기온(ta)':<20} {'temperature_2m':<35} [OK]")
print(f"{'ASOS 습도(hm)':<20} {'relative_humidity_2m':<35} [OK]")
print(f"{'ASOS 풍속(ws)':<20} {'wind_speed_10m':<35} [OK]")
print(f"{'(사용 안 함)':<20} {'cloud_cover':<35} [N/A] 제공되나 plan상 제외")
print(f"{'(추후)':<20} {'precipitation':<35} [TBD] 단기예보 PTY/PCP와 비교 필요")
