"""
한국남동발전 공공데이터포털 API 데이터 수집 스크립트
- 시간대별 화력 발전실적
- 시간대별 태양광 발전실적
"""
import requests
import json
import os
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DATA_GO_KR_API_KEY")

def fetch_thermal_hourly(date_str):
    """시간대별 화력 발전실적 현황 API (서비스ID: 15130618)"""
    url = "https://apis.data.go.kr/B553061/hourlyThermalPowerGeneration/getHourlyThermalPowerGenerationList"
    params = {
        "serviceKey": API_KEY,
        "numOfRows": 100,
        "pageNo": 1,
        "returnType": "json",
        "searchDate": date_str,  # YYYYMMDD
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        print(f"[화력] Status: {resp.status_code}")
        print(f"[화력] URL: {resp.url[:200]}")
        data = resp.json()
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
        return data
    except Exception as e:
        print(f"[화력] Error: {e}")
        # Try raw text
        try:
            print(f"[화력] Raw response: {resp.text[:1000]}")
        except:
            pass
        return None

def fetch_solar_hourly(date_str):
    """시간대별 태양광 발전실적 현황 API"""
    url = "https://apis.data.go.kr/B553061/hourlySolarPowerGeneration/getHourlySolarPowerGenerationList"
    params = {
        "serviceKey": API_KEY,
        "numOfRows": 100,
        "pageNo": 1,
        "returnType": "json",
        "searchDate": date_str,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        print(f"[태양광] Status: {resp.status_code}")
        print(f"[태양광] URL: {resp.url[:200]}")
        data = resp.json()
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
        return data
    except Exception as e:
        print(f"[태양광] Error: {e}")
        try:
            print(f"[태양광] Raw response: {resp.text[:1000]}")
        except:
            pass
        return None

def fetch_wind_hourly(date_str):
    """시간대별 풍력 발전실적 현황 API"""
    url = "https://apis.data.go.kr/B553061/hourlyWindPowerGeneration/getHourlyWindPowerGenerationList"
    params = {
        "serviceKey": API_KEY,
        "numOfRows": 100,
        "pageNo": 1,
        "returnType": "json",
        "searchDate": date_str,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        print(f"[풍력] Status: {resp.status_code}")
        print(f"[풍력] URL: {resp.url[:200]}")
        data = resp.json()
        print(json.dumps(data, indent=2, ensure_ascii=False)[:3000])
        return data
    except Exception as e:
        print(f"[풍력] Error: {e}")
        try:
            print(f"[풍력] Raw response: {resp.text[:1000]}")
        except:
            pass
        return None

if __name__ == "__main__":
    # 2024년 7월 1일 데이터 시도 (여름 태양광 피크 시즌)
    test_date = "20240701"
    print(f"=== 조회 날짜: {test_date} ===\n")

    print("--- 화력 발전실적 ---")
    fetch_thermal_hourly(test_date)

    print("\n--- 태양광 발전실적 ---")
    fetch_solar_hourly(test_date)

    print("\n--- 풍력 발전실적 ---")
    fetch_wind_hourly(test_date)
