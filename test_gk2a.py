"""GK-2A 수집 전 2회 테스트:
1. 2024-07-01 12:00 KST — IP 차단 해제 여부 & 정상 수집 확인
2. 2025-01-01 12:00 KST — 2025년 API 제공 여부 확인
각 테스트는 키 1개 사용 + 용량 카운트.
"""
import sys
from crawl_gk2a import KeyManager, API_KEYS, BYTES_LIMIT_PER_KEY, kst_to_utc_str, download_nc, extract_ghi
import os, tempfile

sys.stdout.reconfigure(encoding='utf-8')

km = KeyManager(API_KEYS, BYTES_LIMIT_PER_KEY)

def test_one(label, year, month, day, hour):
    key = km.get_usable_key()
    utc_str = kst_to_utc_str(year, month, day, hour)
    print(f"[{label}] {year}-{month:02d}-{day:02d} {hour:02d}:00 KST → UTC {utc_str}")
    nc_path, nbytes = download_nc(utc_str, key)
    if nc_path:
        km.add_usage(key, nbytes)
        ghi = extract_ghi(nc_path)
        os.unlink(nc_path)
        print(f"  ✅ 성공: {nbytes/1024:.1f} KB")
        # 고흥만 값 샘플
        g = ghi.get('고흥만수상')
        print(f"  샘플 고흥만 DSR = {g} W/m²")
        return True
    else:
        print(f"  ❌ 실패 (IP 차단 또는 데이터 없음)")
        return False

ok1 = test_one("TEST 1: IP 상태 체크 (2024)", 2024, 7, 1, 12)
ok2 = test_one("TEST 2: 2025 API 제공 여부", 2025, 1, 1, 12)

print(f"\n총 사용: {km.total_usage_gb()*1024:.2f} MB")
print(f"남은 쿼터: {km.remaining_gb():.2f} GB")

if ok1 and ok2:
    print("\n→ 모든 테스트 성공. 2022-01~2025-12 수집 가능.")
    sys.exit(0)
elif ok1 and not ok2:
    print("\n→ 2024는 OK, 2025 API 미제공. 2022-01~2024-12로 수집.")
    sys.exit(2)
else:
    print("\n→ IP 차단 상태. 수집 불가.")
    sys.exit(1)
