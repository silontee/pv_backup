# PoC 가치 정량화 — Re-weighting 시스템

## 측정 결과 (2025년 test 데이터, 4,090 daytime hours)

### 1h ahead (가장 의미있는 운영 horizon)
- Imbalance 감소: **632.5 MWh/year**
- 비용 절감:     **104.36 백만원/year**
- 절감률:        **3.9%**
- σ 감소:        **15.8%** (LNG 마진 narrowing)

### 전체 horizon 누적
- Imbalance 감소: 832.2 MWh
- 비용 절감:     137.31 백만원

## PoC 한계 명시

본 PoC 범위: **8 사이트 / 11 호기 / 77.3 MW capacity**.
분당 LNG 920 MW 대비 PV 변동 분담 비율 ≈ 8.4%.

→ 절대 절감액 작아 보이지만, 본 PoC는 **방법론 검증**이 목적.

전국 확장 시 (한국 PV 27 GW, ~350배):
- 추정 1h ahead 절감액: ~37 십억원/year (단순 비례 가정)

## 주요 발견

1. **MAE 개선은 미미** (1h ahead +3.9%) — 평균 예측 정확도는 거의 같음
2. **σ 감소가 진짜 가치** (15.8%) — 분포 narrowing → LNG 보수적 마진 ↓
3. **Re-weighting은 1~2h horizon에 집중** (ACF lag-1 0.63 → ACF 감쇠 빠름)
4. **6h+ horizon은 효과 미미** — 그건 Open-Meteo re-forecast가 담당
