# PV EDA Plan — 아키텍처 증명 중심

> **EDA의 목적**: "Bayesian 1층 + FiLM 2층 + 시나리오 생성"이 이 데이터에 적합하다는 **정량 근거**를 축적한다.
> 모델 설계가 먼저 있었고 뇌피셜이었으므로, EDA가 아이디어를 **사후 검증**하고 세부를 결정한다.
> 각 노트북 섹션 말미에 "→ 모델 함의" 블록 필수.
> 데이터 범위는 `plan/main/data_strategy.md §2` (호기 12개).

---

## 논리 흐름 (왜 이 순서인가)

```
01 overview
   ↓ [확정된 12 호기, 포트폴리오 규모 감각]
02 feature importance
   ↓ [어떤 변수가 얼마나 중요한가 — icsr 0.88, 습도·지면온도·풍속 등]
03 regional variation
   ↓ [그 중요도가 지역별로 다른가 — 수상 vs 내륙, 위도 차이]
     → "공통이 있지만 지역별 차별화 필요" 근거 확보
04 seasonality
   ↓ [공통 계절 성분이 실제로 강한가 — 80%+ 분산 공유]
     → Bayesian 1층 hierarchical pooling 정당화
05 ghi response
   ↓ [GHI→gen 반응이 선형인가 비선형인가]
     → 1층은 선형 링크 + 2층 비선형 잔차 보정 정당화
06 uncertainty structure
   ↓ [잔차가 heteroscedastic인가, ramp가 얼마나 자주 나오는가]
     → Bayesian variance head + AR(1) 정당화
07 site tilting
   ↓ [글로벌 모델 vs 사이트별 모델 차이]
     → FiLM conditioning 정당화
08 fuel bridge
   ↓ [PV 예측 분포 → LNG 기동 의사결정 인터페이스]
     → 플랫폼 통합
```

---

## 노트북 매핑

| # | 노트북 | 상태 | 핵심 | 증명하는 것 |
|---|--------|------|------|-----------|
| 1 | `01_site_overview.ipynb` | ✅ | 12 호기 확정, 규모 비교 | 학습 대상 데이터 존재 확인 |
| 2 | `02_feature_importance.ipynb` | ✅ | PV × ASOS 호기별 상관, 시간대별 상관 | **icsr이 주 입력**, 이차 변수 순위 |
| 3 | `03_regional_variation.ipynb` | ⏳ 예정 | 수상(고흥만) vs 내륙(예천 등), 지역 클러스터 | **지역별 차별화 필요 → 계층화 정당** |
| 4 | `04_seasonality.ipynb` | ✅ | 사이트 간 공통 계절 패턴 | **공통 성분 80%+ → 1층 pooling 정당** |
| 5 | `05_ghi_response.ipynb` | ✅ | GHI → capacity factor 반응 곡선 | **비선형 존재 → 2층 필요, heteroscedastic → variance head** |
| 6 | `06_uncertainty_structure.ipynb` | ⏳ | 잔차 시간 구조, ramp 이벤트 | **AR(1) 강화 1 정당, variance head 정당** |
| 7 | `07_site_tilting.ipynb` | ⏳ | 글로벌 vs 사이트별 잔차 비교 | **FiLM 2층 conditioning 정당** |
| 8 | `08_fuel_bridge.ipynb` | ⏳ | PV 시나리오 → LNG 의사결정 인터페이스 | **플랫폼 통합 (하류 UC)** |

---

## 모델 아키텍처와 EDA 근거의 대응

| 모델 설계 요소 | EDA 근거 노트북 | 결론 |
|--------------|---------------|------|
| 1층 공통 μ_{r,i}(x) 계층화 | 04 seasonality | 공통 계절 패턴 강함 (PC1 80%+) |
| 1층 지역 random slope b_r | 03 regional variation | 풍속 민감도 고흥만 +30% 등 |
| 1층 Fourier 시간 주기 | 01, 02 | 일주기 명확 |
| 1층 AR(1) 잔차 상관 | 06 uncertainty | ramp event 시 시간 상관 확인 |
| 1층 GP 공간 상관 | 03 + 06 | 같은 날 사이트들 동시 흐림 |
| 1층 Beta / ZIB 분포 | 05 ghi response | 일출/일몰 0 + [0,1] 구간 |
| 1층 heteroscedastic φ | 05 ghi response | 중간 GHI에서 분산 최대 |
| 2층 GRU 비선형 잔차 | 05 + 07 | 잔차 구조적, 비선형 |
| 2층 FiLM/AdaLN-Zero | 07 site tilting | 사이트별 반응 곡선 다름 |
| 2층 probabilistic head | 06 uncertainty | 잔차 분산 조건부 |
| 시나리오 번들 (1000개) | 전체 통합 | Stochastic UC 입력 |

---

## 현재 상황 (2026-04-21 기준)

- **GK-2A GHI 수집 중** (throttle 상태, 느리게 진행) — 수집 완료 전에는 **ASOS `icsr` (9개 호기 커버)** 로 대체 진행 중
- 나머지 3 호기(남해·구미 관측소 icsr 없음)는 GK-2A 도착 후 추가 분석
- 지금 가능한 EDA: 01~05 노트북 (ASOS 기반). 06~08는 모델 학습 이후 또는 GK-2A 이후
- 모델 수식은 `plan/pv/v2.tex`, 수집 근거는 `plan/main/data_strategy.md`

---

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-04-20 | 초안. 6단계. |
| 2026-04-20 | 아키텍처 증명 중심으로 재구성. 7 노트북. |
| 2026-04-21 | **흐름 재정렬**: feature importance → regional variation → seasonality 순서. 모델 아키텍처와 EDA 근거 대응표 추가. 노트북 파일명 재정렬. |
