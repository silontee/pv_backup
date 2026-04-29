# ADR: PV 모델에 Stage 0 (MOS) 추가 — 옵션 C 채택

## Context

플랜 v8(현 `current/plan.md`)는 PV 예측 모델 자체에만 집중하는 Hybrid 2-Track 구조를 확정했다.

- Track A: LightGBM 기준선
- Track B: Bayesian Hierarchical (FiLM-GRU 계열, `pv/paper/pv_predict.tex`)

이 구조는 **학습 입력 = 추론 입력** 가정 하에서만 일관성을 가진다. 그러나 실제 운용에서는 분포 불일치가 발생한다.

| | 학습 단계 | 추론(D-1 예측) 단계 |
|---|---|---|
| GHI 소스 | GK-2A 위성 실측 | Open-Meteo (ECMWF/GFS) NWP 예보 |
| 기온/습도/풍속 | ASOS 실측 | 기상청 단기예보 |

**Distribution shift 두 측면**:

1. **시간대 결손 차이**: GK-2A는 일출/일몰 인근(예: 겨울 17~18시)을 NaN으로 비움. Open-Meteo는 그 시간대도 GHI를 발행함 → 추론 시 모델이 입력받지만 학습 데이터엔 없는 영역.
2. **편향 차이**: NWP는 평활화 경향. 위성 실측 대비 흐림/구름 가장자리에서 체계적 편향. 학습이 실측에만 노출되면 모델은 NWP 편향을 보정할 신호를 못 받음.

GK-2A v1 데이터 검증(2026-04-28):
- 0~5시·19~23시 야간: 100% NaN (자연 현상, 문제 없음)
- 6시·18시 일출/일몰: 41~45% NaN (위성 결손이 아니라 천문학적 결손)
- 9~15시 한낮: 0% NaN
- 6~18시 사이 `sw_dqf=0` 인데 DSR 값 살아있는 174,217행 존재 (dawn/dusk QC reject) — 학습에 포함하기로 결정

영흥 2024-01-15 검증 결과:
- Open-Meteo는 8시(1 W/m²), 17시(143 W/m²), 18시(16 W/m²) 모두 GHI 잡음
- GK-2A는 같은 시간대 NaN
- 한낮(10~14시)은 두 소스 강한 일치, dawn/dusk와 흐림에서 격차 큼

## Decision

PV 예측 파이프라인을 **2-Stage 구조**로 재정의한다.

```
Stage 0 (MOS): historical forecast → bias-corrected GHI/기상
   ├ 학습:  Open-Meteo historical-forecast-api (4년치) → GK-2A/ASOS 실측
   └ 모델:  사이트별 + 시간대별 bias 보정 (XGBoost 또는 quantile regression)

Stage 1 (PV): bias-corrected GHI/기상 → PV 발전량
   ├ 학습:  GK-2A 실측 + ASOS 실측 → KOEN 호기별 PV
   └ 모델:  v8 Hybrid 2-Track 그대로 (Track A: LightGBM, Track B: Bayesian/FiLM-GRU)

추론 시: Open-Meteo 예보 → Stage 0 → Stage 1 → PV
```

옵션 비교에서 옵션 C (현 결정)를 채택한 이유:
- 옵션 A (실측만 학습 → 예보로 추론): distribution shift 무대응. 모델이 NWP 편향에 무지.
- 옵션 B (예보만 학습 → 예보로 추론): 분포 일치하나 GK-2A 실측의 정밀함을 활용 못 함.
- **옵션 C (MOS 보정 후 PV 학습)**: 학계/현장 표준. 책임 분리(MOS=편향, PV=물리), 재학습 주기 분리 가능, GK-2A 가치 보존.

## Consequences

### 즉시 영향

- `current/acceptance.md` 폐기 (v8 Hybrid 2-Track 종속, MOS 추가 후 재작성).
- `plan/pv/plan_v1.md` 신설 — 옵션 C 큰 틀과 EDA 검증 항목 정의.
- `current/plan.md`, `current/data_strategy.md` 상단에 PIVOT 배너 추가. 본문 갱신은 `pv/plan_v1.md` 안정화 후.
- `CHANGELOG.md` 2026-04-28 항목 추가.

### EDA에서 검증할 것 (Go/No-Go 게이트)

1. Open-Meteo `historical-forecast-api` (실제 발행된 과거 예보)가 dawn/dusk GHI를 잡는지 검증. archive(ERA5)와 다르므로 별도 확인 필요.
2. MOS 학습 가능성: 예보-실측 편향이 reproducible 패턴인지 random noise인지. 사이트별/시간대별/계절별 편향 구조 분석.
3. MOS 통과 후 PV 모델 성능이 v8 baseline 이상.

세 게이트 모두 통과해야 옵션 C 정식 채택.

### 미해결 (보류)

- `data_strategy.md` 본문 갱신: Open-Meteo historical-forecast-api 섹션, MOS 학습 데이터 계약. EDA 결과 반영 후 일괄 갱신.
- `plan/pv/eda_plan.md` v3.2 갱신: forecast-vs-observation 정합 Stage 추가. EDA 진행 중 점진 갱신.
- `plan_v1.md`이 검증 통과 시 → `current/plan.md` (v9)로 승격, 기존 v8은 `history/main/plan_v8.md`로 이동.

### 롤백 조건

위 EDA 게이트 중 하나라도 실패 시:
- `plan/pv/plan_v1.md` 폐기 또는 옵션 A/B로 재설계
- `current/` 파일들 배너 제거, v8 그대로 진행
- ADR에 "실패: <사유>" 기록 추가
