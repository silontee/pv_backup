# PV 예측 Plan v1 — 2-Stage (MOS + PV) 구조

> 기준일: 2026-04-28
> 상위 계획: `plan/current/plan.md` (v8, Hybrid 2-Track) — 본 v1에서 Stage 1로 흡수
> 결정 근거: `plan/decisions/2026-04-28-pv-mos-stage.md`
> 데이터 SSoT: `plan/current/data_strategy.md` (forecast 섹션은 본 v1 안정화 후 갱신)
> 상태: **DRAFT** — EDA Go/No-Go 게이트 통과 시 `current/plan.md` (v9)로 승격

---

## 1. 목적

학습 분포(GK-2A 실측)와 추론 분포(Open-Meteo 예보) 간의 distribution shift를 명시적으로 다루는 PV 예측 파이프라인을 정의한다.

핵심 가설: **예보-실측 편향은 reproducible 패턴**이며, 4년치 historical forecast로 학습 가능하다.

가설이 깨지면(편향이 random noise라면) 옵션 C는 기각하고 옵션 A 또는 B로 회귀한다.

---

## 2. 2-Stage 아키텍처

```
[추론 시점 (D-1)]

  Open-Meteo 예보  ──┐
  (GHI, T, RH, WS)   │
                     ▼
            ┌──────────────┐
            │   Stage 0    │  사이트별·시간대별
            │     MOS      │  bias 보정
            │  (보정 모델) │
            └──────┬───────┘
                   ▼
       bias-corrected GHI + 기상
                   │
                   ▼
            ┌──────────────┐
            │   Stage 1    │  v8 Hybrid 2-Track
            │      PV      │  (Track A: LGBM)
            │  (예측 모델) │  (Track B: Bayesian/FiLM-GRU)
            └──────┬───────┘
                   ▼
            PV 발전량 예측
            + 불확실성 구간
```

### Stage 0 (MOS)

- **입력**: Open-Meteo `historical-forecast-api` 시간별 예보
  - `shortwave_radiation` (GHI), `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`
  - 메타: lead time, 발행 시각
- **타깃**: GK-2A `dsr` 시간 평균 (사이트 픽셀), ASOS 실측
- **모델 후보**: XGBoost (사이트·시간대 conditioning), quantile regression, 또는 가벼운 MLP
- **산출**: 사이트별·시간별 보정된 GHI/기상 + 보정 후 잔차 분포 (불확실성 전파용)

### Stage 1 (PV)

- v8의 Hybrid 2-Track 그대로 흡수 (재기술 X, `current/plan.md` §5 참조)
- 입력만 변경: GK-2A 실측 → MOS 출력 (학습 시 *bypass*하여 GK-2A 실측 직접 사용 가능)
- Track A: LightGBM 기준선
- Track B: Bayesian Hierarchical (FiLM-GRU)

**학습 시 분리**: Stage 0과 Stage 1은 별도 학습. 추론 시에만 직렬.
- 이유: Stage 1은 GK-2A 실측을 정답으로 학습할 수 있어 데이터 효율 ↑.
- 추론 시 Stage 0의 잔차 분포가 Stage 1의 입력 불확실성으로 전파.

---

## 3. EDA Go/No-Go 게이트 (옵션 C 채택 검증)

ADR 2026-04-28-pv-mos-stage.md의 3개 게이트를 EDA 노트북에서 명시 검증한다.

### Gate 1. Open-Meteo historical-forecast-api 커버리지

**노트북**: `pv/notebooks/07_om_forecast_inspect.ipynb` (신규)

- 영흥 2024-01-15 (겨울, 일출 늦음) 한 시점만 historical-forecast-api로 받기
- archive-api (ERA5) 결과와 비교 — 발행 시점·예보 lead time·dawn/dusk GHI 차이
- **Pass 기준**: dawn/dusk(7~8시, 17~18시) GHI > 0 nonzero 시간대 ≥ 80%

이 게이트가 실패하면 (예보 자체가 dawn/dusk 못 잡으면) 옵션 C 의미 없음 → 옵션 A로 회귀.

### Gate 2. MOS 편향의 학습 가능성

**노트북**: `pv/notebooks/08_mos_feasibility.ipynb` (신규)

- Open-Meteo historical forecast 4년치 부분 표본 수집 (대표 사이트 3개 × 1년)
- GK-2A v1 실측과 시간 정렬 → 잔차 = (실측 - 예보) 계산
- 잔차 구조 분석:
  - 사이트별 평균 편향 (시스템적 bias 존재 여부)
  - 시간대별 편향 (오전/오후/dawn/dusk 차이)
  - 계절별 편향 (겨울 vs 여름)
  - 운량 조건별 편향 (Open-Meteo `cloud_cover` 사용)
- **Pass 기준**: 잔차의 R² > 0.3 with 위 conditioning features (편향이 reproducible)
- **Fail 기준**: 잔차가 random noise (R² < 0.1) → MOS 학습 불가, 옵션 A로 회귀

### Gate 3. MOS 통과 PV 성능

**노트북**: `pv/notebooks/09_mos_pv_pipeline_baseline.ipynb` (신규, Gate 1·2 통과 후)

- Stage 0 (간단한 XGBoost MOS) → Stage 1 (LightGBM 기준선) 직렬 파이프라인
- 비교 baseline: GK-2A 실측 직접 사용한 v8 Track A 학습 결과
- **Pass 기준**: 추론 시 MOS-PV 파이프라인 NMAE가 v8 직접 학습 NMAE의 1.2배 이내
  - (분포 일치로 인한 generalization 이득이 MOS 노이즈를 상쇄해야 의미)

세 게이트 모두 Pass 시 → 옵션 C 정식 채택, plan_v1 → current 승격.

---

## 4. 데이터 추가 수집 (Gate 1·2 통과 시)

### Open-Meteo historical-forecast-api 4년치

- 11 사이트 × 4년 (2022-01 ~ 2025-12)
- 시간별 (24h × 365 × 4 = 35,040 시간 × 11 사이트 = 385,440 행)
- 변수: GHI, T_2m, RH_2m, WS_10m, cloud_cover (참고)
- 무료 API, rate limit 고려해서 batch 수집
- 저장: `data/openmeteo_historical_forecast/<site>_<YYYYMM>.csv` 형식 (TBD)

### data_strategy.md 갱신 항목 (Gate 통과 후)

- §1 학습용 표에 `Open-Meteo historical forecast` 행 추가
- §4 변수별 소스 선택 근거에 MOS 학습 입력 명시
- §7 학습-예보 소스 불일치 해결 섹션 갱신: "MOS 학습으로 보정"

---

## 5. 모델 학습 분리 전략

### Stage 0 학습

- **입력**: Open-Meteo historical forecast at issue_time, lead_time
- **정답**: GK-2A 시간 평균 (status_v2='usable' 행만, weight 적용)
  - `ok` (sw_dqf=1): weight 1.0
  - `sw_dqf_reject` (sw_dqf=0, dsr 살아있음): weight 0.5
  - `nan_value`: 학습 제외 (입력 정답 없음)
- **분할**: site-stratified time-based split (2022~2024 학습, 2025 검증)
- **평가**: 잔차 RMSE, 잔차 calibration (예측 분산 vs 실제 잔차 분산)

### Stage 1 학습

- **입력**: GK-2A 실측 직접 사용 (Stage 0 우회)
  - 이유: Stage 1이 다루는 문제 = "정확한 GHI일 때 PV 어떻게 나오는가"
  - MOS 출력으로 학습하면 MOS 노이즈가 PV 모델에 누적
- **정답**: KOEN solar_hourly 호기별 PV
- **분할/평가**: v8 §8 그대로

### 추론 (D-1)

```
1. Open-Meteo forecast API 호출 → 24~48h 시간별 예보
2. Stage 0 적용 → 보정된 GHI, T, RH, WS
3. Stage 1 적용 → PV 점예측 + 신뢰구간
4. (선택) Stage 0 잔차 분포 + Stage 1 신뢰구간 합성 → 최종 불확실성 구간
```

---

## 6. v8 자산 보존

다음은 v1에서도 그대로 유효:

- v8 §3 Data Contract (site-hour 단위, 필수 컬럼) — Stage 1에서 그대로
- v8 §4 EDA V3 Stage A~D — Stage 1 학습 데이터 검증에 그대로
- v8 §5 Hybrid 2-Track (Track A LGBM, Track B Bayesian) — Stage 1 모델 후보
- v8 §6 베이지안/계층 채택 이유 — Stage 1 Track B 정당화로 그대로
- v8 §7 운영 문서화 규칙 — 그대로

본 v1이 추가하는 것: **Stage 0 신설 + 학습-예보 정합 게이트 추가**.

---

## 7. 미해결 / Open Questions

- **MOS 모델 선택**: XGBoost vs quantile regression vs MLP. Gate 2 결과 보고 결정.
- **Stage 0 출력 단위**: 점예측만 vs 잔차 분포까지. 후자가 Stage 1에 불확실성 전파에 유리하나 학습 복잡도 ↑.
- **lead time 처리**: D-1 10시 vs 17시 예보의 차이. 두 시점을 별도 모델로 학습할지 lead time을 feature로 줄지.
- **사이트 일반화**: 11 사이트 각각 따로 학습 vs 사이트 임베딩 공유. 데이터 양 보고 결정.
- **운량 변수**: `cloud_cover` Open-Meteo에 있으나 data_strategy.md §4에 "GHI 직접 사용하므로 운량 불필요" 명시. MOS에선 conditioning에 쓸 가치 있을 수 있음. Gate 2에서 검증.

---

## 8. 다음 작업 순서

1. **Gate 1 노트북** (`07_om_forecast_inspect.ipynb`) — 1일치 historical-forecast-api 검증
2. Gate 1 Pass 시 → **Gate 2 노트북** (`08_mos_feasibility.ipynb`) — 1년치 표본 수집·잔차 분석
3. Gate 2 Pass 시 → **Open-Meteo 4년치 대량 수집** + **Gate 3 노트북** 시작
4. Gate 3 Pass 시 → 본 v1을 `current/plan.md` (v9)로 승격, v8을 history로 이동
5. 모든 Gate 통과 후 `data_strategy.md` 일괄 갱신
