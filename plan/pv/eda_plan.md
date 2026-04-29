# PV EDA Plan v3.2 (2026-04-23 개정)

> 기준일: 2026-04-23
> 상위 계획: `plan/current/plan.md`
> 데이터 SSoT: `plan/current/data_strategy.md`
> 모델 근거: `pv/paper/pv_predict.tex`

## 1. EDA 목적

- 시각화 자체가 아니라 **모델 의사결정 근거**를 생산하는 절차.
- 그래프는 보조, **텍스트 결과(수치/판정)가 주**.
- 각 셀 상단에 **판독 기준**을 명시하고, 실행 후 핵심 수치를 반드시 `print`한다 (예: 상관계수, 피크 월, overlay 재현성 지표).
- Stage마다 Go/No-Go 게이트로 다음 단계 진입 결정.

## 2. 고정 입력 계약

분석 단위: `site-hour` (호기-시간).

필수 컬럼:

- 키: `site`, `unit`, `site_unit`, `datetime_kst`
- 타깃: `gen_kwh`
- 기상/복사: GK-2A `ghi`, ASOS 변수 후보 (Stage 2에서 선별)
- 품질 메타데이터: `source`, `status_primary` (GK-2A v2 출력, `data_strategy.md §4` 참조)

**source 해석 정책** (data_strategy.md §7):
- `exact`, `interp_*`, `zero_seasonal`: 학습 포함
- `unrec_archive_gap`: NaN 표기, 학습 제외 또는 censored 처리
- `unrec_transport`, `unrec_mixed`: resume run 완료 후 자동 해결

## 3. 실행 단계 (4 Stage 선형)

기존 Stage C(DQF sensitivity) 와 Stage D(Trainability Grading)는 폐기. 이유:
- C: 실증으로 DQF가 `{1, NaN}` 2단계뿐 → 민감도 분석할 층위 없음 (Source Sensitivity로 이름 바꿔봤자 4단계뿐 — Stage 3에 흡수).
- D: 사용 가능 11 호기는 이미 `data_strategy.md §2`에서 **실데이터 피크 시각/가동 정상성 기준으로 확정**됨. EDA로 다시 등급 매길 필요 없음.

---

### Stage 1. 포트폴리오 개요 + 시간축 무결성 (노트북 01)

**역할**: "학습할 데이터가 어떤 것이고, 믿을 수 있는가"를 한 노트북에서 확정.

분석:

1. **사용 가능 11 호기 리스트** (`data_strategy.md §2` 참조 — 근거·판정은 거기가 SSoT)
2. **포트폴리오 48개월 발전 패턴** — 전체 합산
3. **고흥만 제외 패턴** — 고흥만이 용량 82% 차지 → 나머지 10 호기 독립 구조 확인
4. **연도별 1년 overlay** — 공통 계절성 재현 검증
   - 핵심 질문: **1년 단위 반복 패턴이 존재하는가?**
   - 포트폴리오(전체/고흥만 제외) 각각에서 확인
5. **호기별 CF 월 overlay** — per-unit stationarity + 호기 간 이질성
6. **시간축 무결성** — KST 일관성, 중복/누락, 정렬
7. **학습 시간대 확정** — 07~18시 + 여름 19시 (`data_strategy.md §4` 0-fill 정책과 동기)

산출물 (텍스트 수치 중심):

- `active_units`: 학습 대상 11 호기 용량/샘플 요약표
- `portfolio_profile`: 월별/연도별 발전량 요약 (총합/일평균/피크월)
- `seasonality_recurrence_score`: 연도 간 월별 CF overlay 유사도 (r, RMSE)
- `per_unit_seasonality`: 호기별 피크 월 + 연도 간 일관성 지표
- `integrity_report`: 시간 규칙 위반 카운트, 중복률, 결측률

Go 조건:

- 11 호기 확정 확인 (`data_strategy.md §2`와 일치)
- 포트폴리오에 **1년 주기 반복 패턴 관측** (공통 계절성 존재)
- 시간 규칙 위반 0건
- 학습 시간대 확정 문서화

---

### Stage 2. 변수 후보 선별 (노트북 02_variable_screening — 신규)

**역할**: 어떤 입력 변수를 모델에 넣을지 결정. Stage 3(세부 반응 분석)보다 **먼저** 해야 하는 이유는 상관 낮은 변수는 세부 분석 자체가 낭비.

분석:

1. **ASOS 변수 전수 나열** — `tm, ta, hm, ws, wd, rn, pa, ps, ss, icsr, dc10Tca, dc10LmcsCa, ts` 등
2. **각 변수 vs `gen_kwh` 상관** (시간별, 사이트별)
3. **GK-2A `ghi` vs `gen_kwh` 상관** — 기준선
4. **ASOS `icsr` vs GK-2A `ghi`** — 두 광원 일관성 확인 (fallback 활용 전제)
5. **결측/이상치 분포** — 학습에 쓸 수 있는 실 커버리지
6. **최종 피처 후보 확정** — 단순 상관 + 물리적 근거 둘 다로

산출물:

- `asos_corr_table`: 변수×사이트 상관 매트릭스
- `feature_shortlist`: 채택 변수 리스트 + 근거 (상관값 + 물리적 해석)
- `missing_profile`: 각 변수의 결측률/이상치 비율
- `ghi_icsr_consistency`: 사이트별 두 소스 상관 + fallback 사용 정책

Go 조건:

- 모델 입력 피처 **1안 확정** (Track A LightGBM에 그대로 투입 가능한 형태)
- 각 피처마다 **"왜 채택했나" 한 줄** 기록

---

### Stage 3. GHI - 발전량 정보차 (노트북 03/04/05)

**역할**: 채택된 GHI(및 기타 피처)가 발전량을 어떻게 설명하는지 세부 파악.

분석:

1. **GHI → 발전량 lag** (0~6h 기본)
2. **시간대/계절별 비선형 반응** (response curve)
3. **사이트 이질성** (공통 구조 vs 사이트 고유)
4. **Source 플래그별 모델 민감도** — `exact`만 vs `interp_*` 포함, 성능 차이

산출물:

- `lag_profile` (사이트별)
- `response_curve` (시간대/계절)
- `variance_decomposition` (공통/고유)
- `source_sensitivity_table` (exact vs interp 성능 차이)
- `site_cluster_assignment` (Track B용)

Go 조건:

- 피처 엔지니어링 방향 확정 (lag, 비선형 변환, 사이트 cluster)
- Source 정책(학습 포함/제외/가중) 1안 확정
- 사이트 cluster 버전 태깅

---

### Stage 4. 모델링 준비 최종 검증 (가볍게)

**역할**: Stage 1~3 결과를 학습 파이프라인에 연결할 수 있는지 마지막 점검.

분석:

1. 학습/검증/테스트 시간 분할 (예: 2022~2024 train / 2025 test)
2. 분할별 샘플 수·커버리지 확인
3. 호기별 최소 샘플 보장

Go 조건:

- **모델 학습 GO** 결정

## 4. 모델 연결 규칙

### Track A (LightGBM 기준선)

EDA 전달 사항:

- Stage 2 `feature_shortlist`
- Stage 3 `lag_profile` (지연 적용 feature)
- Stage 1 학습 시간대 필터

성공 판단:

- 홀드아웃 기준 성능 하한
- 피처 중요도 해석 가능

### Track B (Bayesian Hierarchical)

EDA 전달 사항:

- Stage 3 `variance_decomposition` / `site_cluster_assignment`
- Stage 3 `source_sensitivity_table`에서 도출한 가중 정책
- 저데이터 사이트 그룹 정의

성공 판단:

- 캘리브레이션/커버리지 기준
- 저데이터 사이트 안정성 개선

## 5. 핵심 지표

데이터 무결성 (Stage 1):

- `valid_rate`, `duplicate_rate`, `missing_rate`
- `seasonality_recurrence_score` (연도 간 월별 CF 상관)

변수 선별 (Stage 2):

- 변수별 `pearson_r`, `missing_rate`
- `icsr_ghi_consistency_r`

정보차/구조 (Stage 3):

- `lag_corr_peak`
- `conditional_response_slope`
- `site_heterogeneity_index`

불확실성 (Track B 단계):

- `pinball_loss_p10_p50_p90`
- `coverage_p10_p90`
- `crps`

## 6. 노트북 매핑 (재구성 2026-04-23)

| 노트북 | Stage | 역할 |
|---|---|---|
| `01_site_overview.ipynb` | Stage 1 | 11 호기 + 포트폴리오 + 연도 계절성 + 호기별 CF + 시간 무결성 |
| `02_variable_screening.ipynb` (신규) | Stage 2 | ASOS·GHI 변수 전수 상관 + 피처 shortlist |
| `03_regional_variation.ipynb` | Stage 3 | 사이트 이질성, cluster 후보 |
| `04_seasonality.ipynb` | Stage 3 | 공통 계절·시간 구조 (비선형 반응 일부) |
| `05_ghi_response.ipynb` | Stage 3 | GHI 비선형 반응 + Source Sensitivity |
| `06_uncertainty_structure.ipynb` (필요시) | Stage 3/Track B | 잔차/커버리지 |
| `07_probabilistic_checks.ipynb` (필요시) | Track B | 확률 예측 지표 |

**제거/흡수**:
- 기존 `00_ghi_integrity.ipynb` 별도 안 만들고 **01에 통합** (시간 무결성 섹션)
- 기존 Stage C/D 내용은 Stage 3(Source Sensitivity) 또는 Stage 1(등급은 이미 확정)로 흡수

## 7. 실행 원칙 (코드 작성 지침 포함)

1. **Stage 1 미통과 시 모델 해석 결론 금지** — 시간축/호기 리스트가 흔들리면 나머지 다 재실행.
2. **모든 셀 상단에 `판독 기준`을 주석/마크다운으로 명시** — 그래프 보기 전 "어떤 수치로 무엇을 판정하는지" 고정.
3. **핵심 수치는 반드시 `print`** — 그래프만 띄우지 말 것. 예:
   - "2022 피크월: 5월 / 2023 피크월: 5월 → 계절 재현 OK"
   - "ASOS `icsr` vs GK-2A `ghi` r=0.889 (n=xxx)"
4. **각 노트북 말미에 2개 블록 고정**:
   - `모델 의사결정 영향`
   - `다음 단계 액션`
5. **EDA 결과는 `채택/보류/폐기`**로 결정 로그 남김.
6. **GK-2A 수집 완료(Phase 3) 후 Stage 1부터 전면 재실행**.

## 8. 변경 이력

- 2026-04-22: v3.1 V8 정합형 재작성.
- 2026-04-23: **v3.2 구조 재편** —
  - Stage C(DQF Sensitivity) 폐기 (DQF 실증이 {1, NaN}뿐)
  - Stage D(Trainability Grading) 폐기 (`data_strategy.md §2`에서 이미 확정)
  - Stage 순서 수정: Stage 1(개요+무결성) → Stage 2(**변수 선별 신규**) → Stage 3(GHI 세부) → Stage 4(모델링 준비)
  - 노트북 01 역할 구체화: 11 호기 리스트 + 포트폴리오(전체+고흥만 제외) + 연도별 1년 overlay 계절성 + 호기별 CF overlay + 시간 무결성
  - 코드 작성 지침 명시 — 그래프 보조, 텍스트 출력 필수, 판독 기준 선행
