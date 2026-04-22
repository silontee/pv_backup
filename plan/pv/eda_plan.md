# PV EDA Plan v3.1 (V8 정합)

> 기준일: 2026-04-22
> 상위 계획: `plan/main/plan_v8.md`
> 데이터 SSoT: `plan/main/data_strategy.md`
> 모델 근거: `pv/pv_predict.md`

## 1. EDA 목적

- 목적은 시각화 자체가 아니라 모델 의사결정 근거를 생산하는 것이다.
- 특히 `GHI ↔ 발전량 정보차`를 정량화해 다음 결정을 내린다.
  - Data Contract 확정
  - Track A/Track B 입력 범위 확정
  - 유닛 A/B/C 등급 확정

## 2. 고정 입력 계약 (EDA 관점)

분석 단위는 `site-hour`로 통일한다.

필수 컬럼:

- 키: `site`, `unit`, `site_unit`, `site_cluster`, `datetime_kst`
- 기상/복사: `ghi`, `temp_c`, `rh`, `wind_ms`
- 품질: `quality_mask`, `dqf_class`, `source_flag`
- 타깃: `gen_kwh_hourly` (선택 병행: `gen_kw_norm`)

계약 위반 배치는 EDA 집계에서 제외하고 별도 오류 테이블에 기록한다.

## 3. 실행 단계와 Go/No-Go

## Stage A. Time Integrity Gate (최우선)

분석:

1. 타임존 일관성(KST) 검증
2. 리샘플 규칙(정시 기준) 검증
3. 누락/중복/역정렬 이벤트 검증

산출물:

- `integrity_summary` (site/unit별 연속성, 중복률, 결측률)
- `integrity_anomaly_log` (규칙 위반 내역)

Go 조건:

- 필수 키/시간 규칙 위반 0건
- 중복 처리 규칙 문서화 완료

## Stage B. Information Gap Diagnosis

분석:

1. GHI-발전량 lag 분석(기본 0~6h)
2. 시간대/계절별 비선형 반응
3. 사이트별 분산 분해(공통/사이트 고유)

산출물:

- 사이트별 `lag_profile`
- 시간대/계절별 `response_curve`
- `variance_decomposition`
- `site_cluster_assignment` (cluster 라벨/근거/버전)

Go 조건:

- 주요 lag와 유효 반응 구간이 사이트별로 정의됨
- 공통 구조와 사이트 고유 구조가 분리됨
- `site_cluster` 정의 및 버전 태깅 완료

## Stage C. Quality Sensitivity (DQF 중심)

분석:

1. DQF 클래스별 표본 분포 비교
2. DQF 클래스별 모델 민감도 비교
3. 재수집 전/후 분포 드리프트 비교

산출물:

- `dqf_distribution_report`
- `dqf_sensitivity_table`
- `pre_post_retry_drift`

Go 조건:

- DQF 정책(제외/감쇠/보정) 1안 확정
- 정책이 성능/커버리지에 미치는 영향 확인

## Stage D. Trainability Grading

분석:

1. 유효 샘플 수
2. 품질 구간 비율
3. 계절 커버리지

산출물:

- `unit_trainability_grade` (A/B/C)
- `model_inclusion_list` (Track A/Track B 투입 범위)

Go 조건:

- 모든 유닛에 등급 부여 완료
- 학습 투입/제외 기준 확정

## 4. 모델 연결 규칙

## Track A (LightGBM 기준선)

EDA에서 전달받는 것:

- 안정 입력 피처 목록
- 제외 구간(quality_mask=0 또는 C 등급)
- 사이트별 최소 샘플 기준

성공 판단:

- 홀드아웃 기준 성능 하한 충족
- 피처 중요도 해석 가능

## Track B (Bayesian Hierarchical)

EDA에서 전달받는 것:

- 사이트 이질성 구조(분산 분해)
- 품질 불확실성 분포(DQF 기반)
- 저데이터 사이트 그룹 정의
- `site_cluster_assignment` 및 cluster별 샘플 커버리지

성공 판단:

- 캘리브레이션/커버리지 기준 충족
- 저데이터 사이트 안정성 개선

## 5. 핵심 지표 세트

데이터 무결성:

- `valid_rate`
- `missing_rate`
- `duplicate_rate`
- `dqf_accept_rate`

정보차/구조:

- `lag_corr_peak`
- `conditional_response_slope`
- `site_heterogeneity_index`

불확실성:

- `pinball_loss_p10_p50_p90`
- `coverage_p10_p90`
- `crps`

운영 연결:

- `scenario_spread`
- `risk_band_stability`

## 6. 노트북 매핑

- `01_site_overview.ipynb`: Stage A/B 기본 구조
- `02_feature_importance.ipynb`: Track A 입력 타당성
- `03_regional_variation.ipynb`: 사이트 이질성, 계층 근거
- `04_seasonality.ipynb`: 공통 계절/시간 구조
- `05_ghi_response.ipynb`: 비선형 반응 및 품질 민감도

신규 권장:

- `00_ghi_integrity.ipynb`: Stage A 전용 게이트
- `06_uncertainty_structure.ipynb`: 잔차 구조/커버리지
- `07_probabilistic_checks.ipynb`: 확률 예측 지표

## 7. 실행 원칙

1. Stage A 미통과 상태에서는 모델 해석 결론을 확정하지 않는다.
2. 모든 노트북 마지막 셀에 아래 2개를 고정 작성한다.
   - `모델 의사결정 영향`
   - `다음 단계 액션`
3. EDA 결과는 반드시 `채택/보류/폐기`로 의사결정 로그를 남긴다.
4. 재수집 완료 후 Stage A~D를 동일 순서로 재실행한다.

## 8. 변경 이력

- 2026-04-22: V8 정합형 v3.1 재작성 (Data Contract, Go/No-Go, Track 연계 명시)
