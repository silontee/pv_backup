# PV 변동성 대응형 예측·운영 지원 플랫폼 기획서 초안

> 기준 문서
> - PV forecasting: `plan/active/pv/model_final.md`
> - LNG planner: `plan/active/lng/plan.md`
> - 상위 framing: `plan/active/main/framing.md`
> - 데이터 전략: `plan/active/main/data_strategy.md`
>
> 본 문서는 현재까지 확정된 최종 production 구조를 기준으로 공모전 제출용 기획서 본문 초안을 다시 정리한 것이다.
> 실험 과정에서 검토했으나 채택하지 않은 대안은 별도 구분하여 기록한다.

---

## 1. 필요성

### 1.1 왜 이 문제가 중요한가

재생에너지 비중이 증가할수록 전력 운영에서 가장 어려워지는 지점은 평균 발전량이 아니라 `짧은 시간 안에 발생하는 계획 이탈`이다. 특히 태양광은 일사 조건이 양호한 날에도 부분 흐림, 구름 통과, site-level 설비 정지, 계측 이상 등의 영향으로 정오 전후에 급격한 출력 저하와 회복을 보인다. 이때 운영자 관점에서 중요한 것은 “오늘 하루 총 발전량이 얼마나 맞았는가”가 아니라 다음 세 가지다.

1. 전일 기준으로 세운 발전 계획과 현재 실적의 차이가 얼마나 벌어졌는가
2. 그 편차가 앞으로 1~3시간, 나아가 남은 하루 동안 지속될 것인가
3. 그 편차를 보완하기 위해 화력 자원을 언제, 얼마나, 어떤 순서로 준비해야 하는가

즉 실제 운영 문제는 단일 예측 모델의 leaderboard 점수를 올리는 문제가 아니다. 실제 문제는:

> **D-1 계획 대비 실시간 편차를 감지하고, 남은 시간의 위험을 다시 예측한 뒤, 그 결과를 LNG 백업 계획과 호기별 재배분 의사결정으로 연결하는 것**

이다.

### 1.2 공모전 요구와의 연결

남동발전이 제시한 과제는 크게 두 줄로 요약된다.

1. 시간대별 신재생 발전 실적과 기상 요인을 연계한 AI 기반 발전량 예측 알고리즘 구축
2. 신재생 출력 변동 시나리오에 따라 화력 발전 부하를 최적으로 배분하는 운영 로직 설계

이 요구를 그대로 해석하면, 예측과 운영을 분리해서 따로 설명하는 방식은 설득력이 약하다. 예측만 잘해도 운영에 연결되지 않으면 실제 활용 가치가 떨어지고, 반대로 운영 로직만 있어도 편차를 미리 해석할 수 없으면 선제 대응이 어렵다. 따라서 본 과제는 처음부터 다음의 일체형 구조를 목표로 했다.

```text
D-1 baseline forecast
-> intraday reforecast
-> outage / event 구분
-> gap 해석
-> LNG backup planning
-> 호기별 재배분
-> dashboard 기반 운영 판단
```

즉 본 과제의 핵심은 “예측 모델을 하나 더 만들었다”가 아니라, **예측 결과가 운영 레이어로 직접 이어지는 구조를 PoC 수준에서 입증한다**는 데 있다.

### 1.3 PoC가 필요한 이유

남동발전과의 질의응답에서 확인된 조건은 분명했다.

- SPC 데이터 등 실제 내부 데이터는 제공되지 않음
- 발전기별 실제 운전 파라미터도 공공데이터가 아님
- 기대 산출물은 상용 시스템이 아니라 **대시보드가 포함된 데모 가능한 PoC**

이 조건은 오히려 프로젝트의 목표를 명확하게 만들어 준다. 즉 본 과제는 실제 시장 최적화기나 KPX dispatch 재현기가 아니라, **공공데이터와 공개 실적만으로도 실시간 운영지원 구조를 설계할 수 있는지 검증하는 개념검증(Proof of Concept)** 이어야 한다.

### 1.4 본 과제가 다루는 실제 문제 정의

따라서 본 과제는 다음처럼 정의하는 것이 가장 정확하다.

> **태양광 D-1 기준 계획 대비 실시간 편차를 감지하고, 남은 하루의 출력 곡선을 재예측하며, LNG 기준 운전상태 위에서 필요한 추가 대응량을 계산해 호기별로 재배분하는 운영 지원형 AI 파이프라인의 PoC**

이 정의에는 세 가지 중요한 제한이 포함된다.

1. 전국 계통 전체 최적화를 직접 수행하지 않는다.
2. 실제 D-1 thermal plan을 복원하지 않는다.
3. 특정 발전소의 실 dispatch를 정답으로 맞히려 하지 않는다.

반대로 다음 세 가지는 분명히 포함한다.

1. D-1 baseline forecast
2. intraday remaining-day reforecast
3. LNG baseline 위 추가 대응량의 short-horizon allocation

---

## 2. 목적

### 2.1 정량적 목적

본 과제의 정량적 목적은 다음과 같다.

1. 태양광 D-1 baseline 대비 intraday 재예측으로 전체 예측 오차를 줄인다.
2. weather-driven event와 non-meteorological outage-like event를 분리해 오판을 줄인다.
3. planner 기준 shortfall을 줄이고 sign flip 및 과도한 진동을 최소화한다.
4. 단일 화력 블록 가정보다 현실적인 LNG fleet 기반 재배분 구조를 제시한다.

### 2.2 정성적 목적

정성적 목적은 다음과 같다.

1. “AI 예측”과 “운영 의사결정”을 연결하는 하나의 스토리를 제시한다.
2. 운영자가 실제로 이해할 수 있는 구조로 event, blackout, backup action을 해석한다.
3. 향후 실제 데이터 연동이 가능해질 경우 어떤 부분이 바로 확장 가능한지 보여준다.

### 2.3 최종 산출물 관점 목적

최종 산출물은 단순 보고서가 아니라 다음을 동시에 만족하는 PoC를 목표로 한다.

1. 예측모델 설명 가능
2. planner 로직 설명 가능
3. KPI로 검증 가능
4. dashboard에서 시연 가능

즉 본 과제는 논문형 단일 모델보다 **서비스형 구조 검증**에 더 가깝다.

---

## 3. 세부내용

### 3.1 전체 시스템 구조

최종 채택 구조는 다음과 같다.

```text
[Phase 1]
D-1 baseline forecast
  ↓
[Phase 2]
Intraday remaining-day reforecast
  ↓
[Outage Override]
rule-based blackout handling
  ↓
[Gap Interpretation]
current gap + forward gap
  ↓
[LNG Planner]
baseline LNG schedule 위 추가 대응량 계산
  ↓
[Unit Allocation]
online fleet 우선 재배분
  ↓
[Residual]
offline startup screening 후보화
  ↓
[Dashboard]
운영자 시각화 / 의사결정 지원
```

이 구조의 중요한 점은 각 레이어가 서로 다른 책임을 가진다는 것이다.

- `Phase 1`은 기준 계획선 형성
- `Phase 2`는 최신 actual 반영
- `Outage Override`는 비기상성 급락 처리
- `Planner`는 gap을 thermal action으로 변환
- `Dashboard`는 사람이 이해할 수 있는 형태로 결과를 보여줌

즉 모든 기능을 하나의 모델 안에 밀어 넣지 않고, **예측 문제와 운영 문제를 분해한 뒤 다시 연결하는 구조**를 취했다.

---

## 4. 데이터 구성

### 4.1 원천데이터

본 과제에서 사용한 원천데이터는 다음과 같다.

| 구분 | 파일/폴더 | 기간 | 역할 |
|---|---|---|---|
| 태양광 실적 | `data/solar_hourly/solar_hourly_YYYYMM.csv` | 2022-01 ~ 2025-12 | 예측 타깃값 |
| 지상 기상 | `data/asos_hourly/asos_hourly_YYYYMM.csv` | 2022-01 ~ 2025-12 | ASOS 시간별 기상 |
| 위성/복사 raw | `data/gk2a_raw/.../*.nc` | 2022-01 ~ 2025-12 | GK2A 원천 복사 자료 |
| 위성 추출본 | `data/gk2a_v1/`, `data/gk2a_v2/` | 2022-01 ~ 2025-12 | 모델 입력용 위성 파생 자료 |
| 사이트 메타 | `data/solar_sites.csv` | 고정 | site mapping |
| LNG 실적 | `data/thermal_hourly/thermal_hourly_YYYYMM.csv` | 2022-01 ~ 2025-12 | 호기별 LNG 실적 원천 |
| 연료 소비 | `data/fuel_consumption/` | 2022-01 ~ 2025-12 | 연료 사용 실적 |
| 연료 조달 | `data/fuel_procurement/` | 2022-01 ~ 2025-12 | 연료 조달 실적 |
| 운영 참고자료 | `data/info/*.csv`, `*.pdf` | 수시 | 운영현황, SMP, 설비 정보 |

### 4.2 가공 데이터

원천데이터를 기반으로 다음의 핵심 가공 데이터를 생성했다.

| 파일 | 내용 | 현재 사용 여부 |
|---|---|---|
| `data/processed/training_set.parquet` | PV 실적 + ASOS + GK2A 통합 학습셋 | 사용 |
| `data/processed/lng_baseline_test.parquet` | 2025 LNG baseline proxy | 사용 |
| `data/processed/lng_unit_specs.parquet` | 호기별 Pmin/Pmax/ramp spec | 사용 |
| `data/processed/phase2_outage_labels.parquet` | outage 실험용 라벨 | 사용 |
| `data/processed/portfolio_predictions.parquet` | 과거 시나리오/EDA용 portfolio 예측 | 보조 |
| `data/processed/scenarios_*.parquet` | 불확실성/시나리오 실험 | 보조 |

### 4.3 데이터 해석 원칙

본 과제는 실제 운영 데이터 부재를 다음 두 가지 proxy 원칙으로 보완했다.

#### 4.3.1 기상예보 proxy

원래는 외부 예보 데이터(Meteologix 등)를 사용하는 방향을 고려했지만, 과거 시점의 일관된 예보 이력을 학습/백테스트용으로 정합성 있게 확보하지 못했다. 따라서 본 PoC에서는:

- ASOS 실측
- GK2A 실측/추출값

을 `forecast proxy`로 사용했다. 즉 본 과제의 D-1 기상 입력은 진짜 archived forecast가 아니라 **perfect-foresight weather proxy**다.

#### 4.3.2 D-1 thermal plan proxy

실제 D-1 thermal schedule도 제공받지 못했기 때문에, LNG 쪽에서는 2025 실측 LNG hourly를 그 시점의 기준 운전상태로 해석했다.

```text
P_DA,u(t) = 2025년 시점 t의 호기 u 실측 LNG 발전량
```

이것은 실제 dispatch 정답을 복원하려는 것이 아니라, **그날의 기준 운전상태 위에서 추가 대응량만 계산하기 위한 proxy**다.

이 두 가지 proxy는 본 과제의 한계이자 동시에 PoC의 전제다.

---

## 5. 데이터 기반 EDA와 설계 의사결정 입증

> 본 절은 단순 시각화가 아니라, **최종 spec 의 모든 임계값/가정/구조 결정** 이 데이터 분석으로 정당화된 과정을 정리한다.
> 본 절은 두 part 로 구성된다. **5A** 는 PV 예측 모델 측 EDA, **5B** 는 LNG planner 측 EDA. 각 part 의 그림·csv 는 각각 `pv/eda_pv_model/` 와 `pv/eda_lng_planner/` 에 저장되며, 본 문서는 *결과 → 설계 결정* 인과만 요약한다.

---

# Part 5A. PV 예측 모델 측 EDA — *데이터 자체에서 모델 결정 도출*

> 위치: `pv/eda_pv_model/` (M1~M5 motivating EDA 5 그림 + 기존 sweep 결과 `_selection_results/`).
>
> 본 part 의 모든 EDA 는 *모델 선정 결과* 가 아니라 **데이터 내재적 특성에서 모델 결정의 근거를 도출** 한다.

### 5A.1 사이트 패턴 이질성 — AdaLN conditioning 정당화 (`M1_site_pattern_heterogeneity.png`)

본 PoC 가 단순 ResMLP / NGBoost 가 아니라 **ResMLP + AdaLN** 을 선택한 핵심 이유는, 8 사이트의 *시간/계절 패턴이 서로 충분히 달라* 단일 함수로 fit 불가능했기 때문이다.

4년 hourly cf 데이터로 측정:

| 측정 | 결과 |
|---|---|
| 사이트 × 시각 평균 cf 패턴 (daily curve) | 사이트별로 일사 시작·peak·종료 시각이 다름 |
| 사이트 × 월 평균 cf 패턴 | 계절성 강도 / peak 월 사이트마다 다름 |
| 사이트 간 daily curve 상관 | **평균 0.983 / 최소 0.921** — 100% 동일 X |

해석:
- 광양항세방 (항만, 동향) — 일찍 peak
- 영농형 (예천) — 그림자 영향 패턴
- 큰 cap 수상 (고흥만수상, 63 MW) — 가장 일관된 종 모양
- 단일 함수로 8 사이트 fit → site-specific bias / 변동성 차이 흡수 불가능

→ **모델 함의**: AdaLN (Adaptive Layer Normalization) 으로 site / hour / month 효과 conditioning 필수. 단순 LSTM 이나 NGBoost 가 ResMLP+AdaLN 대비 약한 이유는 *site embedding* 이 약하거나 *temporal conditioning* 이 명시적이지 않기 때문.

### 5A.2 PV 변동의 시간 척도 — Phase 2 / H=6 정당화 (`M2_temporal_scale.png`)

**왜 D-1 한 번 예측이 아니라 매 시간 갱신 (Phase 2) 이 필요한가?** + **왜 H=6 look-back 인가?** 두 질문에 데이터로 답한다.

| 측정 | 결과 | 함의 |
|---|---|---|
| 시간당 \|Δcf/h\| 분포 | p95 = **0.219**, p99 = **0.283** (max 0.5+) | fat tail — 간헐적 *큰 변동* 발생 → D-1 단일 예측 한계 |
| cf residual autocorr (일사주기 제거 후) | lag 1h = **0.888**, lag 3h = **0.612**, lag 6h = **0.422** | *6시간까지 자기상관 유의* → H=6 정당 |
| cloud-pass 지속시간 (174 events) | median **1h**, p75 2h, p90 3h | *대부분 1~3h 안에 종료* → Phase 2 lead 1~3h 가 운영 가능 영역 |
| 시간대별 평균 변동성 | 12-14시 peak | 정오 ±2h 가 Phase 2 가치 집중 |

→ **모델 함의**:
- 시간당 변동이 *fat tail* → D-1 single forecast 만으로는 불충분, **intraday 갱신 (Phase 2) 필수**
- residual autocorr 가 lag 6h 까지 0.4+ 유지 → **H=6 look-back window 가 sufficient**
- cloud-pass 가 1~3h 단위라 → Phase 2 의 lead 1~3 + EOD 곡선 이 *운영 의사결정 영역과 일치*

### 5A.3 입력 변수 marginal value — ASOS + GK2A 조합 정당화 (`M3_input_marginal.png`)

**왜 위성 자료 (GK2A v2) 가 필요한가?** ASOS 만으로는 부족한가?

cf 와 각 입력 변수의 상관 (daytime 9-16, n=171,907):

| 변수 | corr with cf | 비고 |
|---|---:|---|
| **dsr_mean (GK2A 위성 일사)** | **+0.85** | 가장 강한 단조 관계 |
| zenith_center (일사각) | -0.65 | daily seasonality (AdaLN 흡수) |
| hm (습도) | -0.49 | 흐림/안개 효과 |
| ws (풍속) | +0.10 | 미미 |
| rn (강수) | -0.22 | 강수 시 cf 저하 |
| ta (기온) | +0.04 | seasonality 잔여 |

해석:
- **dsr (위성 일사)** 이 cf 와 가장 강한 단조 관계 → 위성 입력 *없으면 큰 정확도 손실*
- ASOS 지상 측정은 *site-local* 이라 광역 일사 정보 부족
- GK2A v2 의 광역 cloud / aerosol 채널 → cloud-pass 시 *공간 추세* 정보 제공
- 부분 흐림 영역 (dsr 분위 Q1-Q4) 에서 cf 분포 폭이 큼 → **Phase 2 event branch 가 잡아야 할 영역**

→ **모델 함의**: ASOS + GK2A v2 *조합* 채택의 데이터 기반 근거. dsr 만으로 평상시는 cover 되고, 부분 흐림 / cloud-pass 는 ASOS + GK2A 조합으로 보강.

### 5A.4 outage / cloud-pass 통계 분리 — rule-based override 정당화 (`M4_outage_separability.png`)

**왜 outage 처리가 *학습형* 이 아니라 *rule-based* 인가?**

test 2025 의 cf<0.03 ∧ μ_p1>0.20 인 *deep zero* 사건 분석 (n=288):

| 측정 | 결과 |
|---|---|
| 전체 daytime 중 deep zero 비율 | < 1% (희소) |
| deep zero 의 z-score 분포 | p1 ≈ -3.5, p5 ≈ -3.0 — *전체 분포의 deep tail* 에 집중 |
| z<-3 비율 (rule 임계) | 24% (rule 이 conservative — false positive 작음) |
| dsr 분위 분포 | *맑은 날 (높은 dsr) 에도 deep zero 발생* — 단순 cloud cover 로 설명 X |

→ **모델 함의**:
1. deep zero 사건이 전체 분포의 *통계적 outlier* (z<-3 deep tail) 에 집중 → P1 baseline 기준 *명확한 신호*
2. dsr 와 무관한 deep zero 발생 → *비기상성 정지* 사건임을 입증
3. **z + dsr + neighbor 의 두 축으로 outage / cloud-pass 분리 가능** → rule-based detector 자연스러운 선택
4. Learned head (Phase 2 joint training) 는 label imbalance (0.21%) + sample 437개 → false positive 폭증 → 폐기 정당

### 5A.5 Phase 1 residual 의 학습 가능성 — Phase 2 동기 (`M5_residual_pattern.png`)

**왜 Phase 2 가 *residual correction* 만 학습해도 가치가 있는가?**

5-seed Phase 1 ensemble 의 residual (= cf − μ_p1) 분석 (test 2025):

| 측정 | 결과 |
|---|---|
| residual autocorr lag 1h | **0.763** ★ — 직전 actual 이 다음 residual 강한 신호 |
| residual autocorr lag 3h | 0.481 |
| residual autocorr lag 6h | 0.278 |
| 시간대별 residual mean | ≈ 0 (Phase 1 이 systematic bias 없음) |
| 시간대별 residual std | 정오 ±2h peak (변동성 큰 시간대) |
| 사이트별 residual std | 광양항 0.13 / 예천 0.13 / 창원 0.15 가 큼 |
| cloud event 시 residual | left tail 두꺼움 — Phase 2 event branch 영역 |

→ **모델 함의**:
1. Phase 1 residual 이 *random noise X, 학습 가능한 패턴* 보유 (lag 1h corr 0.76)
2. **Phase 2 가 H=6 actual look-back 만 보고도 lead 1~3 correction 가능** — autocorr 가 그 영역에서 강함
3. 사이트별 residual std 차이 → site-specific event branch 필요 (광양항 / 예천)
4. cloud event 시 residual left tail 두꺼움 → *event branch + soft gate (λ=2.0)* 가 잡아야 할 영역

→ Phase 2 의 *2-branch + soft gate* 채택의 데이터 내재적 근거. Phase 1 anchor frozen + residual correction 만 학습하는 구조가 *데이터로부터 도출* 됨.

### 5A.6 Outage 처리 — 3단계 negative result + rule-based 채택

본 단락은 §7 (Outage Override Layer) 에서 *상세 정량 결과* 와 함께 다룬다 (요약):

1. ❌ **Phase 1 outage-masked retrain** — NMAE +0.14~0.16pp 악화 → 폐기
2. ❌ **Integrated outage probability head (joint loss)** — FP 폭증, cloud-pass 훼손 → 폐기
3. ✅ **Rule-based outage override** — Normal 손상 0pp, Blackout 55.7→0.39%, cloud-pass −10~20pp → 채택

→ 자세한 정량 결과는 §7.5 표 참조.

### 5A.7 모델 선정 *결과* — 부록 자료 (`_selection_results/`)

5A.1~5A.5 의 motivating EDA 로 *모델 구조 결정* 후, 실제 sweep 결과는 부록으로 분리하여 보관:

- `_selection_results/01_site_selection.png` — 사이트별 NMAE / Cov / capacity
- `_selection_results/02_model_comparison.png` — 5종 모델 NMAE 비교 (LSTM 7.26%, ResMLP+AdaLN ensemble 5.88%)
- `_selection_results/03_phase2_architecture.png` — H/λ/gate/L sweep 결과
- `_selection_results/04_calibration_seed.png` — Cov80 85% / 5-seed std 0.099pp

→ 본 결과들은 *5A.1~5A.5 의 motivating EDA 결과* 와 *일치하는 방향* 으로 도출되었다. 즉:
- M1 (사이트 이질성) → AdaLN ensemble 가 LSTM 대비 우세
- M2 (시간 척도) → H=6 sweep 결과가 H=3/4/5 와 비교해 top-event 우수
- M3 (위성 압도) → ASOS-only baseline 대비 GK2A 추가가 NMAE 개선
- M4 (outage 분리) → rule-based detector 의 0% FP 달성
- M5 (residual 패턴) → Phase 2 residual correction 의 lead 1h NMAE −24%

본 PoC 의 PV 포트폴리오 = 8 사이트, 합계 77.28 MW. 사이트별 NMAE / bias / Cov / capacity 분포 (Phase 1 5-seed ensemble, test 2025):

| 사이트 | cap (MW) | NMAE (%) | bias (%) | Cov80 | Cov95 | 평가 |
|---|---:|---:|---:|---:|---:|---|
| 경상대 | 0.91 | **3.38** | +0.37 | 92.0 | 97.5 | 가장 모델링 잘됨 |
| 영흥 | 5.49 | 4.36 | +0.45 | 88.0 | 95.1 | 적정 |
| 고흥만수상 | **63.48** | 4.41 | +0.12 | 89.4 | 96.7 | 큰 cap, 적정 |
| 삼천포 | 1.34 | 4.68 | -0.23 | 88.2 | 95.3 | 적정 |
| 구미 | 0.99 | 5.17 | +2.02 | 83.4 | 94.3 | 약한 outage 영향 |
| 예천 | 2.00 | **8.12** | -0.67 | 78.0 | 90.8 | anomaly_zero 4.95% |
| 광양항세방 | 2.99 | **8.33** | **+7.02** | 69.9 | 84.7 | bias 큼, microclimate |
| 창원 | 0.08 | 8.55 | +2.22 | 90.9 | 98.4 | cap 0.08 MW (잡음) |

**Capacity-weighted portfolio 효과**:
- 단순 site mean NMAE: 5.87%
- cap-weighted site NMAE: 4.66%
- **portfolio NMAE: 3.75%**

→ **모델 함의**: 고흥만수상 (63 MW, 82% 비중) 이 portfolio dominant. 사이트 outlier (광양항 8.3% bias +7%, 예천 8.1%) 가 *capacity 기준 portfolio 에서는 영향 작음*. 이에 따라:
- 8 사이트 모두 유지하되 처치 차별화: 예천 anomaly_zero 마스킹 X (운영 모듈은 actual=0 신호도 학습), 광양항 bias 후보, 창원 잡음 검증 용도

### 5A.2 Backbone 모델 비교 — 5종 head-to-head (`02_model_comparison.png`)

`pv/experiments/` 의 60+ 실험 폴더 중 *동일 평가 protocol (test 2025, 32,443 rows)* 로 비교 가능한 5종 모델 NMAE:

| 모델 | NMAE | bias | 비고 |
|---|---:|---:|---|
| LSTM baseline | 7.26% | +0.01 | 시계열 RNN |
| NGBoost baseline | 6.33% | -0.25 | tree + uncertainty |
| ResMLP+AdaLN (single) | 6.13% | -0.11 | tabular + AdaLN |
| FT-Transformer | 6.11% | +0.62 | tabular transformer (무거움) |
| **ResMLP+AdaLN v2 ensemble (★ 채택)** | **5.88%** | +1.42 | 5-seed |

(주: 본 표는 *32,443 rows 공통 평가* 기준. model_final §1.2 의 4.66% 는 *clean 평가 (예천 anomaly_zero 처리)* 기준)

**채택 이유** (5-seed ensemble):
1. 5종 중 NMAE 최우수 (LSTM 대비 −1.38pp)
2. **ensemble 구조가 sigma_total 을 자연스럽게 제공** → Phase 2 의 frozen anchor
3. AdaLN conditioning 으로 site/hour/month 효과 자연 흡수
4. tabular 입력 (ASOS + GK2A + static) 에 적합
5. **Phase 2 residual correction 을 붙이기 구조적으로 적합** — 가장 결정적

→ **모델 함의**: 단순 forecast 정확도가 아니라 *다음 단계가 의존할 수 있는 안정적 anchor* 라는 점이 채택 핵심.

### 5A.3 Phase 2 architecture sweep — 모든 hyperparameter 결정 (`03_phase2_architecture.png`)

Phase 2 (intraday reforecast) 의 모든 설계 결정 (H, λ, gate, L) 이 sweep 결과로 정당화됨.

**(A) H (look-back window) sweep — H=6 채택**

| H | Overall | Lead 1h | 03-23 cloud-pass | 04-26 | per-seed std |
|---|---:|---:|---:|---:|---:|
| 3 | 4.717 | 4.239 | 45.31 | 34.75 | 0.020 |
| 4 | 4.715 | 4.237 | 44.39 | 34.51 | 0.004 |
| 5 | 4.715 | 4.242 | 42.93 | 34.41 | 0.013 |
| **6 ★** | 4.729 | 4.256 | **41.98** | **34.31** | 0.049 |

→ Overall 차이 0.01~0.02pp (사실상 동률) 이지만 **top cloud-pass event 에서 H=6 가장 좋음**. PoC = top-event robustness 우선 → H=6.

**(B) 2-branch λ sweep — λ=2.0 채택**

| 비교 | NMAE | Lead 1h | per-seed std |
|---|---:|---:|---:|
| H=6 single-branch | 4.729 | 4.256 | 0.049 |
| 2-branch λ=1.0 | 4.738 | 4.271 | 0.023 |
| 2-branch λ=1.5 | 4.742 | 4.261 | 0.014 |
| **2-branch λ=2.0 ★** | **4.712** | **4.209** | **0.026** |

→ λ=2.0 이 NMAE 우세 + per-seed std −47%. gate top-event 활성화 1.9x (max 0.95).

**(C) Sparse gate sweep — 폐기 (negative result)**

| config | gate ratio | dead seeds | NMAE |
|---|---:|---:|---:|
| **T=1.0 τ=0 (★ 최종)** | 1.9x | 0/3 | 4.712 |
| T=0.5 τ=0 | 5.8x | 0/3 | 4.743 |
| T=0.5 τ=0.1 | 16.8x | **2/3** | 4.732 |
| T=0.5 τ=0.2 | 0x | **3/3** | 4.729 |

→ Selectivity↑ 시 vanishing gradient 로 dead seed 폭증 → 운영 PoC 안정성 부적합 → 폐기.

**(D) L (output horizon) — L=12 EOD truncation 채택**

- `L=3`: lead-1 즉시 보정 강하지만 *남은 하루 shape* X
- `L=6`: L=12 에 dominate
- **`L=12 EOD truncation`**: lead-1~3 정확도 대부분 보존 + recovery / persistence / sunset shape 추가

→ Phase 2 가 *short patch* 에서 **intraday remaining-day reforecast** 로 해석 전환.

### 5A.4 Calibration & 5-seed ensemble 효과 (`04_calibration_seed.png`)

5-seed Phase 1 per-seed (test 2025):

| seed | NMAE | bias | Cov80 | Cov95 | NLL |
|---|---:|---:|---:|---:|---:|
| 42 | 4.77 | -0.21 | 83.1 | 93.2 | -1.187 |
| 7 | 4.89 | +0.28 | 86.9 | 95.3 | -1.162 |
| 123 | 4.68 | +0.64 | 79.3 | 90.7 | -1.139 |
| 202 | 4.86 | +1.53 | 81.8 | 92.9 | -1.153 |
| 999 | 4.93 | -0.17 | 78.5 | 90.4 | -1.125 |
| **per-seed 평균** | 4.83 | | 81.9 | 92.5 | |
| **ensemble (5-seed)** | **4.66** | | **85.0** | **94.1** | **-1.218** |

→ **모델 함의**:
1. 5-seed ensemble: per-seed std 0.099pp → ensemble 0pp (안정화)
2. Cov80 85.0% (목표 80% ±5pp), Cov95 94.1% (목표 95%) — **σ well-calibrated**
3. Phase 2 가 *σ 그대로 사용* (Phase 1 그대로) 정당화 — ensemble σ 가 이미 잘 보정됨
4. 사이트별 outlier: 광양항 Cov80 70% (underconfident) — bias correction 후보

### 5A.5 Outage 처리 — 3단계 negative result + rule-based 채택

본 단락은 §7 (Outage Override Layer) 에서 *상세 정량 결과* 와 함께 다룬다 (요약):

1. ❌ **Phase 1 outage-masked retrain** — NMAE +0.14~0.16pp 악화 → 폐기
2. ❌ **Integrated outage probability head (joint loss)** — FP 폭증, cloud-pass 훼손 → 폐기
3. ✅ **Rule-based outage override** — Normal 손상 0pp, Blackout 55.7→0.39%, cloud-pass −10~20pp → 채택

→ 자세한 정량 결과는 §7.5 표 참조.

---

# Part 5B. LNG planner 측 EDA

> 위치: `pv/eda_lng_planner/` (8 그림 + 8 csv).

### 5B.1 초기 가정

프로젝트 초기에는 태양광 급락의 상당 부분을 `cloud-pass` 또는 `partial cloud` 이벤트로 해석했다. 따라서 Phase 2의 역할도 "짧은 시간 안에 발생한 구름 이벤트를 residual correction으로 더 잘 흡수하는 것"으로 생각했다.

### 5B.2 EDA D — 사이트별 outage 패턴 (proposal §7 outage override 정당화)

후속 EDA에서 일부 대표 이벤트가 단순 weather-driven miss가 아니라, **맑은 날씨에도 발생하는 near-zero 급락**을 포함하고 있음을 확인했다.

`pv/eda_lng_planner/D_site_outage_heatmap.png` (사이트 × day-of-year × hour 외양 히트맵):

| 사이트 | cap (MW) | outage 시간 | 비고 |
|---|---:|---:|---|
| 광양항세방 | 3.0 | 279h (8.5%) | 가을 (10월) cluster |
| 예천 | 2.0 | 362h (11.0%) | 단속 운영 |
| 구미 | 1.0 | 197h (6.0%) | 여름 outage |
| 영흥 | 5.5 | 160h (4.9%) | 점진적 |
| **고흥만수상** | **63.5** | 74h (2.3%) | 점심시간 반복, 큰 cap → portfolio 영향 큼 |
| 삼천포 | 1.3 | 137h (4.2%) | |
| 경상대 | 0.9 | 124h (3.8%) | |
| 창원 | 0.08 | 71h (2.2%) | |

처음에 top cloud-pass 로 보였던 사례 중 일부는 실제로는 weather event 가 아니라 **non-meteorological operating event** 에 가까웠다.

→ **모델 함의**: outage 가 *site-level non-meteorological* 사건임이 데이터로 확인 → Phase 2 학습으로 일관 흡수 어려움 → rule-based override 가 적합.

### 5B.3 이 발견이 가져온 구조 변경

기존 가정:
- 모든 급락은 Phase 2가 학습으로 흡수해야 한다

수정된 가정:
- weather-driven 변동은 Phase 2가 담당
- weather로 설명되지 않는 outage-like 급락은 별도 검출 및 override가 담당

이 구조 분리가 없으면 모델 평가도 왜곡된다. 실제로 일부 날짜는 "Phase 2가 cloud-pass를 잘 잡았다"기보다 "outage contamination까지 같이 흡수해 좋아 보였던 것"으로 재해석되었다.

### 5B.4 EDA C — Forward signal 임계값 정당화 (planner §7.3 임계값)

planner 의 forward signal 분류 임계값 (`TH_NOISE=3, TH_INCREASE=5, TH_SLOPE=2`) 은 *임의값* 이 아니라 **데이터 분위수 기반** 임을 입증.

`pv/eda_lng_planner/C_forward_gap_distribution.png` (n=3,302, lead=1, daytime 8-17):

```
|FG1| 분위:   p50=1.31  p75=2.54  p80=2.94  p90=4.23  p95=5.98  p99=15.61
|slope| 분위: p50=0.65  p75=1.15  p90=1.80  p95=2.40  p99=15.73
```

→ TH_INCREASE=5 ≈ p95, TH_NOISE=3 ≈ p80, TH_SLOPE=2 ≈ |slope| p95.
→ **모든 임계값이 데이터 기반 round 값**. 운영 직관 ("미세 변동 무시, 상위 5% 만 큰 신호") 과 일치.

### 5B.5 EDA E — 시간대별 운영 부담 (운영자 모니터링 가이드)

`pv/eda_lng_planner/E_hourly_burden.png`:

| 시각 | 실시간 차이>0 합 | LNG 대응 차이 | 예열 명령 | 추가 대응 모드 |
|---|---:|---:|---:|---:|
| 9 | 317 | 36 | 0 | 2 |
| 10 | 499 | 77 | 3 | 11 |
| 11 | 638 | 180 | **6 ★** | 21 |
| 12 | 889 | 381 | 2 | 25 |
| 13 | 1,034 | **585 ★** | 4 | 27 |
| 14 | 1,081 | 561 | 3 | 20 |
| 15 | 763 | 250 | 1 | 6 |
| 16 | 520 | 127 | 1 | 3 |
| 17 | 357 | 52 | 0 | 0 |

→ LNG 대응 차이 peak = **13시** (정오 ±1h 가장 큰 부담).
→ 예열 명령 peak = **11시** (변동 1~2h 전 *선제* 예열, 운영 직관 부합).

### 5B.6 EDA H — 호기 운전 패턴 (peaker 가정 폐기 근거)

LNG planner 초기 spec 에서는 "peaker = 단발 spike 만 잡는다" 가정을 가졌으나, 4년 hourly 데이터 분석 결과 **peaker 도 평균 5~8h 연속 운전** 을 함을 발견 → mode 가정 폐기.

`pv/eda_lng_planner/H_unit_pattern.png`:

| 호기 | mode | 운전율 | 평균 연속 운전 | 최대 연속 |
|---|---|---:|---:|---:|
| CS2 | baseload | 51.9% | 33.5h | 1,399h |
| CG6 | baseload | 47.6% | 32.1h | 1,455h |
| CG2 | **peaker** | 23.0% | **19.3h** | 539h |
| CG4 | **peaker** | 19.8% | **19.7h** | 991h |

→ peaker (CG2/CG4) 도 *한 번 켜지면 평균 6~7h 연속*. mode 차이는 *운전 빈도* 차이일 뿐 *연속 운전 길이* 차이 X.
→ **호기 선택 룰: mode 우선순위 X, 헤드룸·ramp 기반 정렬이 더 정직**.

---

### 5B.7 EDA B — 호기 ramp 비대칭 (over-commit 원인 입증)

잔존 over-commit 640 MWh 의 메인 원인이 *호기 자체의 느린 하강 한도* 임을 데이터로 입증.

`pv/eda_lng_planner/B_ramp_asymmetry.png` (4년 hourly, 운전 중 transition):

| 호기 | mode | 상승 p90 | 하강 p90 | 비대칭 비율 |
|---|---|---:|---:|---:|
| CG6 | baseload | 18.2 | **1.87** | **9.7x** |
| CG8 | mid-merit | 19.0 | 2.39 | 8.0x |
| CG7 | mid-merit | 25.7 | 3.19 | 8.1x |
| CS2 | baseload | 18.5 | 2.74 | 6.8x |
| CS1 | mid-merit | 29.3 | 7.42 | 4.0x |
| CG2 | peaker | 28.9 | 11.18 | 2.6x |

→ baseload 호기는 *상승은 빠르고 하강은 매우 느림* (실측 패턴, 모델 인위 X).
→ 한 번 올라간 출력이 회복 후에도 *천천히* 만 내려가서 잔류 ΔP = over-commit.
→ **잔존 over-commit 640 MWh 는 운영 데이터 자체의 비대칭이 만든 결과**.

### 5B.8 EDA A — DEADBAND 민감도 (자체 흡수폭 4 MW 정당화)

DEADBAND=4 가 우연이 아니라 *합리적 plateau 영역* 임을 sweep 으로 입증. planner 를 DEADBAND ∈ {0, 2, 4, 6, 8} 로 재실행한 결과:

`pv/eda_lng_planner/A_deadband_curve.png`:

| DEADBAND | 부족분 | 과대보충 | 비율 | 총 추가 발전 |
|---:|---:|---:|---:|---:|
| 0 | 1,197 | 770 | 0.64 | 5,670 |
| 2 | 699 | 700 | 1.00 | 3,485 |
| **4 ★** | **441** | **640** | **1.45** | **2,448** |
| 6 | 303 | 583 | 1.92 | 1,897 |
| 8 | 224 | 535 | 2.39 | 1,602 |

→ DEADBAND=0 (LNG 전량 책임): over-commit 폭증, 비현실적.
→ DEADBAND=4 (LNG ≈50% 담당, 보조서비스 정산금 근거): 부족·과대 균형.
→ DEADBAND=8: LNG 책임 너무 작아짐 → PoC 의미 약화.

→ **DEADBAND=4 는 보조서비스 정산 (LNG ≈50%) 근거 + KPI 균형이 일치하는 영역**.

### 5B.9 EDA G — Phase 2 가치의 일별 분포

평균 -19.1% 라는 수치가 *분산 큰 평균* 임을 보임.

`pv/eda_lng_planner/G_phase2_marginal.png`:

| 구간 | 일수 | 비고 |
|---|---:|---|
| 안정 평일 (eff_gap < 1 MWh) | 213 일 (58%) | Phase 2 효과 0 |
| 작은 변동 (1 ≤ eff_gap < 5) | ~140일 | 미미한 효과 |
| 큰 변동일 (Phase 2 효과 > 5 MWh) | 3 일 | 09-06, 09-07, 03-22 등 |

→ Phase 2 의 *평균* 효과는 -19% 지만, 가치는 **큰 변동일 (Top 10 일) 에 집중**.
→ Top 10 일 대부분에서 예열 명령 동시 발동 → 운영자 사전 대비 가능 (PoC 핵심 메시지).

### 5B.10 EDA F — startup 분리 (cold-start 비용 회피 입증)

기존 단일 startup_count 가 *부정확* 함을 데이터로 보이고 분리.

`pv/eda_lng_planner/F_startup_split.png`:

| 구분 | Phase 1 단독 | Phase 1+2 |
|---|---:|---:|
| **신규 가동** (offline → 발전, cold-start 비용) | **1 회** | **0 회** |
| **추가 발전 전환** (이미 운전 중 + ΔP) | 931 회 | 941 회 |

→ 진짜 신규 가동 1년 0~1회 — cold-start 비용 (~5천만원/회) 거의 회피.
→ 941회는 *이미 운전 중 호기의 추가 발전 시작* — cost-aware screening 부재 상태에서도 *자연스럽게 비용 효율적 운영*.

### 5B.11 EDA 기반 설계 결정 요약 (전체)

**PV 예측 모델 측 (5A) — motivating EDA**:

| EDA | 그림 | 입증한 설계 결정 |
|---|---|---|
| 5A.1 | `M1_site_pattern_heterogeneity.png` | 사이트 패턴 이질성 → **AdaLN site/hour conditioning 필수** |
| 5A.2 | `M2_temporal_scale.png` | residual autocorr lag 6h=0.42 + cloud-pass 1~3h → **Phase 2 H=6 + lead 1~3** |
| 5A.3 | `M3_input_marginal.png` | dsr (위성) corr +0.85 압도적 → **GK2A 위성 입력 필수** |
| 5A.4 | `M4_outage_separability.png` | deep zero 가 z 분포 deep tail 집중 → **rule-based outage override** |
| 5A.5 | `M5_residual_pattern.png` | Phase 1 residual lag 1h corr 0.76 → **Phase 2 residual correction 가능** |

**LNG planner 측 (5B)**:

| EDA | 그림 | 입증한 설계 결정 |
|---|---|---|
| 5B.2 | `D_site_outage_heatmap.png` | weather event vs outage 분리 → rule-based override (§7) |
| 5B.4 | `C_forward_gap_distribution.png` | forward signal 임계값 (TH_INCREASE=5, TH_SLOPE=2) — p95 기반 |
| 5B.5 | `E_hourly_burden.png` | 시간대별 운영 부담 = 정오 ±2h, 예열 1~2h 선제 |
| 5B.6 | `H_unit_pattern.png` | peaker 가정 폐기 → 호기 선택 룰 mode 우선순위 X (§8.4) |
| 5B.7 | `B_ramp_asymmetry.png` | over-commit 640 MWh 원인 = ramp_dn 비대칭 (모델 인위 X) |
| 5B.8 | `A_deadband_curve.png` | DEADBAND=4 MW 정당화 (sweep 결과 + 보조서비스 정산 근거) (§8.3) |
| 5B.9 | `G_phase2_marginal.png` | Phase 2 가치는 평균이 아니라 *큰 변동일 집중* — 운영 의미 |
| 5B.10 | `F_startup_split.png` | cold-start 비용 회피 — 운영비 효율성 |

---

## 6. 예측모델 설계

> 본 절의 *모든 모델 측면 의사결정* 의 EDA 정리는 `pv/eda_pv_model/_summary.md` 에 한 곳으로 모아져 있다. 각 결정마다 *60+ 실험 폴더* 와 *14개 EDA 노트북* 의 어느 결과로 정당화되었는지 추적 가능하다.
>
> 본 절은 결정의 *결과* 만 요약한다.

### 6.0 모델 측면 의사결정 한눈

| 결정 | EDA 근거 | 결과 |
|---|---|---|
| 8 사이트 선정 + 처치 차별화 | `01_site_selection.py` | portfolio NMAE 3.75% (cap-weighted, 고흥만수상 dominant) |
| Backbone = ResMLP+AdaLN v2 ensemble (5-seed) | `02_model_comparison.py` (5종 비교) | LSTM 7.26% > NGBoost 6.33% > FT-T 6.11% > **ResMLP+AdaLN ensemble 5.88%** |
| Phase 2 H=6 (look-back) | notebook 12, `03_phase2_architecture.py` | top-event robustness 우선 (03-23 41.98 vs H=5 42.93) |
| Phase 2 2-branch λ=2.0 | `03_phase2_architecture.py` | NMAE 4.712, per-seed std −47% |
| Phase 2 L=12 EOD truncation | model_final §1.3 | lead-1 정확도 보존 + remaining-day shape |
| Sparse gate (T<1, τ>0) 폐기 | `03_phase2_architecture.py` | dead seeds 폭증, negative result |
| Calibration 검증 | `04_calibration_seed_ensemble.py` | Cov80 85.0% / Cov95 94.1% (목표 ±5pp) |

### 6.1 Phase 1: Day-Ahead Baseline Forecast

Phase 1은 D-1 기준의 기준선 예측이다.

- 모델: `ResMLP + AdaLN v2 ensemble`
- 출력: site별 24시간 `mu_phase1`, `sigma_total`
- seed: 5-seed ensemble
- 역할: KPI용 기준 forecast, Phase 2의 frozen anchor

이 모델을 baseline으로 채택한 이유는 다음과 같다.

1. test 2025 기준 NMAE와 calibration이 안정적이었다.
2. ensemble 구조가 uncertainty(`sigma_total`)를 함께 제공했다.
3. Phase 2가 residual correction을 붙이기에 구조적으로 적합했다.

즉 Phase 1은 “최종 운영 예측”이라기보다, **다음 단계가 의존할 수 있는 안정적 anchor**라는 의미가 더 크다.

### 6.2 Phase 2: Intraday Remaining-Day Reforecast

Phase 2는 frozen Phase 1 위에서 최신 actual과 observed weather를 반영해 남은 하루를 다시 예측한다.

- 모델: `2-branch TCN residual correction`
- 구조:
  - `base branch`
  - `event branch`
  - `soft gate`
- 입력:
  - 최근 H=6시간 actual PV
  - 최근 H=6시간 observed weather
  - frozen Phase 1 forecast
  - static condition (site, hour, month)
  - event-aware gate signals

핵심 변화는 horizon 해석이었다.

초기에는 “next 1~3h patch”처럼 보았지만, 최종 채택 구조는:

> **remaining-day reforecast with EOD truncation**

이다. 즉 issue time마다 1~3시간만 고치는 것이 아니라, 그 시점부터 남은 하루 전체 곡선을 다시 그리는 구조다.

### 6.3 왜 이 구조가 중요했는가

이 구조는 planner와 직접 연결된다.

- `lead 1h` 오차를 줄이는 것은 즉시 대응에 중요
- `남은 하루 recovery shape`를 보여주는 것은 준비/해제 판단에 중요

즉 Phase 2는 더 이상 단순 patch model이 아니라, **운영적으로 읽을 수 있는 남은 하루 곡선 생성기**로 해석된다.

### 6.4 최종 성능

최종 production 기준 성능은 다음과 같다.

| 지표 | Phase 1 | Phase 2 only | Phase 2 + Override |
|---|---:|---:|---:|
| 전체 NMAE | 4.747% | 4.432% | **4.220%** |
| Normal 시점 | 4.50% | 4.23% | **4.23%** |
| Blackout 시점 | 68.6% | 55.7% | **0.39%** |
| 2025-03-23 | 30.0% | 23.3% | **2.89%** |
| 2025-04-26 | 15.0% | 13.7% | **3.81%** |
| 2025-05-04 | 18.7% | 16.1% | **3.79%** |
| 광양항 10/10-12 | 28.2% | 15.5% | **0.09%** |

이 표가 의미하는 것은 단순하다.

1. Phase 2는 weather-driven correction에서 baseline보다 낫다.
2. 그러나 outage-like event까지 완전히 설명하진 못한다.
3. outage override가 붙으면 정상 구간 손상 없이 blackout 구간을 크게 개선한다.

즉 최종 구조는 `Phase 2 only`가 아니라 **`Phase 2 + Override`** 다.

---

## 7. Outage Override Layer

### 7.1 왜 별도 레이어가 필요했는가

실험 결과, outage-like 이벤트는 weather feature만으로 잘 설명되지 않았다. 실제로 integrated outage head를 붙여 모델 내부에서 같이 학습시키는 실험도 했지만, false positive가 과도하게 늘어나 정상 시점까지 망가뜨렸다.

즉 blackout 처리는:

- 예측모델이 학습으로 해결할 문제

가 아니라,

- 운영 단계에서 명시적으로 rule-based 처리할 문제

로 보는 것이 더 맞았다.

### 7.2 최종 detector

최종 채택 rule은 다음과 같다.

```text
outage_flag = (
    cf < 0.03
    and mu_phase1 > 0.20
    and dc10Tca < 7
    and z < -3.0
    and neighbor_confirmed
)
```

이 규칙은 다음 의미를 동시에 만족한다.

1. 실제 발전은 거의 0이다.
2. baseline은 의미 있는 발전을 기대하고 있다.
3. 강한 흐림은 아니므로 단순 cloud-pass로 보기 어렵다.
4. Phase 1 uncertainty 기준에서도 통계적 outlier다.
5. 단발 노이즈가 아닌 연속된 사건이다.

### 7.3 override 방식

실시간 운영 해석은 state machine으로 구현했다.

- 상태: `NORMAL` / `BLACKOUT`
- blackout 동안: `mu_phase2 = 0`
- recovery: `actual_cf > 0.10`이 1시간 지속되면 해제

즉 blackout은 soft penalty가 아니라 **명시적 hard override**다.

### 7.4 채택 근거

이 구조는 다음 조건을 동시에 만족했다.

1. 전체 NMAE 개선
2. cloud-pass 대표 날짜 개선
3. 광양항 연속 shutdown 완벽 처리
4. normal 시점 손상 0

이 네 가지를 동시에 만족한 대안은 rule-based override뿐이었다.

### 7.5 채택하지 않은 대안 (정량 결과 포함)

#### A. Phase 1 outage-masked retrain

- 학습 데이터에서 outage-like rows를 제외하고 baseline 재학습 (2 변형: narrow 278 mask / wide 539 mask)
- 정량 결과:

| variant | NMAE | Phase 1 baseline 대비 |
|---|---:|---:|
| baseline (제외 X) | 4.747% | — |
| narrow (278 mask) | 4.89% | **+0.14 pp 악화** |
| wide (539 mask) | 4.91% | **+0.16 pp 악화** |

- 원인: 0.2~0.4% 마스킹으로 모델 분포 변화 미미. 모델은 outage 를 noise 로 robust 하게 학습 중.
- 결론: 학습 분포만 흐리고 실익 없음 → 폐기.

#### B. Integrated outage probability head

- Phase 2 architecture 에 outage probability head 추가, joint loss 학습
- 정량 결과:
  - positive class 희소성 (전체의 0.2%) → class imbalance
  - 정상 시점 FP 폭증 → cloud-pass 훼손
  - blackout 검출률은 향상되었으나 정상 NMAE 악화
- 결론: learned detector 보다 rule-based detector 가 더 적합 → 폐기.

#### C. Rule-based outage override (★ 채택)

- 정량 결과 (§6.4 표):
  - 전체 NMAE 4.432% → 4.220% (개선)
  - blackout 시점 55.7% → 0.39% (극적 개선)
  - normal 시점 4.23% → 4.23% (손상 없음) ✓
  - 광양항 10/10-12 cluster: 15.5% → 0.09%
- 정상 손상 0 + blackout 극적 개선 + 대표 cloud-pass 개선 *동시 만족* → 채택.

따라서 문서상 최종 입장은 명확하다.

> **weather-driven correction은 모델이 담당하고, non-meteorological outage는 rule-based layer가 담당한다.**

---

## 8. LNG 백업 계획 및 호기별 재배분

### 8.1 LNG planner의 위치

LNG planner는 본 과제에서 매우 중요하지만, 그 역할을 과장하면 안 된다. 이는:

- full unit commitment solver가 아니고
- 전국 계통 경제급전기가 아니며
- 실제 dispatch 재현기도 아니다

대신 다음을 수행한다.

1. PV 기준선과 실 발전 사이 gap을 해석
2. 그 gap을 LNG 추가 대응량으로 변환
3. 그 대응량을 baseline LNG 운전상태 위에 호기별로 재배분

즉 planner는 **예측 결과를 thermal action으로 번역하는 운영 레이어**다.

다만 최신 LNG plan 기준에서 이 planner는 단순 “부족하면 더 올리는 규칙”이 아니라, 명확한 운영 철학을 가진 구조로 정리되어 있다. 그 철학은 다음과 같다.

1. LNG가 책임지는 것은 전체 부족분이 아니라 `계통 자체 흡수분을 제외한 marginal deficit`이다.
2. 현재 시각의 실 발전 결정과 다음 시각 대비 행동을 분리한다.
3. peak event에서 shortfall을 줄이는 것보다 `불필요한 over-commit을 줄이는 것`을 더 중요한 KPI로 둔다.

즉 이 planner의 최종 목적은 “가능한 많이 메우기”가 아니라:

> **LNG가 실제로 책임져야 하는 부족분만 정교하게 메우고, 작은 변동까지 과잉 반응하지 않도록 만드는 것**

이다.

### 8.2 baseline thermal schedule의 해석

실제 D-1 thermal plan은 없으므로, LNG planner는 2025 실측 LNG hourly를 baseline proxy로 사용한다.

```text
P_DA,u(t) = 2025년 시점 t의 호기 u 실측 LNG 발전량
```

중요한 점은 이걸 “정답 dispatch”로 보는 것이 아니라, **그날의 기준 운전 상태**로 보는 것이다.

planner가 계산하는 것은:

```text
P_new,u(t) = P_DA,u(t) + ΔP_u(t)
```

이며, baseline 자체를 새로 짜지는 않는다.

### 8.3 gap 해석

planner가 읽는 핵심 신호는 두 개다.

#### (1) Instant Gap / Effective Gap

최신 LNG plan에서 가장 중요한 변화는 `look-back realized gap`보다 **현재 시각의 instant gap**을 직접 제어 입력으로 쓰는 쪽으로 정리되었다는 점이다.

```text
instant_gap_t   = PV_baseline_t - PV_actual_t
effective_gap_t = max(0, instant_gap_t - DEADBAND_MW)
```

여기서:

- `instant_gap_t`는 현재 시각의 실제 부족분
- `DEADBAND_MW = 4.0`은 계통 자체 흡수 cushion
- `effective_gap_t`는 LNG가 실제로 책임지는 marginal deficit

즉 planner는 전체 부족을 그대로 LNG에 넘기지 않는다. 작은 편차는 계통 예비력, AGC, 기타 조정자원이 자체적으로 흡수할 수 있다고 보고, LNG는 그 초과분부터 대응한다.

**DEADBAND=4 MW 의 근거** (두 축 동시 정당화, EDA A 참조):

1. **외부 데이터 근거**: 한국 전력거래소 보조서비스 정산금 (`(최종) 연료원별 보조서비스정산금_202603.xlsx`) 분석 결과 LNG 가 전국 보조서비스 (운영예비력) 정산의 ≈50% 담당. 즉 PV 변동의 절반 정도만 LNG 가 책임진다는 운영 현실.
2. **내부 sweep 결과 (EDA A)**: DEADBAND ∈ {0, 2, 4, 6, 8} 로 planner 재실행 시 부족·과대보충 비율이 4 MW 에서 **1.45x 균형점**. 0 (계통 흡수 X) 에서는 over-commit 폭증, 8 이상에서는 LNG 책임이 너무 작아 PoC 의미 약화.

→ 외부 정산 근거 + 내부 sweep 결과가 *4 MW 영역에서 일치* → **데이터 기반 자체 흡수폭** 으로 채택.

이 DEADBAND 가정의 운영적 의미는 명확하다.

1. 3MW 정도의 작은 변동은 LNG가 직접 따라가지 않는다.
2. 10MW 부족이 발생해도 LNG는 6MW만 책임지는 구조가 된다.
3. 과거처럼 작은 회복 구간에서도 LNG가 계속 따라가며 over-commit을 만들 가능성이 줄어든다.

이 설계는 planner의 성격을 바꾸었다. 초기 구조는 shortfall을 줄이는 데 집중되어 있었다면, 최신 구조는 **계통이 자체 흡수할 수 있는 영역과 LNG가 직접 대응할 영역을 구분하는 구조**로 바뀌었다.

#### (2) Forward Gap

forward gap은 Phase 2 remaining-day reforecast가 제공하는 미래 부족 신호다. 하지만 최신 plan에서는 forward gap의 역할을 축소해서 쓰지 않는다. 오히려 역할을 더 명확히 분리한다.

forward gap은:

- 현재 시각 출력을 직접 결정하는 주입량이 아니라
- **다음 시각 준비 여부를 판단하는 신호**

다. 즉 앞으로 부족이 커질 것 같으면 지금 출력을 더 억지로 올리는 것이 아니라, **다음 시각 사용할 수 있는 호기를 미리 준비시키는 데** 사용한다.

즉:
- `effective_gap`은 **현재 action**
- `forward_gap`은 **다음 시각 preparedness**

로 역할이 다르다.

### 8.4 호기별 배분 원리

현재 v4 allocator의 핵심은 **Layer A / Layer B 분리**다.

#### Layer A: 실 발전 명령

Layer A는 현재 시각의 실제 부족분을 메우는 레이어다.

입력:

```text
즉시_보정필요량 = effective_gap_t
```

행동:

1. 현재 online 호기들의 가용 헤드룸을 계산한다.
2. `Headroom × Ramp` 기반으로 분배한다.
3. online fleet만으로 부족하면 새 호기 기동을 검토한다.
4. 이때는 **GT 우선 startup**을 적용한다.

핵심은 단순하다.

> **Layer A는 지금 시각의 effective gap을 100% 즉시 추종하는 실 발전 레이어다.**

이 구조에서 중요한 점은, Layer A가 더 이상 과거 look-back 평균 부족분에 끌려가지 않고, `지금 시각의 부족분`을 직접 기준으로 쓴다는 것이다. 이 변화는 over-commit 감소에 크게 기여했다.

#### Layer B: 사전 대비 warm-up 명령

Layer B는 forward signal을 읽지만, **현재 시각 출력에는 직접 영향하지 않는다.**

Layer B의 역할은 다음과 같다.

1. 다음 시각 online fleet headroom이 부족할 것 같은지 판단
2. 부족 예상 시 일부 호기에 대해 warm-up 명령 발행
3. 해당 호기를 다음 시각 분배 후보로 합류시킴

즉 Layer B는 output layer가 아니라 **availability preparation layer**다.

이게 최신 LNG plan의 핵심 설계다.

- Layer A = 지금 출력 결정
- Layer B = 다음 시각 대비 준비

초기 planner의 핵심 문제는 forward signal이 현재 시각 출력까지 밀어올리면서 불필요한 over-commit을 만들었다는 점이다. Layer B를 warm-up only로 분리한 뒤 이 문제가 크게 줄어들었다.

#### GT-only warm-up

Layer B warm-up은 **GT(CG 계열) 전용**으로 정의되어 있다.

이유는 물리적이다.

- GT는 cold-start가 빠르다
- ST는 short-horizon lead time 안에서 준비시키기 어렵다
- 따라서 1시간 단위의 intraday planner에서는 GT가 사전 대비용으로 적합하다

즉 최종 planner는:

- Layer A startup도 GT 우선
- Layer B warm-up도 GT only

로 정리된다.

#### Online 호기 우선 재배분

Layer A 안에서 실제 분배는 여전히 online fleet 우선이다.

1. 기준 시점에 이미 돌고 있는 호기만 online subset으로 정의
2. `Headroom × Ramp × Priority` 비례로 분배
3. 호기별 headroom cap 적용
4. 부족분이 남으면 `ΔP_residual`로 분리

이 구조의 장점은 단순하면서도 해석 가능하다는 점이다.

- headroom이 큰 호기
- 빠르게 올릴 수 있는 호기
- 운영상 우선순위가 높은 호기

에 더 많은 추가 대응량이 간다.

다만 최신 plan은 priority 자체를 과장하지 않는다. 실제 실험에서는 priority보다 **headroom 효과가 훨씬 지배적**이었다. 예를 들어 CS1은 priority 1.0이 아니더라도 가장 큰 headroom 덕분에 누적 분담량 1위가 되었다. 따라서 이 planner는 “priority 기반”이라기보다 **현실적인 headroom/ramp 기반 allocator**라고 보는 것이 더 정확하다.

### 8.5 v2와 v4의 차이

#### v2

- CS2 단일 호기 중심
- 큰 online headroom 가정
- 설명은 쉽지만 실제 fleet 제약 반영이 약함

#### v4

- 10 LNG units (CS1, CS2, CG1~8)
- 실제 2025 baseline loading 반영
- 각 호기가 이미 일정 수준 운전 중이라는 현실 제약 포함

즉:

> `v2`는 idealized upper-capacity response,  
> `v4`는 realistic fleet-constrained response

로 보는 것이 맞다.

### 8.6 현재 결과와 의미

최신 LNG plan에서 평가 기준은 shortfall 하나가 아니다. 현재는:

- `shortfall`
- `over-commit`
- `shortfall / over-commit 비율`

을 함께 보고, 그중에서도 **over-commit 감소를 1순위**로 둔다.

이 원칙은 단순 취향이 아니라 실험 결과에서 나온 결론이다. 초기 planner는 큰 cloud-pass event를 더 잘 메우는 방향으로 가면 쉽게 “좋아 보일” 수 있었지만, 실제로는 작은 회복 구간과 평상시에도 LNG가 계속 따라가면서 과도한 추가 출력을 만들었다. 따라서 본 과제는 “최대한 많이 메우는 planner”가 아니라 **작은 변동에 과잉 반응하지 않는 planner**를 채택했다.

최신 production형 spec의 핵심 KPI는 다음과 같다.

| 지표 | 결과 |
|---|---:|
| shortfall | **442 MWh** |
| over-commit | **640 MWh** |
| 비율 | **1.45x** |
| warm-up (예열 명령) 발동 | 20회 (전부 GT) |
| **신규 가동** (offline → 발전, cold-start 비용) | **0회** |
| **추가 발전 전환** (이미 운전 중 + ΔP 시작) | 941회 |
| 1년 총 추가 발전 | 2,448 MWh (시간당 평균 0.75 MW) |
| 기존 단일 호기 cover (v2) 대비 | **−89.9%** |

**startup 분리의 운영 의미** (EDA F):
- 진짜 신규 가동 (cold-start 비용 발생) 1년 0회 — 운영비 크게 절약 (~5천만원/회 회피)
- 941회는 모두 *이미 baseline 운전 중인 호기가 PV 변동 대응으로 ΔP 시작* — cost-aware screening 데이터 부재 상태에서도 자연스럽게 비용 효율 운영

이 수치는 최종 spec이 단순히 shortfall을 조금 줄인 것이 아니라, **over-commit을 매우 큰 폭으로 줄인 결과**라는 것을 보여준다.

이를 가능하게 만든 핵심 변화는 다음과 같다.

1. `RG` 중심 반응을 줄이고 `instant_gap` 직접 사용
2. Layer A / Layer B 분리
3. Layer B = warm-up only
4. GT-only warm-up
5. DEADBAND=4 MW 도입

즉 planner는 지금 “예측이 나빠지면 무조건 LNG를 많이 올리는 구조”가 아니라:

> **실제 LNG가 책임지는 부족분만 계산하고, 현재 출력과 미래 대비를 분리해, over-commit을 최소화하는 구조**

로 정리되었다.

### 8.7 v2와 v4 비교의 해석

override 반영 후 planner 결과는 다음과 같이 해석된다.

- v2 + override: 단일 호기 가정으로 일부 이벤트에서 매우 공격적으로 shortfall 흡수
- v4 + override: overall shortfall은 더 낮지만, 큰 이벤트에선 online headroom 한계가 드러남

즉 v4는 “나쁜 결과”가 아니라, **현실 제약을 드러낸 더 정직한 결과**다.

v2는 CS2 단일 블록이 큰 폭으로 움직일 수 있다고 보는 이상화된 가정에 가깝고, v4는 실제 2025 baseline 위에 이미 돌아가고 있는 10개 호기의 여유만 사용하는 구조다. 따라서 일부 이벤트에서 v2가 더 좋아 보인다는 것은 v2가 더 현실적이라는 뜻이 아니라, **현실 제약을 덜 반영했기 때문에 더 낙관적인 것**에 가깝다.

특히 residual 약 888 MWh가 남는다는 것은:

> online fleet allocator만으로는 부족하고, offline startup screening이 다음 단계로 필요하다

는 뜻이다.

### 8.8 planner의 현재 한계

현재 planner는 다음을 아직 직접 해결하지 않는다.

1. full startup cost optimization
2. cold start trajectory 최적화
3. SMP 포함 순이익 최적화
4. 계통 전체 제약

**잔존 over-commit 640 MWh 의 메인 원인** (EDA B):

4년 hourly LNG 운영 데이터 분석 결과, 호기 자체의 *하강 한도(ramp_dn)* 가 매우 비대칭임을 확인:

| 호기 | mode | 상승 p90 | 하강 p90 | 비대칭 비율 |
|---|---|---:|---:|---:|
| CG6 | baseload | 18.2 MW/h | **1.87 MW/h** | **9.7x** |
| CS2 | baseload | 18.5 MW/h | **2.74 MW/h** | 6.8x |
| CG7 | mid-merit | 25.7 | 3.19 | 8.1x |

- 한 번 올라간 출력이 회복 후에도 *천천히* 만 내려갈 수 있음 (실제 운영 패턴)
- effective_gap 이 줄어도 호기 ΔP 는 ramp_dn 한도 만큼만 감소 → 잔류 ΔP = over-commit
- 이는 *모델이 만든 인위적 결과 X*. 실제 운영 데이터 그대로의 비대칭.

향후 추가 개선 옵션 (`옵션 A` in `lng/plan.md §15`):
- ramp_dn p99 사용 (CG6 1.87 → 48 MW/h) → 더 빠른 release, over-commit 추가 감축
- 다만 운영 직관 약화 (실 plant 도 천천히 내림이 일반)
- 본 PoC 에서는 1.45x 비율 = 합격선 으로 보고 미적용

**가능한 것**:

1. unit-level 추가 출력량 추정
2. shortfall / over-commit 비교
3. state stability 분석
4. residual 발생 시 offline screening 필요량 제시
5. 진짜 신규 가동 vs 추가 발전 전환 분리 측정 (운영비 측면)

즉 planner는 아직 상용급 dispatch optimizer는 아니지만, **운영지원 PoC로서 충분히 해석 가능한 수준**까지 올라와 있다.

---

## 9. 대시보드 및 시연 구조

본 과제의 시연 포인트는 단순 예측 그래프가 아니다. 최종 대시보드는 다음 흐름을 보여준다.

1. Phase 1 baseline
2. Phase 2 remaining-day reforecast
3. outage override 이후 최종 forecast
4. gap과 backup plan
5. LNG 호기별 추가 대응량

이 구조는 심사 관점에서 중요하다. 왜냐하면 본 과제의 강점은 “점수가 몇 % 좋아졌다”보다:

> **예측 -> 사건 해석 -> backup planning -> 호기별 배분 -> 운영 시각화**

로 이어지는 end-to-end flow를 보여줄 수 있다는 데 있기 때문이다.

즉 대시보드는 부가 기능이 아니라, 본 PoC가 실제 운영 판단 구조를 갖고 있음을 보여주는 핵심 산출물이다.

---

## 10. EDA 및 비채택 검토 결과

### 10.1 왜 비채택 결과도 중요한가

이 프로젝트는 단순히 성공한 실험만 나열하면 오히려 설득력이 약해진다. 실제로는 여러 대안을 검토했고, 그중 일부는 실패했다. 그러나 그 실패가 최종 구조의 정당성을 오히려 강화했다.

### 10.2 비채택 대안

본 과제에서 검토했으나 최종 production에는 채택하지 않은 항목은 다음과 같다.

1. Phase 1 outage masking + retrain
2. Phase 2 integrated outage head
3. sparse gate sharpening
4. 단일 호기 planner를 최종 구조로 유지하는 해석

### 10.3 EDA로서 남는 가치

이들 정보는 현재 안 쓰더라도 다음 역할을 한다.

1. 왜 outage가 모델 내부 학습보다 운영 레이어에서 더 잘 처리되는지 설명
2. 왜 cloud-pass와 outage-like event를 구분해야 하는지 근거 제공
3. 왜 v4 allocator가 더 현실적인지 설명
4. 어떤 한계가 아직 남아 있는지 투명하게 보여줌

즉 비채택 결과는 단순 폐기물이 아니라, **최종 구조 채택의 근거 자료**다.

### 10.4 LNG planner v4 spec 변화 누적 결과 (실험 추적)

LNG planner 도 단번에 현재 spec 으로 정착한 것이 아니다. 다음 5 단계 실험을 거쳐 누적적으로 개선됨:

| 단계 | 변경 사항 | shortfall | over-commit | 비율 |
|---|---|---:|---:|---:|
| (a) v4 초기 (priority 가중 + Layer B 출력 + RG) | — | 3,236 | 8,003 | 2.5x |
| (b) +Layer A/B 분리 + priority 제거 | Layer B = warm-up only | 3,955 | 7,011 | 1.8x |
| (c) +Layer A `instant_gap` 추가 (RG 와 병기) | cloud-pass 회복 | 1,185 | 7,247 | 6.1x |
| (d) +RG (look-back 평균) 폐기 | over-commit −25% | 1,197 | 5,398 | 4.5x |
| **(e) +DEADBAND=4 MW 도입 ★ 현재** | 계통 자체 흡수 모델링 | **442** | **640** | **1.45x** |

각 단계의 *가설 → 실험 → 결과 → 다음 가설* 로 진행되었으며, 모든 단계가 EDA 결과 또는 사용자 직관 (보조서비스 정산금 분석 등) 으로 정당화됨.

(a) → (e) 누적 변화:
- shortfall **−86%** (3,236 → 442)
- over-commit **−92%** (8,003 → 640)
- vs v2 baseline (CS2 단일 호기): **−89.9%**

### 10.5 EDA 결과 산출물 위치

본 과제의 EDA 결과는 두 폴더로 분리되어 저장된다.

**(A) `pv/eda_pv_model/` — PV 예측 모델 측 EDA (motivating)**

| EDA | 그림 | csv | 입증 메시지 |
|---|---|---|---|
| 5A.1 | `M1_site_pattern_heterogeneity.png` | `M1_site_{hour,month}_cf.csv` | AdaLN site/hour conditioning |
| 5A.2 | `M2_temporal_scale.png` | `M2_autocorr.csv` | Phase 2 / H=6 정당화 |
| 5A.3 | `M3_input_marginal.png` | `M3_{corr,hour_corr}.csv` | ASOS + GK2A 조합 |
| 5A.4 | `M4_outage_separability.png` | — | rule-based outage |
| 5A.5 | `M5_residual_pattern.png` | `M5_{residual_autocorr,hour_residual,site_residual}.csv` | Phase 2 residual correction |

추가 자료: `_summary.md` (메타 SSOT), `_selection_results/` (sweep 결과 부록), `pv/notebooks/00~13` (14 EDA 노트북), `pv/experiments/` (60+ 모델 실험).

**(B) `pv/eda_lng_planner/` — LNG planner 측 EDA**

| EDA | 그림 | csv | 입증 메시지 |
|---|---|---|---|
| 5B.2 | `D_site_outage_heatmap.png`, `D_site_cf_violin.png` | `D_site_stats.csv` | 사이트별 outage 패턴 |
| 5B.4 | `C_forward_gap_distribution.png` | — | 임계값 정당화 |
| 5B.5 | `E_hourly_burden.png` | `E_hourly_stats.csv` | 시간대별 운영 부담 |
| 5B.6 | `H_unit_pattern.png` | `H_unit_pattern.csv` | 호기 운전 패턴 |
| 5B.7 | `B_ramp_asymmetry.png` | `B_ramp_stats.csv` | over-commit 원인 |
| 5B.8 | `A_deadband_curve.png` | `A_deadband_results.csv` | DEADBAND=4 정당화 |
| 5B.9 | `G_phase2_marginal.png` | `G_phase2_daily.csv` | Phase 2 가치 분포 |
| 5B.10 | `F_startup_split.png` | — | startup 분리 |

---

## 11. 기대효과

### 11.1 정량적 기대효과

최종 구조 기준으로 기대되는 직접 효과는 다음과 같다.

1. Phase 1 대비 전체 NMAE 개선
2. blackout/outage-like 시점 오차 극적 감소
3. planner shortfall 감소
4. over-commit 대폭 감소
5. sign flip 및 과대 진동 감소

### 11.2 정성적 기대효과

정성적 효과는 다음과 같다.

1. 예측과 운영을 분리하지 않은 통합 구조 제시
2. weather-driven event와 outage-like event를 구분하는 운영 인사이트 제공
3. LNG 추가 대응량을 단일 블록이 아닌 fleet 관점으로 설명 가능
4. 운영자가 실제로 이해할 수 있는 dashboard 시연 가능
5. 계통 자체 흡수분과 LNG 직접 대응분을 구분하는 현실적 운영 논리 제시

### 11.3 운영 인사이트

본 과제는 예측 성능 외에도 site-level 운영 패턴을 드러냈다.

- 광양항세방 10월 shutdown cluster
- 고흥만수상 점심시간 반복 outage
- 구미의 신규 outage-like event

이는 단순 forecasting beyond-score의 의미를 가진다. 즉 모델은 예측만이 아니라 **운영 데이터의 이상 패턴을 드러내는 보조도구**로도 활용될 수 있다.

---

## 12. 한계와 향후 확장

### 12.1 한계

본 과제의 한계는 명확히 적어야 한다.

1. 기상 입력은 실제 archived forecast가 아니라 proxy다.
2. LNG baseline은 실제 D-1 thermal schedule이 아니라 2025 actual proxy다.
3. planner는 full UC와 시장 최적화를 직접 수행하지 않는다.
4. offline startup screening은 다음 단계 구현 과제로 남아 있다.
5. DEADBAND는 PoC 수준의 운영 가정이며, 실제 계통 규칙과 1:1로 동일하다고 주장하지 않는다.

이 한계는 약점이지만, PoC라는 과제 범위 안에서는 동시에 합리적인 제한이기도 하다.

### 12.2 향후 확장 방향

향후 확장 가능성은 분명하다.

1. 실제 forecast history 연동
2. offline startup screening 구현
3. 호기별 startup cost / variable cost 추정 고도화
4. portfolio 범위 확대 및 더 넓은 자원 집합 적용
5. 실제 운영데이터 연동 시 shadow-operation PoC로 확장

---

## 13. 결론

본 과제는 단일 예측모델의 정확도를 겨루는 프로젝트가 아니다. 최종적으로는:

- `Phase 1`이 D-1 기준선을 만들고
- `Phase 2`가 최신 actual을 반영해 남은 하루 곡선을 다시 그리며
- `Outage Override`가 비기상 운영 사건을 명시적으로 분리 처리하고
- `LNG Planner`가 DEADBAND를 차감한 `effective gap`만을 LNG 책임분으로 보고, `Layer A(실 발전)` / `Layer B(warm-up only)`로 역할을 분리해 기준 LNG 계획선 위에서 추가 대응량을 호기별로 재배분하는

**운영 지원형 예측-의사결정 통합 PoC**

를 제시하는 것이 본 과제의 핵심이다.

즉 본 과제는 다음 두 가지를 동시에 보여준다.

1. AI 기반 신재생 발전량 예측이 실제로 얼마나 운영 가치로 이어질 수 있는가
2. 그 예측을 화력 백업 계획과 호기별 대응으로 연결하는 구조가 공공데이터 기반 PoC 수준에서도 성립하는가

특히 LNG planner는 작은 편차까지 무조건 LNG가 따라가게 하지 않고, 계통이 자체 흡수할 수 있는 영역과 LNG가 실제로 책임져야 하는 영역을 분리한다는 점에서, 단순한 “예측 오차 보정기”보다 더 운영적인 의미를 갖는다.

이 점에서 본 과제는 공모전 요구사항인:

- AI 기반 발전량 예측 알고리즘 구축
- 신재생 변동에 따른 화력 발전 부하 배분 운영 로직 설계

를 분리된 두 과제가 아니라 **하나의 end-to-end 운영 지원 서비스**로 제시하는 데 의미가 있다.
