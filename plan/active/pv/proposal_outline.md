# PV 변동성 대응형 예측·운영 지원 플랫폼 기획서 초안 (v2 재구성)

> **재구성 일자**: 2026-05-10
> **이전 버전**: `proposal_outline_v1_backup.md`
> **재구성 원칙**: EDA(데이터 성질) / 모델 설계(선택·sweep) / planner 설명을 명확히 분리.
>
> 기준 문서:
> - PV forecasting: `plan/active/pv/model_final.md`
> - LNG planner: `plan/active/lng/plan.md`
> - 데이터 SSoT: `plan/active/main/data_strategy.md`
> - 상위 framing: `plan/active/main/framing.md`

---

## 1. 필요성 및 목적

### 1.1 재생에너지 비중 확대와 변동성 문제

대한민국의 전력 공급 구조는 최근 5년 사이 큰 폭으로 변화하였다. 한국에너지공단의 「KEA 에너지 이슈 브리핑」에 따르면 2024년 신·재생에너지 발전 비중은 **10.6%** 로 처음으로 두 자릿수를 돌파하였으며, 2025년 4월 기준 태양광 단독 발전 비중은 **9.2%** 로 직전 최고치 (2024년 5월 8.7%) 를 넘어섰다. 정부의 제11차 전력수급기본계획에 따르면 신·재생에너지 설비용량은 2025년 39 GW에서 2038년 **121.9 GW (이 중 태양광 77.2 GW)** 로 약 3배 증가할 예정이다.

이러한 양적 확대는 동시에 운영 측면의 새로운 부담을 가져왔다. 한국전력거래소 자료에 따르면 호남권 태양광 출력제어 (curtailment) 는 2024년 1년 26건 (7,473 MW) 수준에서 **2025년 상반기에만 44건 (38,840 MW)** 으로 급증하였다. 봄·가을 경부하기 대책 운영 기간 또한 2023년 61일 → 2024년 72일 → 2025년 93일 → **2026년 107일** 로 매년 확대되고 있다. 이는 *시·분 단위의 변동성* 이 계통 운영 의사결정에 미치는 영향이 점점 커지고 있음을 의미한다.

### 1.2 평균 정확도가 아니라 *시간 단위 계획 이탈* 이 문제다

태양광은 일사가 양호한 날에도 부분 흐림, 구름 통과, 사이트 단위 설비 정지 등의 이유로 정오 전후 급격한 출력 저하와 회복을 보인다. 따라서 운영자가 매시간 마주하는 실제 문제는 *오늘 하루 총 발전량이 얼마나 맞았는가* 가 아니라, **D-1 계획 대비 실시간 편차의 방향과 지속성** 이다.

> 첫째, D-1 기준으로 세운 발전 계획과 현재 실적의 차이가 얼마나 벌어졌는가.
> 둘째, 그 편차가 앞으로 1~3 시간, 나아가 남은 하루 동안 지속될 것인가.
> 셋째, 그 편차를 보완하기 위해 화력 자원을 언제, 얼마나, 어떤 순서로 준비해야 하는가.

이 세 질문은 단일 예측 모델의 leaderboard 점수만으로는 답할 수 없다. 운영자가 진짜로 필요로 하는 것은 *예측 정확도* 자체가 아니라, **예측 결과를 그대로 화력 백업 의사결정에 연결할 수 있는 운영 지원 구조** 이다.

### 1.3 화력 백업의 구조적 부담

한국 전력계통은 재생에너지 변동을 흡수하기 위해 **1차 (Governor Free)·2차 (AGC)·3차 (대기 발전기 기동)** 의 운영예비력을 운용하고 있으며, ESS·양수·가스터빈 등 유연성 자원의 비중을 확대하고 있다. 한국전력거래소 보조서비스 정산 자료에 따르면 LNG 복합화력은 *빠른 응답성과 시간대별 부하 추종 능력* 으로 보조서비스 정산금의 **약 50%** 를 담당한다. 즉 LNG 는 신재생 변동 대응의 핵심 자원이다.

그러나 LNG 호기는 *짧은 시간에 빠르게 증발* 할 수는 있어도 한 번 올라간 출력을 다시 내릴 때는 운영 관행상 매우 천천히 감소한다. 본 과제에서 분당화력 LNG 10 호기의 4년치 시간별 운영 데이터를 분석한 결과, baseload 호기 (CG6, CS2) 의 시간당 상승 한도 (p90) 대비 하강 한도 (p90) 비율이 **약 7~10 배 비대칭** 임을 확인하였다. 변동 대응 시점의 *과대 보충 (over-commit)* 은 모델이 만든 인위적 결과가 아니라 *실제 운영 데이터 자체에서 관찰되는 구조적 비대칭* 의 결과이다.

### 1.4 본 과제가 답할 문제

> **태양광 D-1 기준 계획 대비 실시간 편차를 감지하고, 남은 하루 출력 곡선을 재예측하며, LNG 기준 운전 상태 위에서 필요한 추가 대응량을 호기별로 재배분하여, 부족 (shortfall) 과 과대 보충 (over-commit) 을 동시에 줄이는 운영 지원 구조의 PoC 구축.**

### 1.5 공모전 요구사항과의 정합성

남동발전이 제시한 본 공모전의 요구사항은 다음 두 줄로 요약된다.

- **AI 기반 발전량 예측 알고리즘** 구축
- 신재생 출력 변동에 따른 **화력 발전 부하 배분 운영 로직** 설계

본 과제는 이 두 요구를 분리된 두 산출물이 아니라 **하나의 운영 지원 파이프라인** 으로 통합한다.

```
D-1 baseline 예측 → 당일 갱신 예측 → 정지 사건 분리 → 차이 해석
   → LNG 백업 계획 → 호기별 재배분 → 대시보드 운영 시각화
```

본 과제의 핵심 기여는 *예측 모델의 정확도 향상* 이 아니라, **예측 결과가 운영 레이어로 직접 이어지는 구조를 PoC 수준에서 입증하는 것** 이다.

### 1.6 PoC 형태의 합리성

남동발전과의 사전 질의응답 결과 다음 조건이 확인되었다.

- SPC (특수목적법인) 데이터 등 내부 운영 데이터는 제공되지 않음
- 발전기별 실제 운전 파라미터 (가동 비용, 기동 비용 등) 는 비공개
- 기대 산출물은 상용 시스템이 아니라 **대시보드를 포함한 데모 가능한 PoC**

본 과제는 KPX 급전 재현기나 시장 최적화기를 만드는 것이 아니라, **공공 데이터와 공개 실적만으로도 실시간 운영 지원 구조의 설계가 가능한지** 를 검증하는 *개념검증 (Proof of Concept)* 이다.

### 1.7 정량적 / 정성적 / 산출물 목적

**정량적 목적**:
1. D-1 baseline 대비 당일 갱신 예측 (Phase 2) 으로 *남은 하루 곡선의 재예측 능력* 확보
2. 기상 변동 사건과 비기상성 정지 사건을 명시적으로 분리
3. planner 의 **부족분 + 과대 보충 동시** 감소
4. 단일 화력 블록 가정을 넘어 *분당화력 LNG 10 호기 fleet 기반* 재배분 구조 제시

**정성적 목적**:
1. AI 예측과 운영 의사결정을 하나의 흐름으로 연결
2. 운영자 관점의 *언어* 로 사건 해석 (자체 흡수폭 / 호기 예열 명령 / 추가 대응 모드)
3. 향후 실제 데이터 연동 시 즉시 확장 가능한 구조

**최종 산출물**: 예측 모델 설명 가능 + planner 로직 설명 가능 + KPI 검증 가능 + 4페이지 대시보드 시연 가능 — *서비스형 구조 검증* PoC.

---

## 2. 데이터 구성과 제약

### 2.1 원천 데이터 — 4년 hourly 공공·실측 자료

본 과제는 *내부 운영 데이터를 일체 사용하지 않고* 공공 데이터 4종으로 학습·평가한다.

| 구분 | 소스 | 해상도 | 기간 | 용도 |
|---|---|---|---|---|
| **태양광 발전 실적** | 남동발전 홈페이지 (시간별, 호기별) | 1시간 | 2022-01 ~ 2025-12 (48개월) | 예측 모델 타깃 |
| **LNG 발전 실적** | 남동발전 홈페이지 (분당화력 10 호기) | 1시간 | 2022-01 ~ 2025-12 | LNG planner baseline + 호기 spec 추정 |
| **GK-2A 위성 일사** | 기상청 NCDC SFTP (DSR/ASR/RSR + DQF) | **10분 → 1시간 집계** | 2022-01 ~ 2025-12 | 광역 일사·반사·흡수 |
| **ASOS 지상 기상** | 기상청 11개소 (기온/습도/풍속/강수) | 1시간 | 2022-01 ~ 2025-12 | 지상 기상 측정 |

추가 참고자료: 발전소 운영 현황, 전력거래 입찰 정보, 신재생에너지 사업현황 (`data/info/*.csv`).

### 2.2 PoC 의 명시적 제약

| 제약 | 내용 | 본 PoC 의 대응 |
|---|---|---|
| **SPC 태양광 513 MW 미공개** | 별도 법인 운영 데이터 비공개 | 자체 설비 22 호기로 학습 한정 |
| **호기별 운전 파라미터 미공개** | 가동 비용 / 기동 비용 / 변동비 / 기동 시간 등 | 4년 hourly 데이터로 *역추정* (P_min, P_max, ramp 등) |
| **D-1 화력 계획 미제공** | KPX 신고 시점의 thermal schedule 부재 | 2025 실측 LNG 를 *baseline proxy* 로 사용 |
| **D-1 기상 archived forecast 미보유** | KMA LDAPS 별도 신청 + TIGGE 50km 부적합 | ASOS·GK-2A 실측을 **perfect-foresight proxy** 로 사용 |

### 2.3 태양광 호기 필터링 — 22 → 11

PV 예측 모델이 학습 가능하려면 입력 신호가 *순수 태양광 발전량* 이어야 한다. 다음 두 조건이 이 전제를 무너뜨린다.

- **시간별 데이터 미제공** — 일합계만 공개되어 시간별 학습 불가
- **ESS 혼재** — 낮 발전을 *오후·저녁* 에 방출하여 피크가 16~18시 로 이동

자연 PV 의 피크는 정오 ±2 시간 (10~14시) 이 물리적 정상 영역이다. 따라서 본 과제는 4년 hourly 실측 데이터에서 호기별 평균 시간 프로필을 산출하여 단계별로 필터링하였다 (`pv/eda_pv_model/D1_data_constraints.png`).

| 단계 | 기준 | 투입 → 산출 | 제외 호기 |
|---|---|---|---|
| 원자료 | 자체 설비 전체 | 22 → 22 | — |
| Step 1 | 시간별 데이터 제공 | 22 → 20 | 여수, 탑선 |
| Step 2 | 피크 시각 ≤ 15시 (Pure PV) | 20 → 12 | 삼천포 #4·#5·#6, 영동, 영흥 #3 (8 호기, 16~18시 피크) |
| Step 3 | 가동 축소·중단 제외 | 12 → **11** | 삼천포 #1 (2025-04 부터 거의 정지) |

ESS 왜곡 호기로 학습하면 *ESS 운영 패턴 추정 모델* 이 되어 본 PoC 의 목적과 직접 충돌한다.

**최종 11 호기 (합계 ~77 MW)**:

| # | 사이트 | 용량 (kW) | 비중 | 비고 |
|---:|---|---:|---:|---|
| 1 | **고흥만 수상태양광** | **63,481** | **82.1%** | ★ portfolio dominant |
| 2 | 영흥태양광#5 | 3,500 | 4.5% | |
| 3 | 광양항세방태양광 | 2,993 | 3.9% | 항만 microclimate, bias +7% |
| 4 | 예천태양광 | 2,000 | 2.6% | 영농형, anomaly_zero 4.95% |
| 5~10 | 삼천포·영흥·구미·경상대 | 5~6 호기 | ~4% | 다양성 확보 |
| 11 | 두산엔진MG (창원) | 77 | 0.1% | 노이즈 검증 |

### 2.4 학습/검증/테스트 분리

| 데이터 | 분리 |
|---|---|
| Phase 1 (D-1 baseline) | 2022-2023 train / 2024 valid / 2025 test |
| Phase 2 (intraday reforecast) | 2024 train / 2025 test |

### 2.5 운영 데이터 부재의 *proxy* 처리

**(a) 기상예보 proxy** — D-1 archived forecast 부재 → ASOS·GK-2A 실측을 *perfect-foresight weather proxy* 로 사용. 따라서 본 PoC 의 모든 PV 모델 성능 수치는 *예보 모델 자체가 도달 가능한 상한선* 이다 (실제 운영 D-1 환경에서는 NWP forecast 오차가 추가됨).

**(b) D-1 thermal plan proxy** — 실제 KPX 화력 계획선 부재 → 2025 실측 LNG hourly 를 *그날의 기준 운전 상태* 로 해석. dispatch 정답 복원이 아니라 *baseline 위 추가 대응량만 계산* 하기 위한 framing.

이 두 proxy 는 본 PoC 의 한계인 동시에 *PoC 라는 범위에서 합리적인 가정* 이며, 향후 실제 데이터 연동 시 즉시 확장 가능한 구조로 설계되었다 (§11).

---

## 3. 변수 선정

> 본 절은 **무슨 입력을 썼고 왜 썼는가** 를 다룬다. EDA(데이터 성질 관찰)는 §4, 모델 비교는 §5.

### 3.1 변수 선정 원칙

본 과제의 입력 변수는 다음 세 원칙 위에서 선정되었다.

**(a) 공공 데이터로 확보 가능** — 본 PoC 는 *운영 환경에 즉시 옮길 수 있는 구조* 를 목표로 한다. 따라서 학습 시점에 사용한 모든 변수는 운영 시점에도 동일한 형태의 *공공 forecast* 또는 *실시간 측정* 으로 변환 가능해야 한다. 내부 SCADA 신호, SPC 운영 데이터, 호기별 운전 비용 등 *공개되지 않은 변수* 는 일체 사용하지 않는다.

**(b) 물리적 / 운영적 해석 가능 신호** — 모든 입력 변수는 cf (capacity factor) 와의 *물리적 단조 관계* 또는 *운영자에게 해석 가능한 신호* 여야 한다. 예: 위성 일사 (dsr) ↑ → cf ↑, 습도 (hm) ↑ → 산란 손실 → cf ↓. *임의 random feature, 차원 축소된 latent vector, 비식별 cluster ID* 같은 black-box feature 는 도입하지 않는다. 이는 운영자 화면에서 *왜 이 호기가 예열 명령을 받았는가* 를 *입력 변수 단위로 추적* 가능하게 만들기 위함이다.

**(c) 학습-예보 단위 일관성** — 학습 시 입력과 운영 시 forecast 입력의 *형태·단위·해상도* 가 일치해야 한다. 그렇지 않으면 실 운영 환경에서 학습된 모델을 사용할 수 없다. 본 과제는 학습 시 *위성 실측 + ASOS 실측* 을 입력으로 사용하지만, 운영 시에는 *Open-Meteo NWP forecast (GHI) + 기상청 단기예보 (기온/습도/풍속)* 로 swap 가능하도록 단위·해상도·변수 정의를 미리 일치시켰다 (§3.7 학습-예보 일관성).

### 3.2 사용 원천 데이터 4종

본 PoC 의 모든 입력 변수는 다음 4종 공공 원천 데이터에서 파생된다.

| 원천 | 위치 | 해상도 | 기간 | 직접 입력 / 파생 |
|---|---|---|---|---|
| `solar_hourly_*.csv` (남동발전) | `data/solar_hourly/` | 1시간, 호기별 | 2022-01 ~ 2025-12 | 학습 타깃 (cf), Phase 2 의 *최근 actual* 입력 |
| `asos_hourly_*.csv` (KMA) | `data/asos_hourly/` | 1시간, 11개소 | 2022-01 ~ 2025-12 | 기온 (ta), 습도 (hm), 풍속 (ws), 강수 (rn) 직접 입력 |
| GK-2A NC 원본 (NCDC SFTP) | `data/gk2a_raw/` | 10분, LCC 2km | 2022-01 ~ 2025-12 | DSR/ASR/RSR 추출 후 시간 집계 → `dsr_mean` |
| `thermal_hourly_*.csv` (남동발전) | `data/thermal_hourly/` | 1시간, 호기별 | 2022-01 ~ 2025-12 | LNG planner 의 `P_DA`, 호기 spec 추정 |

### 3.3 PV 입력 변수군 (Phase 1 / Phase 2)

PV 모델의 입력은 4 군으로 분류된다 — *기상*, *시간*, *사이트*, *실적 기반* (Phase 2 한정).

#### 3.3.1 기상 변수군

기상 변수는 *광역 일사·구름 정보 (위성)* + *지상 미세 기상 (ASOS)* 의 *상호 보완* 구조다.

| 변수 | 출처 | 단위 | 단일 cf corr | 운영 의미 |
|---|---|---|---:|---|
| **`dsr_mean`** ★ | GK-2A 위성 DSR (10분 평균) | W/m² | **+0.72** | *Downward Shortwave Radiation* — 사이트 위 픽셀에 도달하는 광역 일사. 구름이 차단한 후의 *실제 입사 일사*. cf 의 *물리적 직접 결정 변수*. |
| **`dc10Tca`** | GK-2A 파생 cloud index (10분 → 시간 집계) | 0~10 | (event branch 입력) | *부분 흐림 (dc 3-7) vs 강한 흐림 (dc 8+)* 구분. partial cloud 이벤트 식별. |
| **`zenith_center`** | 천문 계산 (lat / lon / time) | ° | -0.61 | *태양 천정각*. daily seasonality 의 결정적 정보. AdaLN conditioning 으로 흡수. |
| **`hm`** | ASOS 습도 | % | -0.54 | *흐림 / 안개 / 수증기에 의한 산란 손실*. dsr 만으로 못 잡는 *국지적 대기 상태* 정보. |
| **`rn`** | ASOS 강수 | mm | -0.20 | *비 오는 시간의 cf 저하*. discrete on/off 신호 가까움 (대부분 0). |
| **`ws`** | ASOS 풍속 | m/s | +0.22 | *모듈 냉각 효과*. 풍속 ↑ → 모듈 온도 ↓ → 효율 ↑. 한국 일반 풍속 영역에서 marginal. |
| **`ta`** | ASOS 기온 | °C | +0.13 | *모듈 효율 온도 의존성*. 한국 기온 영역에서 효과 작음. |

GK-2A 변수 *3종 (DSR / ASR / RSR)* 중 본 과제는 `dsr_mean` 만 직접 입력으로 사용한다. ASR (지표 흡수), RSR (지표 반사) 은 *파생 feature 후보* (예: 알베도 = RSR/DSR — 수상 PV 의 특이성) 로 유지하지만, 학습 안정성과 학습-예보 일관성 (Open-Meteo 가 ASR/RSR 미제공) 차원에서 *직접 입력에는 포함시키지 않았다*.

#### 3.3.2 시간 변수군

| 변수 | 형식 | 의미 |
|---|---|---|
| `hour` | 0~23 | hour-of-day. 일사 곡선의 결정적 형태 (정오 ±2h peak). |
| `month` | 1~12 | 계절성. 월별 일사량 / 기온 / 일조 시간 변화. |

이 두 변수는 *raw integer* 가 아니라 *AdaLN conditioning* 입력으로 사용된다 (§5.2). 즉 *site-specific × hour × month* 의 *동적 layer normalization* 에 들어가, 모델이 시간/계절 효과를 *조건부* 로 학습한다.

#### 3.3.3 사이트 변수군

| 변수 | 형식 | 의미 |
|---|---|---|
| `site_id` | one-hot (8) | *사이트별 정적 bias* 흡수 (광양항 +7%, 예천 anomaly 등) |
| `site_capacity_kw` | 정수 | 사이트 용량 (정규화 / cf 변환 시 사용) |
| `lat`, `lon` | 좌표 | 천정각 계산 + GK-2A 격자 매핑용 (모델 직접 입력 아님) |

*정적 site_oh 만으로는 부족* (§3.4 R² 분석에서 +0.012pp 만 추가). AdaLN 의 *동적 site conditioning* 이 결정적 — 사이트별로 시간/계절 패턴이 *함수 형태로 다름* (§4.1 EDA).

#### 3.3.4 실적 기반 변수군 (Phase 2 한정)

Phase 2 의 *residual correction* 에 사용되는 입력. *최근 actual* 정보가 *다음 시각 forecast* 의 가장 강한 신호임을 §4.3 에서 입증.

| 변수 | 정의 | 역할 |
|---|---|---|
| 최근 H=6h actual cf | 직전 6시간 실측 cf 시퀀스 | residual correction 의 *anchor* — Phase 1 baseline 위에 *최근 동향* 반영 |
| `drop_now_3h` | 직전 3h 평균 actual − μ_p1 차이 | event branch 의 *gate signal* — 최근 부족이 큰 경우 event 활성 |
| `abs_realized_gap_3h` | 직전 3h \|actual − μ_p1\| 평균 | gate signal — 최근 *변동 강도* |
| `gap_x_sigma` | gap 을 σ_p1 로 나눈 정규화 신호 | gate signal — *통계적 비정상성* (z-score 와 동일 개념) |

이 4 변수는 *Phase 1 의 점예측이 부족한 영역 (cloud-pass / outage-like)* 에서 *event branch* 를 활성화하기 위한 입력이다 (§5.3 의 2-branch + soft gate 구조).

### 3.4 누적 R² — 변수 추가별 cf 설명력 증가

학습셋 8,212 rows 정합 행에서 누적 R² 측정 (`pv/eda_pv_model/D2_variable_selection.png`):

| 변수 조합 | 변수 수 | R² (Linear) | R² (Random Forest) | Δ vs 이전 |
|---|---:|---:|---:|---:|
| ① zenith only | 1 | 0.054 | 0.230 | — |
| ② + dsr (GK-2A) | 2 | **0.613** ★ | **0.771** ★ | +0.559 / +0.541 |
| ③ + ASOS 기상 (ta, hm, ws, rn) | 6 | 0.633 | 0.827 | +0.020 / +0.056 |
| ④ + site_oh (8 dummies) | 14 | 0.645 | 0.839 | +0.012 / +0.012 |

**해석**:

1. **zenith → +dsr 단일 변수 추가만으로 R² 0.05 → 0.61 (linear 12배 / RF 3.4배)** — 위성 일사가 cf 의 *압도적 단조 결정 변수*. 이 결과가 GK-2A 입력 *없이는 PoC 자체가 성립 불가능* 함을 입증.

2. **+ASOS 4 변수의 marginal value** — Linear 에서는 +0.020 (작음) 이지만 RF 에서는 +0.056 (의미 있음) → ASOS 변수의 *비선형 marginal value* 가 큼. 특히 *부분 흐림 영역* 에서 dsr 만으로 못 잡는 산란 / 강수 / 모듈 온도 효과 보강.

3. **+site_oh 의 한계** — 정적 dummy 만으로 R² +0.012pp 만 추가. 사이트별 *시간/계절 패턴 차이* 를 잡으려면 *함수 형태 자체가 사이트별로 달라야* 함 → 단순 dummy 가 아니라 **AdaLN 의 동적 site × hour × month conditioning 이 필요** (§4.1 + §5.2 정합).

4. **Random Forest R² 0.84** — 4 변수군 + site_oh 만으로 cf 의 *84%* 가 예측 가능. 그러나 *남은 16% 가 운영 의사결정의 핵심 영역* — cloud-pass, outage, 사이트별 미세 bias 등. 이 영역은 단순 R² 향상이 아니라 *Phase 2 + Outage Override 의 구조적 분리* 로 풀어야 한다.

### 3.5 LNG 입력 변수군 (planner)

LNG planner 의 입력은 3 군으로 분류된다 — *PV gap*, *LNG baseline*, *호기 spec*.

#### 3.5.1 PV gap 변수군

planner 의 *control input*. PV 예측·실측의 차이를 LNG 추가 대응량으로 변환.

| 변수 | 정의 | 역할 |
|---|---|---|
| `instant_gap` | μ_p1(t) − pv_actual(t) | 현재 시각 실측 부족 (raw) |
| **`effective_gap`** ★ | max(0, instant_gap − DEADBAND_MW) | **LNG 책임분** (DEADBAND=4 MW 차감, §6.6) |
| `FG1` | μ_p1(t+1) − μ_p2(t+1, issue=t) | forward gap, 1시간 뒤 예상 부족 |
| `FGmean3` | mean(FG1, FG2, FG3) | 3시간 평균 |
| `FGslope` | (FG3 − FG1) / 2 | 시간당 변화 추세 (운영 모드 분류용) |

#### 3.5.2 LNG baseline 변수군

planner 의 *기준 운전 상태* — 추가 대응량 위에 더해질 baseline.

| 변수 | 정의 | 역할 |
|---|---|---|
| `P_DA,u(t)` | 2025 실측 호기별 시간 발전량 | *그날의 기준 운전 상태* proxy. 학습 X, 단순 lookup. |
| `is_online(u, t)` | (P_DA,u(t) > P_min,u × 0.5) | online 호기 판정 |
| `Avail_u(t)` | 직전 7일 같은 hour-of-day 운전 빈도 | offline 호기의 *가동 가능성* |

#### 3.5.3 호기 spec 변수군

4년 hourly 데이터에서 *역추정* 된 호기 운전 사양.

| 변수 | 정의 | 산출 방식 |
|---|---|---|
| `P_min` | 운전 중 출력 p05 | 4년 hourly 의 운전 시점 (mw>5) 에서 5% 분위 |
| `P_max` | 운전 중 출력 p95 | 95% 분위 |
| `Ramp_up` | 시간당 상승 한도 p90 | 운전 중 ΔP > 0 의 90% 분위 |
| `Ramp_dn` | 시간당 하강 한도 p90 | 운전 중 ΔP < 0 의 90% 분위 |
| `mode` | baseload (≥45%) / mid-merit (28~45%) / peaker (<28%) | running_pct (운전 시간 비율) 분류 |
| `priority` | 헤드룸 × ramp_up × running_pct/100 정규화 | 운영 ranking (분배 가중치 X, §7.7) |

이 모든 spec 은 *남동발전이 미공개* 한 변수를 *4년 hourly 운전 데이터로부터 직접 도출* 한 것이다 (§2.2 "호기별 운전 파라미터 미공개" 제약에 대한 대응).

### 3.6 사용하지 않은 변수 (명시적 제외)

| 변수 | 사용 안 함 이유 |
|---|---|
| **외부 archived weather forecast** (Meteologix, ECMWF archived ensemble 등) | KMA LDAPS forecast endpoint 별도 신청 (1~2일 대기 필요) + TIGGE 50km 해상도가 한국 PoC 부적합. 학습용 *일관된 forecast history* 를 PoC 시간 안에 확보 불가능 → ASOS·GK-2A 실측을 **perfect-foresight proxy** 로 대체. 본 PoC 의 명시적 한계 (§10.2). |
| **실제 D-1 thermal plan (KPX 신고 자료)** | 외부 미공개 → 2025 실측 LNG hourly 를 *그날의 기준 운전 상태* proxy 로 사용 (§7.2). dispatch 정답 복원이 아니라 *baseline 위 추가 대응만* 계산하는 framing. |
| **호기별 startup cost (SU) / variable cost (VC) / 기동 시간** | 남동발전 측 미공개. PoC 에서는 cost-aware screening 미구현. 단 *기동 시간 (GT 30분 / ST 1~3h) 은 일반 산업 spec 기반 추정* 으로 사용 (§7.6). |
| **SMP / 시장 정산 정보** | 본 PoC 의 평가는 *운영 KPI (shortfall / over-commit)* 중심. SMP 신호는 PBP 시장 전환 후 추가 가능 (§11.3). |
| **수동 운량 (cloud cover, total cloud)** | GK-2A `dsr_mean` 자체가 구름 영향을 *반영한 광역 일사값* 임. 별도 운량 변수가 dsr 와 강한 상관 → 중복 정보. *부분 흐림 강도 구분* 만 dc10Tca 로 보강. |
| **GK-2A ASR / RSR 직접 입력** | 학습-예보 일관성 위배 (Open-Meteo / 기상청 단기예보가 ASR/RSR 미제공). 다만 4년 데이터에 보존하여 *향후 알베도 파생 feature* 후보로 유지 (수상 PV 특이성 등). |

### 3.7 학습-예보 일관성 — 운영 환경 swap 시나리오

본 PoC 는 *학습 시점에 사용한 모든 변수* 가 *운영 시점에도 동일 형태로 입력 가능* 하도록 설계되었다. 변수별 학습/운영 매핑:

| 변수 | 학습 (실측) | 운영 (forecast / 실시간) | 단위 일치 |
|---|---|---|---|
| 광역 일사 (dsr) | GK-2A 위성 실측 (W/m²) | Open-Meteo GHI (NWP, W/m²) — MOS layer 예정 | ✓ |
| 기온 (ta) | ASOS 11 개소 실측 (°C) | 기상청 단기예보 API 동네예보 (°C) | ✓ |
| 습도 (hm) | ASOS 실측 (%) | 기상청 단기예보 (%) | ✓ |
| 풍속 (ws) | ASOS 실측 (m/s) | 기상청 단기예보 (m/s) | ✓ |
| 강수 (rn) | ASOS 실측 (mm) | 기상청 단기예보 (mm/3h) — 시간 변환 필요 | △ |
| 운량 (dc10Tca) | GK-2A 파생 (10분 → 1h) | 기상청 단기예보 / Open-Meteo 운량 | △ |
| 시각·날짜 | 결정적 | 결정적 | ✓ |
| 사이트 정보 | 정적 | 정적 | ✓ |
| 최근 actual cf (Phase 2) | 학습셋 직접 사용 | 실시간 SCADA / KOEN 발전 실적 (~1시간 지연) | ✓ |

→ 변수 정의 / 단위 / 시간 해상도가 학습과 운영에서 일치 → 운영 시 NWP forecast 입력으로 *단순 swap 가능* → 모델 재학습 없이도 운영 배포 가능 (단 perfect-foresight 가정의 *NWP forecast 오차 추가* 는 별도 measure 필요, §10.2).

### 3.8 변수 선정의 운영 의미 종합

| 변수 | 운영 시점 의미 |
|---|---|
| `dsr_mean` (GK-2A) | *광역 구름 패턴* — cloud-pass 1~3h 사전 신호. Phase 2 의 핵심 입력. |
| `dc10Tca` | *부분 흐림 / 강한 흐림* 구분 — event branch 활성 신호 |
| `zenith / hour / month` | *daily / seasonal 주기* — AdaLN conditioning 으로 흡수 |
| `hm` | *안개 / 부분 흐림* — DSR 만으로 못 잡는 산란 손실 |
| `rn` | *강수 시 cf 저하* — discrete 신호 |
| `ws / ta` | *모듈 냉각 / 효율* — marginal 보강 |
| `site_oh` (정적) + AdaLN (동적) | *사이트 bias + 시간/계절 conditioning* — 광양항 +7%, 예천 anomaly 등 처치 차별화 |
| 최근 H=6h actual + gate signals | *intraday correction* — Phase 2 의 anchor + event branch 활성 |
| `effective_gap` (LNG 입력) | *LNG 책임분* — DEADBAND 차감으로 자체 흡수 영역 분리 |
| `P_DA, headroom, ramp` (호기 spec) | *호기 fleet 의 운전 제약* — 데이터로부터 직접 추정 |

이 변수 조합으로 *Phase 1 (D-1 baseline) → Phase 2 (intraday reforecast) → Outage Override → LNG planner* 까지 *모든 의사결정 layer* 가 동일한 입력 단위로 연결된다.

---

## 4. PV 데이터 EDA

> 본 절은 **데이터의 성질만** 기록한다. *모델 비교 / sweep 결과* 는 §5 로 분리하였다.
> 위치: `pv/eda_pv_model/M1~M5_*.png`, `D1~D2_*.png` (motivating EDA + supplementary).
>
> 본 절의 6 소절은 모두 다음 질문에 답한다 — *왜 단일 점예측으로는 부족하고, 왜 Phase 2 / Outage Override / portfolio 기준 평가 가 필요한가*.

### 4.1 사이트별 이질성 (`M1_site_pattern_heterogeneity.png`)

본 PoC 의 8 사이트는 *capacity 4 자릿수 차이* + *시간/계절 패턴 차이* 가 동시에 존재한다. 단일 함수로 fit 불가능한 *근본적 이질성* 이다.

#### 4.1.1 capacity 분포

| 사이트 | capacity (kW) | portfolio 비중 | 환경 특성 |
|---|---:|---:|---|
| 고흥만수상 | 63,481 | **82.1%** | 수상 PV (낮은 알베도, 광역 평면) |
| 영흥태양광 #5 | 3,500 | 4.5% | 서해안, 옥상형 |
| 광양항세방 | 2,993 | 3.9% | 항만 microclimate (해풍, 안개 잦음) |
| 예천 | 2,000 | 2.6% | 영농형 (그림자, 작물 영향) |
| 영흥 #1, #2 | 1,000 + 993 | 2.6% | 서해안 |
| 삼천포 #2 | 990 | 1.3% | 남해안 |
| 구미 | 992 | 1.3% | 내륙 정수장 옥상 |
| 경상대 | 905 | 1.2% | 학교 옥상 |
| 삼천포 #3 | 350 | 0.5% | 부분 용량 |
| 두산엔진MG (창원) | 77 | 0.1% | 매우 작음 (노이즈 검증용) |

→ *고흥만수상 1 사이트가 portfolio 의 82%* 를 차지하는 *극단적 비대칭* 구조. 따라서 *site mean NMAE* 와 *portfolio NMAE* 는 *완전히 다른 의미* 를 가진다 (§4.6).

#### 4.1.2 사이트 × 시각 평균 cf 패턴

4년 hourly 평균 cf 의 hour-of-day 곡선 비교:

| 사이트 | peak 시각 | 곡선 형태 | 특이점 |
|---|---|---|---|
| 고흥만수상 | 13시 | 종 모양 (가장 일관) | 수상 → 안정 |
| 광양항세방 | 12-13시 | *조기 peak + 빠른 감쇠* | 항만 동향 + 해풍 |
| 영흥태양광 #5 | 13시 | 표준 종 모양 | 서해안 |
| 예천 | 13-14시 | *늦은 peak + plateau* | 그림자 / 작물 |
| 경상대 | 13시 | 표준 | 학교 옥상 |
| 창원 (두산MG) | 14시 | *지연 peak* | 소형 (77kW), 그림자 영향 |

#### 4.1.3 사이트 × 월 평균 cf 패턴

봄·가을 (4-5월, 9-10월) 이 일반적인 PV peak 월이지만 사이트별로 *peak 월이 다르다*:
- 고흥만수상 — 5월 peak (남해안 일사 강함)
- 광양항세방 — 4-5월 + 9-10월 *쌍peak* (해무 패턴)
- 예천 — 5월 peak + 7-8월 dip (장마 + 작물 그림자)
- 영흥 #5 — 5월 peak + 12월 dip (서해안 적설)

#### 4.1.4 사이트 간 daily curve cross-correlation

8 × 8 사이트 daily curve 의 Pearson 상관 행렬:

| 측정 | 값 |
|---|---|
| 평균 cross-correlation | 0.983 |
| 최소 cross-correlation | **0.921** (광양항-예천) |
| 최대 cross-correlation | 0.998 (영흥 #1-#2, 같은 위치 수개 호기) |

**관측**: 평균 0.98 로 *높지만 100% 동일은 아님*. 사이트 간 *최저 상관 0.92* — *7%의 형태 차이* 가 존재. 이 7% 가 *사이트별 bias / 변동성 차이* 다.

#### 4.1.5 시사점 — 모델 구조 함의

**(a) 단일 함수 fit 불가능** — 8 사이트의 daily curve / 월별 패턴 / peak 시각이 모두 달라, *동일 함수* 로 fit 시 *site-specific 잔차* 가 누적된다.

**(b) 정적 site_oh dummy 만으로는 부족** — §3.4 의 R² 분석에서 site_oh 추가가 +0.012pp 만 기여. *bias 만* 흡수하고 *형태 차이* 는 못 잡음.

**(c) AdaLN 의 동적 conditioning 필요** — *site × hour × month* 의 *3중 조건부 layer normalization* 이 사이트별 *함수 형태 자체* 를 다르게 만든다. 이는 §5.2 backbone 채택의 핵심 동기.

### 4.2 기상-출력 관계 (`D2_variable_selection.png`)

cf 가 기상 변수와 *어떻게 결정되는지* 의 함수 형태 분석.

#### 4.2.1 dsr (위성 일사) → cf 의 piecewise linear 영역

dsr_mean 분위 (Q1~Q10) × cf 분포 box plot:

| 분위 | dsr 중간값 (W/m²) | cf mean | cf std | 영역 해석 |
|---|---:|---:|---:|---|
| Q1 (저일사) | ~50 | 0.05 | 0.05 | 흐린 새벽/저녁, 강한 흐림 |
| Q2 | ~150 | 0.10 | 0.10 | 흐린 날 |
| Q3-Q4 | 200~400 | 0.20~0.30 | **0.18~0.20 ★** | **부분 흐림 — cf 분포 가장 큼** |
| Q5 | ~500 | 0.40 | 0.15 | 약한 흐림 |
| Q6-Q7 | 600~700 | 0.45~0.50 | 0.10 | 좋은 날 |
| Q8-Q10 | 750+ | 0.55~0.65 | 0.06 | clear-sky 영역, 분포 좁음 |

**핵심 관측**: cf 의 *분포 폭 (std)* 이 *고일사 영역 (Q8-Q10) 에서 0.06* 으로 좁고, *부분 흐림 영역 (Q3-Q4) 에서 0.18~0.20* 으로 *3배 산포*.

#### 4.2.2 변수별 cf corr — 기상 변수의 marginal value

전체 daytime (n=171,907) 에서:

| 변수 | corr | 시간대별 안정성 | 비고 |
|---|---:|---|---|
| `dsr_mean` (GK-2A) | **+0.72** | 9~16시 일관 +0.62~+0.75 | *광역 일사* — 절대적 결정 변수 |
| `zenith_center` | -0.61 | 9~16시 -0.49~-0.71 | daily seasonality |
| `hm` (습도) | -0.54 | 9시 -0.39, 13시 -0.55 | *흐림 강도 proxy* — 정오에 가장 강함 |
| `ws` (풍속) | +0.22 | 시간 의존 큼 (15-16시 +0.24) | 모듈 냉각 (오후 효과) |
| `rn` (강수) | -0.20 | 시간대 무관 | discrete 신호 |
| `ta` (기온) | +0.13 | 미미 | 모듈 효율 |

**관측**:
- dsr 와 hm 가 *상호 보완* (corr +0.72 vs −0.54). dsr 만 보면 *광역 일사*, hm 와 결합 시 *부분 흐림 영역* 보강.
- ws 와 ta 는 *오후 시간대* (14~16시) 에서만 marginal — 모듈 온도가 누적되는 시간

#### 4.2.3 시사점 — 모델 영역 분리

**(a) 고일사 영역 (clear-sky)** — dsr 만으로 cf 거의 결정 (R² ~0.85). *Phase 1 baseline 만으로 충분* 한 영역.

**(b) 부분 흐림 영역 (dsr Q1-Q4)** — cf 분포 폭 0.18+ 로 *큰 산포*. 이 영역이 **Phase 2 event branch 가 잡아야 할 가치 영역**.

**(c) 강수 / 안개** — rn (강수) + hm (습도) 결합으로 부분 보강. 단 marginal 영역.

### 4.3 시간적 지속성 (`M2_temporal_scale.png`, `M5_residual_pattern.png`)

PV 변동이 *어떤 시간 척도* 에서 발생하는지 + 직전 시각 정보가 *얼마나 다음 시각* 예측에 도움이 되는지 분석.

#### 4.3.1 시간당 변동 분포 — fat tail

portfolio cf 의 시간당 변화 \|Δcf/h\| 분포:

| 분위 | 값 | 비고 |
|---|---:|---|
| p50 | 0.020 | 평소엔 작은 변동 |
| p75 | 0.055 | |
| p90 | 0.130 | |
| **p95** | **0.219** | *간헐적 큰 변동* |
| p99 | 0.283 | *cloud-pass / outage* 영역 |
| max | 0.5+ | 극단 사건 |

**관측**: 시간당 변동의 *p99 = 0.283* 으로 평균 (0.07) 의 *4 배*. 즉 *fat tail 분포*. *D-1 single forecast 의 분포 가정 (Gaussian σ ~0.10)* 으로 잡기 어려움.

#### 4.3.2 cf residual 자기상관 (일사 주기 제거 후)

raw cf 자기상관은 daily seasonality (24h 주기) 가 dominate → *cf − hour-of-day mean* 으로 *residual* 산출 후 자기상관:

| lag (h) | 상관 |
|---|---:|
| 1 | **0.888** ★ |
| 2 | 0.769 |
| 3 | 0.612 |
| 4 | 0.521 |
| 5 | 0.475 |
| 6 | **0.422** |
| 8 | 0.310 |
| 10 | 0.225 |
| 12 | 0.180 |

**관측**:
- *직전 1시간 cf 편차* 가 *다음 시각 cf 편차의 0.89* — 매우 강한 시간적 지속성
- lag 6h 까지 0.42 유지 — *6시간 누적* 정보가 *유의 신호*
- lag 12h+ 에서 0.2 이하 — daily 주기에 가까워짐

#### 4.3.3 cloud-pass event 지속 시간 분포

연속 *cf < daily peak × 0.3* 인 시각의 길이 (4년 174 events):

| 측정 | 값 |
|---|---:|
| event 수 | 174 |
| median 지속 | **1h** |
| p75 | 2h |
| p90 | **3h** |
| max | 5h |

→ *대부분 cloud-pass 가 1~3 시간 안에 종료*. *Phase 2 의 lead 1~3h* 가 *운영 대응 가능 영역과 정확히 일치*.

#### 4.3.4 Phase 1 residual 자기상관 (Phase 2 동기 직접 입증)

5-seed Phase 1 ensemble 의 *signed residual* (cf − μ_p1) 의 자기상관 (사이트 × 일 그룹 내):

| lag (h) | residual autocorr |
|---|---:|
| 1 | **0.763** ★ |
| 2 | 0.612 |
| 3 | 0.481 |
| 4 | 0.395 |
| 5 | 0.323 |
| 6 | **0.278** |
| 7 | 0.220 |

**관측**:
- *Phase 1 residual* 이 *random noise X, 자기상관 보유*
- *직전 actual* 만 보면 *다음 시각 residual 의 강한 예측 신호* (lag 1h corr 0.76)
- lag 6h 까지 0.28 유지 → *H=6 look-back* 이 sufficient

#### 4.3.5 시간대별 변동성

mean \|Δcf/h\| 의 시간대 분포:

| 시각 | mean \|Δcf/h\| | 비고 |
|---|---:|---|
| 9~10 | 0.04 | 일출 직후, 안정 |
| 11~13 | **0.08~0.09** | 정오 ±2h peak (구름 발생) |
| 14~15 | 0.07 | 오후 |
| 16~17 | 0.05 | 일몰 접근 |

→ *정오 시간대* 가 가장 변동 큼. Phase 2 의 운영 가치도 이 시간대에 집중.

#### 4.3.6 시사점 — 모델 구조 함의

| 관측 | 함의 |
|---|---|
| 시간당 변동 fat tail (p99 0.28) | *D-1 single forecast 만으로 잡기 어려움* → **intraday 갱신 (Phase 2) 필수** |
| cf residual lag 1h corr 0.89 | 직전 시각이 다음 시각의 매우 강한 신호 → **rolling reforecast 정당** |
| cloud-pass median 1h, p90 3h | **Phase 2 lead 1~3h** 가 운영 대응 가능 영역과 일치 |
| Phase 1 residual lag 1h corr 0.76 | Phase 2 가 H=6 actual 만 보고도 **lead 1h correction 가능** |
| residual lag 6h corr 0.28 | **H=6 look-back window** 가 sufficient |
| 정오 ±2h 변동 peak | Phase 2 의 *시간대별 가치 집중* — 정오 운영 모니터링 우선 |

### 4.4 이벤트 유형 분리 (`M4_outage_separability.png`, `D_site_outage_heatmap.png`)

cf=0 사건이 *날씨* 와 *비기상 정지* 두 종류로 *통계적으로 분리 가능* 함을 입증.

#### 4.4.1 deep zero 사건의 통계적 특성

test 2025 daytime 중 `cf < 0.03 ∧ μ_p1 > 0.20` 인 *deep zero* 사건 (n=288):

| 측정 | 값 |
|---|---|
| 전체 daytime 비율 | < 1% (희소) |
| z-score 분포 (z = (cf − μ_p1)/σ_p1) | p1=-3.5, p5=-3.0, p50=-1.7 |
| z<-3 비율 (rule 임계) | 24% |
| dsr (구름 강도) 분위 분포 | *맑은 날 (Q5+) 에도 발생* |

**z-score 분포 해석**:
- 일반 daytime z 분포는 평균 0, σ 1 의 정규분포 가까움
- deep zero 사건의 z 분포는 *deep tail (z<-3)* 에 집중 — 통계적 outlier
- σ가 *well-calibrated* 이라 z<-3 임계는 *통계적으로 유의한 deep tail* 에 정확히 해당 (§4.5 정합)

#### 4.4.2 dsr 무관성 — 비기상성 사건임을 입증

deep zero 사건의 dsr 분포:

| dsr 분위 | deep zero 비율 |
|---|---:|
| Q1 (저일사, 흐림) | 1.2% |
| Q3 (중일사) | 0.8% |
| Q5 (보통) | 0.6% |
| **Q7-Q10 (맑은 날)** | **0.4-0.7%** ★ |

**관측**: *맑은 날 (높은 dsr) 에서도 deep zero 발생*. 단순 cloud cover 로 설명 안 됨 → **비기상성 운영 사건** 임을 입증.

#### 4.4.3 사이트별 outage 패턴 (heatmap)

사이트 × day-of-year × hour 의 outage 발생 heatmap (`D_site_outage_heatmap.png`):

| 사이트 | cap (MW) | outage 시간 | 패턴 |
|---|---:|---:|---|
| 광양항세방 | 3.0 | 279h (8.5%) | *가을 (10/10-12) 3일 종일 cluster* — site shutdown |
| 예천 | 2.0 | 362h (11.0%) | 단속 운영 + 5월 peak 흐름 |
| 구미 | 1.0 | 197h (6.0%) | *6월 중 repeated outage-like* — 점검 의심 |
| 영흥 #5 | 5.5 | 160h (4.9%) | 점진적 |
| **고흥만수상** | **63.5** | 74h (2.3%) | *점심시간 반복 outage* — 정기 점검 의심 (큰 cap → portfolio 영향 큼) |
| 삼천포 | 1.3 | 137h (4.2%) | |
| 경상대 | 0.9 | 124h (3.8%) | |
| 창원 (두산MG) | 0.08 | 71h (2.2%) | |

#### 4.4.4 시사점 — outage 처리 구조

**(a) 통계 분리 가능** — z-score (Phase 1 σ 기반 통계 outlier) + dsr 분포 (구름 무관) + neighbor confirm (단발 노이즈 X) 의 *세 축* 으로 outage 분리 가능.

**(b) Rule-based detector 가 자연스러운 선택** — 사건이 *본질적으로 비기상성 site-specific 운영* 이므로 *learned head (joint training)* 보다 *rule-based* 가 적합. (§5.4 에서 정량 비교)

**(c) 운영 인사이트 부수 효과** — outage detection 으로 *광양항 10/10-12 3일 shutdown*, *고흥만 12-14시 정기 outage* 같은 *site-level 운영 패턴* 자동 검출 가능 → 모델이 *예측만이 아니라 운영 진단 도구* 로 확장 가능.

### 4.5 Uncertainty Calibration (`04_calibration_seed.png` in `_selection_results/`)

5-seed Phase 1 ensemble 의 σ 가 *well-calibrated* 임을 검증 — outage detector 의 z-score 임계값이 *통계적으로 유의* 함의 근거.

#### 4.5.1 5-seed per-seed 변동

| seed | NMAE | bias | Cov80 | Cov95 | NLL |
|---|---:|---:|---:|---:|---:|
| 42 | 4.77 | -0.21 | 83.1 | 93.2 | -1.187 |
| 7 | 4.89 | +0.28 | 86.9 | 95.3 | -1.162 |
| 123 | 4.68 | +0.64 | 79.3 | 90.7 | -1.139 |
| 202 | 4.86 | +1.53 | 81.8 | 92.9 | -1.153 |
| 999 | 4.93 | -0.17 | 78.5 | 90.4 | -1.125 |
| **per-seed 평균** | **4.83** | | **81.9** | **92.5** | -1.153 |
| **5-seed ensemble** | **4.66** | | **85.0** | **94.1** | **-1.218** |

→ ensemble 이 NMAE -0.17pp 안정화 + Cov80/Cov95 모두 목표 ±5pp 안.

#### 4.5.2 사이트별 calibration

| 사이트 | NMAE (%) | bias | Cov80 | Cov95 |
|---|---:|---:|---:|---:|
| 경상대 | 3.38 | +0.37 | 92.0 | 97.5 |
| 영흥 | 4.36 | +0.45 | 88.0 | 95.1 |
| 고흥만수상 | 4.41 | +0.12 | 89.4 | 96.7 |
| 삼천포 | 4.68 | -0.23 | 88.2 | 95.3 |
| 구미 | 5.17 | +2.02 | 83.4 | 94.3 |
| 예천 | 8.12 | -0.67 | 78.0 | 90.8 |
| **광양항세방** | 8.33 | **+7.02** | **69.9 ★** | 84.7 |
| 창원 | 8.55 | +2.22 | 90.9 | 98.4 |

**관측**:
- 대부분 사이트가 Cov80 = 80% ±5pp 안 — well-calibrated
- *광양항* 만 Cov80 70% (underconfident) → bias +7% 영향. post-hoc bias correction 후보 (§10.3 한계)
- *창원* 은 Cov80 91% (over-confident) — 작은 cap (0.08 MW) 의 noise

#### 4.5.3 시사점 — outage detector 의 calibration 기반 정당성

**(a) σ 가 well-calibrated** → *z-score = (cf − μ_p1) / σ_p1* 가 *통계적으로 의미 있는 outlier 지표*

**(b) z<-3 임계 = p0.13% deep tail** → 약 1000시간 중 1.3시간 — 본 PoC 의 deep zero 비율 (< 1%) 과 정합

**(c) Phase 2 가 σ 그대로 사용** → ensemble σ 가 이미 잘 보정되어 σ 재학습 불필요

→ outage detector 의 z-score 임계가 *통계적 deep tail* 에 정확히 위치 — *임의 임계가 아니라 calibration 기반 정량 임계*.

### 4.6 Portfolio Aggregation 관점

site-level miss 가 *어떻게 portfolio gap 으로 변환* 되는지 + *왜 portfolio 기준 평가* 가 필요한지 분석.

#### 4.6.1 NMAE 변환 — 3 단계

| 지표 | 값 | 산출 |
|---|---:|---|
| 단순 site mean NMAE | 5.87% | 8 사이트 NMAE 평균 (cap 무관) |
| cap-weighted site NMAE | **4.66%** | 사이트 NMAE × capacity 가중 |
| **portfolio NMAE** | **3.75%** | 시간별 portfolio 합산 후 NMAE |

→ *cap-weighted ≈ portfolio* — 큰 사이트 (고흥만수상 63 MW) 가 dominant.
→ *site mean → portfolio* 변환 시 *2.12pp 개선* — 평균화 효과.

#### 4.6.2 site-level outlier 의 portfolio 영향

site-level outlier 가 *capacity 기준 portfolio 에서는 영향이 작다*:

| 사이트 | NMAE | bias | cap 비중 | portfolio 영향 |
|---|---:|---:|---:|---|
| 광양항 | 8.33% | +7.02 | 3.9% | 0.27pp |
| 예천 | 8.12% | -0.67 | 2.6% | 0.18pp |
| 창원 | 8.55% | +2.22 | 0.1% | **0.01pp (무시)** |
| 고흥만수상 | 4.41% | +0.12 | **82.1%** | **3.62pp ★** |

→ *광양항 / 예천 / 창원* 같은 outlier 가 *portfolio 에서 감지되지 않음*. 운영 KPI 는 *고흥만수상 영향에 dominate*.

#### 4.6.3 site → portfolio 부정합의 위험성

cap-weighted 평균이 portfolio 에 가깝지만, *시간별로는 부정합* 발생 가능:
- 정오에 *광양항 cloud-pass + 고흥만 정상* → site-level 평균은 변동 작지만 *portfolio* 는 변동 ↑
- 반대로 *모든 사이트 cloud-pass 동시 발생* → portfolio 변동 *제곱* 증가

→ 운영 의사결정은 *portfolio 시간별 곡선* 기준 — site-level 평균 NMAE 만 보면 위험.

#### 4.6.4 시사점

**(a) 운영 KPI = portfolio 기준** — 화력 백업 의사결정은 *portfolio 시간별 부족* 으로 측정. site mean NMAE 로 평가하면 *고흥만 dominant 효과 누락*.

**(b) site-level outlier 처치 차별화** — 정직성 차원에서 *광양항 bias correction*, *예천 anomaly 처리* 는 *분리 보고*. portfolio 평균에 묻혀 *site-level 진단* 못하면 안 됨.

**(c) 사이트 추가/제거 시 영향 분석 필요** — 향후 SPC 513 MW 통합 시 portfolio 비중 재배분 → 모델 재평가 필요.

### 4.7 PV EDA 종합 — *왜 Phase 1 단독으로 부족한가*

위 6 절의 *데이터 내재적 결과* 를 *모델 구조 결정* 으로 매핑:

| § | 데이터 관찰 | 모델 함의 |
|---|---|---|
| 4.1 | 8 사이트 daily curve 이질성 (cross-corr 0.92~0.98) | *site × hour × month* 동적 conditioning 필요 → **AdaLN 채택** |
| 4.2 | 부분 흐림 영역 (dsr Q1-Q4) cf 분포 폭 0.18 (3배) | clear-sky 영역은 baseline 충분, partial cloud 영역은 추가 보정 필요 → **Phase 2 event branch** |
| 4.3 | residual lag 1h corr 0.76, 6h corr 0.28 + cloud-pass 1~3h | **intraday 갱신 (Phase 2) + H=6 look-back + lead 1~3** |
| 4.4 | deep zero 가 z<-3 deep tail 집중 + dsr 무관 | **rule-based outage override** (learned head 부적합) |
| 4.5 | σ well-calibrated (Cov80 85%) | z-score 임계의 *통계적 정당성* — outage detector 신뢰 |
| 4.6 | portfolio 가 site mean 보다 안정 (3.75% vs 5.87%) | 운영 KPI = portfolio 기준 / site-level 처치 차별화 |

**핵심 결론**: Phase 1 단독으로는 (사이트 이질성 + 부분 흐림 산포 + outage-like event + 시간 지속성) *모두를 동시에 잡을 수 없다*. 이 EDA 결과들이 **Phase 2 (intraday reforecast) + Outage Override (rule-based) + portfolio 평가 의 도입 동기** 를 *데이터 자체로* 정당화한다.

구체적 모델 구조 / hyperparameter sweep / 비채택 대안은 §5 에서 다룬다.

---

## 5. PV 예측모델 설계 및 채택 근거

### 5.1 최종 구조

```
[D-1 17:00] Phase 1: ResMLP+AdaLN v2 ensemble (5-seed, frozen)
              inputs: ASOS + GK-2A + zenith + site/hour/month
              outputs: 24h × 8 site (μ_p1, σ_p1)
                ↓
[D-day, t ∈ {7..16}] Phase 2: 2-branch TCN residual correction
              base branch + event branch + soft gate (λ=2.0, T=1, τ=0)
              inputs: 최근 H=6h actual + frozen Phase 1 + future weather + gate signals
              outputs: μ_p2(t+1..일몰) — remaining-day reforecast (L=12 EOD truncation)
                ↓
[Override] rule-based outage detection
              cf<0.03 ∧ μ_p1>0.20 ∧ dc<7 ∧ z<-3 ∧ neighbor_confirmed
              recovery: actual_cf>0.10 for 1h
              → mu_phase2 := 0 if outage detected
```

각 layer 가 서로 다른 책임을 가진다:
- Phase 1 = *기준 계획선* (frozen anchor)
- Phase 2 = *최신 actual + 남은 하루 reforecast*
- Override = *비기상성 정지 처리*

### 5.2 Backbone 모델 비교 — ResMLP+AdaLN v2 ensemble 채택

`pv/experiments/` 60+ 실험 중 동일 평가 protocol (test 2025, 32,443 rows) 비교:

| 모델 | NMAE | bias | 비고 |
|---|---:|---:|---|
| LSTM baseline | 7.26% | +0.01 | 시계열 RNN |
| NGBoost baseline | 6.33% | -0.25 | tree + uncertainty |
| ResMLP+AdaLN (single) | 6.13% | -0.11 | tabular + AdaLN |
| FT-Transformer | 6.11% | +0.62 | tabular transformer |
| **ResMLP+AdaLN v2 ensemble (5-seed) ★** | **5.88%** | +1.42 | 채택 |

(주: model_final §1.2 의 4.66% 는 clean 평가 기준)

**채택 이유**:
1. 5종 중 NMAE 최우수 (LSTM 대비 −1.38pp)
2. **ensemble 구조가 σ_total 자연 제공** → Phase 2 의 frozen anchor + outage detector 의 z-score
3. AdaLN conditioning 으로 site/hour/month 효과 흡수 (§4.1 사이트 이질성 EDA 정합)
4. tabular 입력 (ASOS + GK-2A + static) 에 적합
5. **Phase 2 residual correction 을 붙이기 구조적으로 적합** — 가장 결정적

### 5.3 Phase 2 architecture sweep

#### 5.3.1 H (look-back window) — H=6 채택

3-seed × H={3, 4, 5, 6}:

| H | Overall | Lead 1h | 03-23 cloud-pass | 04-26 | per-seed std |
|---|---:|---:|---:|---:|---:|
| 3 | 4.717 | 4.239 | 45.31 | 34.75 | 0.020 |
| 4 | 4.715 | 4.237 | 44.39 | 34.51 | 0.004 |
| 5 | 4.715 | 4.242 | 42.93 | 34.41 | 0.013 |
| **6 ★** | 4.729 | 4.256 | **41.98** | **34.31** | 0.049 |

→ Overall 차이 0.01~0.02pp (사실상 동률) 이지만 **top cloud-pass event 에서 H=6 가장 우수** → 채택. residual autocorr (§4.3 EDA) 가 lag 6h 까지 0.4+ 유지하는 결과와 정합.

#### 5.3.2 2-branch λ — λ=2.0 채택

H=6 single-branch 대비:

| 비교 | NMAE | Lead 1h | per-seed std |
|---|---:|---:|---:|
| H=6 single-branch | 4.729 | 4.256 | 0.049 |
| 2-branch λ=1.0 | 4.738 | 4.271 | 0.023 |
| 2-branch λ=1.5 | 4.742 | 4.261 | 0.014 |
| **2-branch λ=2.0 ★** | **4.712** | **4.209** | **0.026** |

→ λ=2.0 NMAE 우세 + **per-seed std −47%**. gate top-event 활성화 1.9x (max 0.95).

#### 5.3.3 Sparse gate — 폐기 (negative result)

| config | gate ratio | dead seeds | NMAE | top 03-23 |
|---|---:|---:|---:|---:|
| **T=1.0 τ=0 (★ 최종)** | 1.9x | 0/3 | 4.712 | 42.12 |
| T=0.5 τ=0 | 5.8x | 0/3 | 4.743 | 43.77 |
| T=0.5 τ=0.1 | 16.8x | **2/3** | 4.732 | 42.56 |
| T=0.5 τ=0.2 | 0x | 3/3 | 4.729 | 41.98 |

→ Selectivity 향상 시 vanishing gradient 로 dead seed 폭증 → **운영 PoC 안정성 부적합 → 폐기**.

#### 5.3.4 Output horizon L — L=12 EOD truncation 채택

| L | 특징 |
|---|---|
| L=3 | lead-1 즉시 보정 강하지만 *남은 하루 shape* 정보 X |
| L=6 | L=12 에 dominate |
| **L=12 EOD** ★ | lead-1~3 정확도 보존 + recovery / persistence / sunset shape 추가 |

→ Phase 2 가 *short patch generator* 에서 **intraday remaining-day reforecast module** 로 해석 전환.

### 5.4 Outage 처리 — 3단계 negative result + rule-based 채택

#### 5.4.1 시도 #1: Phase 1 outage-masked retrain — 폐기

| variant | mask 수 | NMAE | baseline 대비 |
|---|---:|---:|---:|
| baseline | 0 | 4.747% | — |
| narrow | 278 | 4.89% | **+0.14 pp** |
| wide | 539 | 4.91% | **+0.16 pp** |

원인: 0.2~0.4% 마스킹으로 분포 변화 미미. 모델은 outage 를 noise 로 robust 학습.

#### 5.4.2 시도 #2: Integrated outage probability head — 폐기

- Phase 2 architecture 에 outage_prob head + joint loss
- 결과: precision 3.3%, recall 41.7% (FP 폭증). 04-26 cloud-pass 13.79% → **20.24% (악화)**
- 폐기 이유: label imbalance (positive 0.21%), outage 가 weather feature 로 예측 불가능 (site-specific 운영 사건), 학습 sample 부족 (437 positive)

#### 5.4.3 채택: Rule-based outage override

```
outage_flag = (cf<0.03 ∧ μ_p1>0.20 ∧ dc10Tca<7 ∧ z<-3 ∧ neighbor_confirmed)
recovery: actual_cf>0.10 for 1h → NORMAL
override: μ_phase2 := 0 if BLACKOUT
```

§4.4 EDA 의 결과 (z-score deep tail + dsr 무관 분포) 와 정합. test 2025 결과:

| 그룹 | n | P1 | P2 | **P2+override** | Δ |
|---|---:|---:|---:|---:|---:|
| 전체 | 167,953 | 4.747% | 4.432% | **4.220%** | −0.21 pp |
| Blackout | 587 | 68.6% | 55.7% | **0.39%** | −55.3 pp |
| Normal | 167,366 | 4.50% | 4.23% | **4.23%** | **+0.000 pp ✓** |
| 03-23 cloud-pass | 440 | 30.0% | 23.3% | **2.89%** | −20.4 pp |
| 광양항 10/10-12 | 165 | 28.2% | 15.5% | **0.09%** | −15.4 pp |

→ Normal 손상 0pp + Blackout 극적 개선 + cloud-pass 개선 *동시 만족*.

### 5.5 최종 spec 요약

| 항목 | 값 | 근거 |
|---|---|---|
| Backbone | ResMLP+AdaLN v2 ensemble (5-seed) | 5.2 |
| H (look-back) | 6 시간 | 5.3.1 + §4.3 EDA |
| Phase 2 구조 | 2-branch + soft gate | 5.3.2 |
| λ (event branch) | 2.0 | 5.3.2 |
| gate (T, τ) | (1.0, 0) | 5.3.3 |
| Output horizon L | 12 EOD truncation | 5.3.4 |
| Outage 처리 | rule-based override | 5.4 |

---

## 6. LNG 데이터 EDA

> 본 절은 **데이터의 성질만** 기록. planner 설계는 §7.
> 위치: `pv/eda_lng_planner/` (8 그림 + 8 csv).

### 6.1 호기별 운전 패턴 (`H_unit_pattern.png`)

분당화력 LNG 10 호기 (CG1~8 가스터빈, CS1~2 증기터빈) 의 4년 hourly 운전 패턴:

| 호기 | mode | 운전율 (%) | 평균 연속 운전 (h) | 최대 연속 (h) |
|---|---|---:|---:|---:|
| CS2 | baseload | 51.9 | 33.5 | 1,399 |
| CG6 | baseload | 47.6 | 32.1 | 1,455 |
| CS1 | mid-merit | 41.9 | 38.2 | 2,444 |
| CG1, CG3, CG7, CG8, CG5 | mid-merit | 28~44 | 24~34 | 700~2,200 |
| CG2 | peaker | 23.0 | **19.3** | 539 |
| CG4 | peaker | 19.8 | **19.7** | 991 |

**관측**:
- peaker (CG2/CG4) 도 한 번 켜지면 **평균 6~7h 연속 운전** (단발 spike 가정 X)
- mode 차이는 *운전 빈도* 차이일 뿐 *연속 길이* 차이 X

**시사점**: 호기 선택 룰에서 mode 우선순위 X, *헤드룸 + ramp 기반* 정렬이 더 정직.

### 6.2 호기 spec 분포 (data-driven, 4년 hourly)

| 호기 | type | P_min | P_max | 헤드룸_p90 | 평균 출력 | 운전율 |
|---|---|---:|---:|---:|---:|---:|
| CS1 | ST | 43 | 142 | **67** ★ | 75 | 41.9% |
| CS2 | ST | 44 | 107 | **55** ★ | 51 | 51.9% |
| CG7 | GT | 49 | 90 | 10 | 76 | 35.8% |
| CG6 | GT | 52 | 87 | 9 | 76 | 47.6% |
| CG2, CG4 | GT (peaker) | 37 | 89 | 8~10 | 74~75 | 19.8~23.0% |
| CG1, CG3, CG5, CG8 | GT | 44~52 | 87~90 | 6~9 | 75~78 | 28~44% |

**시사점**: ST 두 호기 (CS1, CS2) 의 헤드룸이 다른 GT 호기 대비 **6~10배** 큼 → portfolio PV balancing 의 절대적 핵심.

### 6.3 호기 ramp 비대칭 (`B_ramp_asymmetry.png`)

| 호기 | mode | 상승 한도 p90 | 하강 한도 p90 | 비대칭 비율 |
|---|---|---:|---:|---:|
| CG6 | baseload | 18.2 | **1.87** | **9.7x** |
| CG8 | mid-merit | 19.0 | 2.39 | 8.0x |
| CG7 | mid-merit | 25.7 | 3.19 | 8.1x |
| CS2 | baseload | 18.5 | 2.74 | 6.8x |
| CS1 | mid-merit | 29.3 | 7.42 | 4.0x |
| CG2 | peaker | 28.9 | 11.18 | 2.6x |

**관측**: baseload 호기는 *상승은 빠르고 하강은 매우 느림* — 한 번 올라간 출력이 회복 후에도 천천히 내려감.

**시사점**:
- 잔존 over-commit 의 *메인 원인* = ramp_dn 비대칭 (모델 인위 X, 실제 운영 패턴)
- 향후 옵션: ramp_dn p99 사용 시 추가 감축 가능 (단 운영 직관 약화)

### 6.4 GT vs ST 차이

| 종류 | 호기 | 기동 시간 (추정) | Layer B 후보 |
|---|---|---|---|
| **가스터빈 (GT)** | CG1~CG8 | ~30분 (warm) ~ 1시간 (cold) | ✓ |
| **증기터빈 (ST)** | CS1, CS2 | 1~3시간 | X (1h lead time 부족) |

**시사점**:
- *1시간 lead time* 의 forward 신호로 예열 명령 가능 = GT 만
- ST 는 baseline online 시점에서 활용 (큰 헤드룸으로 분배 dominant)

### 6.5 Gap 분포 (`C_forward_gap_distribution.png`)

forward gap 분포 (test 2025 portfolio, lead=1, daytime 8-17, n=3,302):

| 분위 | \|FG1\| | \|slope\| |
|---|---:|---:|
| p50 | 1.31 | 0.65 |
| p75 | 2.54 | 1.15 |
| p80 | 2.94 | 1.29 |
| p90 | 4.23 | 1.80 |
| p95 | **5.98** | **2.40** |
| p99 | 15.61 | 15.73 |

**시사점**:
- TH_INCREASE=5 ≈ p95 → 상위 5% 만 *큰 신호* 분류
- TH_NOISE=3 ≈ p80 → 잡음 영역 컷
- TH_SLOPE=2 ≈ \|slope\| p95 → 명확한 변화 추세
- 시간대별 mean \|FG1\| peak = 12-14시 (정오 PV 부담 시간대)

### 6.6 자체 흡수폭 (DEADBAND=4 MW) 근거 (`A_deadband_curve.png`)

DEADBAND ∈ {0, 2, 4, 6, 8} sweep:

| DEADBAND | 부족분 | 과대보충 | 비율 | 총 추가 발전 |
|---:|---:|---:|---:|---:|
| 0 | 1,197 | 770 | 0.64 | 5,670 |
| 2 | 699 | 700 | 1.00 | 3,485 |
| **4 ★** | **441** | **640** | **1.45** | **2,448** |
| 6 | 303 | 583 | 1.92 | 1,897 |
| 8 | 224 | 535 | 2.39 | 1,602 |

**근거 두 축**:
1. **외부 데이터** — 한국전력거래소 보조서비스 정산금 분석 결과 LNG 가 *전국 보조서비스 (운영예비력) 정산의 ≈50% 담당*
2. **내부 sweep** — DEADBAND=4 에서 부족·과대 비율 1.45x 균형 plateau

→ *외부 정산 근거 + 내부 sweep 결과가 4 MW 영역에서 일치* → 데이터 기반 자체 흡수폭으로 채택.

### 6.7 over-commit 의 운영 의미

LNG planner 의 1순위 평가 KPI 가 *부족분 (shortfall)* 이 아니라 **과대보충 (over-commit)** 인 이유:

- *부족 메우는* 단순 응답이라면 항상 LNG 출력 최대화하면 됨 — 그러나 운영비 낭비
- 한 번 올라간 출력이 baseload 호기 ramp_dn 한계 때문에 천천히 내려감 → 회복 후 잔류 ΔP = over-commit
- *작은 변동까지 LNG 가 따라가는* 단순 응답 = 불필요한 과대 출력 누적

**시사점**: planner 의 진짜 운영 가치는 *부족 충족* 이 아니라 **작은 변동에 과잉 반응하지 않으면서 큰 변동만 적극 대응** 하는 것. DEADBAND + Layer A/B 분리 설계의 동기.

---

## 7. LNG 백업 계획 및 재배분 로직

> 위치: `plan/active/lng/plan.md` (SSOT, 22 sections).

### 7.1 planner 의 위치 / 범위 / 가정

본 PoC 의 planner 는 다음을 *수행하지 않는다*:
- full unit commitment optimizer
- 전국 계통 경제급전기
- 실제 dispatch 재현기

본 PoC 의 planner 는 다음을 *수행한다*:
- PV 기준선과 실 발전 사이 gap 해석
- 그 gap 을 LNG 추가 대응량으로 변환
- 그 대응량을 baseline LNG 운전 상태 위에 호기별 재배분

즉 **예측 결과를 thermal action 으로 번역하는 운영 레이어** 이다.

### 7.2 baseline thermal schedule — `P_DA,u(t)` proxy

```
P_DA,u(t) = 2025년 시점 t의 호기 u 실측 LNG 발전량
```

이는 *실제 dispatch 정답* 이 아니라 **그날의 기준 운전 상태** 로 해석한다. planner 가 계산하는 것은:

```
P_new,u(t) = P_DA,u(t) + ΔP_u(t)
```

baseline 자체는 새로 짜지 않는다.

### 7.3 effective_gap — DEADBAND 차감 핵심 입력

```
instant_gap_t   = μ_p1_t − pv_actual_t
effective_gap_t = max(0, instant_gap_t − DEADBAND_MW)     # DEADBAND=4
즉시_보정필요량 = effective_gap_t                          # Layer A 입력
```

작은 편차는 계통 자체 흡수, LNG 는 그 marginal 만 책임. §6.6 EDA 가 정당화.

### 7.4 Layer A — 실 발전 명령

```
즉시_보정필요량 = max(0, instant_gap_now − 4)

흐름:
  1. online 호기 합 헤드룸 ≥ 즉시_보정필요량  → online 분배만
  2. cap 초과 + P_min ≤ 잔여 ≤ P_max  → GT 우선 신규 가동, 출력 = 잔여 그대로
  3. 그 외  → shortfall 감수
```

분배 가중치 (online 호기들끼리):

```
w_raw,u  = Headroom_u(t) × Ramp_up,u
w_norm,u = w_raw,u / Σ_v w_raw,v
ΔP_actual,u(t) = min(w_norm,u × 즉시_보정필요량, Headroom_u(t))
```

**핵심 변경 (이전 버전 대비)**:
- *priority 제곱 가중 제거* — priority 는 §6.3 ranking 표용으로만 (분배 가중 X)
- *RG (look-back 평균) 폐기* — `instant_gap_now` 만 사용 (over-commit 잔재 제거)

### 7.5 Layer B — 사전 대비 (warm-up only)

forward signal 강신호 시 *호기 예열 명령* 만 발행, **출력 X**.

```
WARMUP_MIN_GAP = 5

trigger:
  expected_next_gap = max(즉시_보정필요량, FG1)
  trigger if  expected_next_gap > max(WARMUP_MIN_GAP, online_headroom_total)
              AND forward_persistent

candidate filter (★ GT only):
  is_online == 0  AND  Avail > 0.1  AND  unit_type == 'GT'  AND  P_max ≥ 5

선택 정렬:
  1. 헤드룸 큰 순  →  2. P_min 작은 순  →  3. Avail 높은 순

효과:
  warm_up_set.add(chosen_unit)        # 다음 시각 online subset 에 합류
  ΔP_u(t) (chosen) = 0                 # 지금 시각 출력 X
  P_DA,u(t+1) = 0 유지                 # P_min 강제 X (다음 시각 actual gap 따라 결정)
```

### 7.6 GT 우선 (cold-start 차이)

§6.4 EDA 결과:
- GT cold-start ~30분 → 1시간 lead time 안에 가용
- ST cold-start 1~3시간 → 1h lead time 부족

→ **Layer A startup 시 GT 우선** + **Layer B warm-up 은 GT 전용**.

### 7.7 운영 모드 (state machine)

| state | 의미 | Layer B 행동 |
|---|---|---|
| **유지 (KEEP)** | 평소, 잡음 | Layer B 비활성 |
| **추가 대응 (INCREASE)** | forward 강신호 (\|FG1\|≥5 또는 slope≥2) | 호기 추가 startup 검토 |
| **점진 해제 (DELAYED_RELEASE)** | forward 회복 신호 | 켜놓은 호기 점진 release (8 MW/h) |

state 진입 조건은 §6.5 forward gap EDA 의 분위수 기반 (TH_INCREASE=5, TH_SLOPE=2).

### 7.8 v2 (단일 호기) vs v4 (10 호기 fleet) 비교

| 측면 | v2 (CS2 single) | v4 (10 호기 fleet) ★ |
|---|---|---|
| Controllable | CS2 단일 블록 | 10 호기 (GT 8 + ST 2) |
| 가정 | 큰 online headroom | 실제 2025 baseline loading 반영 |
| 결과 | idealized upper-capacity | realistic fleet-constrained |
| shortfall | 4,366 MWh | **442 MWh (−89.9%)** |

→ v4 = "나쁜 결과" 가 아니라 *현실 제약을 드러낸 더 정직한 결과*.

### 7.9 startup 분리 — `startup_real` vs `startup_ramp`

| 구분 | Phase 1 단독 | Phase 1+2 | 의미 |
|---|---:|---:|---|
| **신규 가동** (offline → 발전, cold-start 비용) | 1 회 | **0 회** | 진짜 신규 가동 |
| **추가 발전 전환** (이미 운전 중 + ΔP 시작) | 931 | 941 | cold-start 비용 X |

**시사점**:
- 진짜 신규 가동 1년 0~1회 → cold-start 비용 (~5천만원/회) 거의 회피
- LNG 호기 대부분 baseline 운전 중 → cost-aware screening 데이터 부재 상태에서도 자연스럽게 비용 효율 운영

### 7.10 잔존 한계

planner 가 직접 해결하지 않는 것:
1. full startup cost optimization
2. cold start trajectory 최적화
3. SMP 포함 순이익 최적화
4. 계통 전체 제약

**잔존 over-commit 640 MWh 의 메인 원인** = 호기 자체의 ramp_dn 비대칭 (§6.3). 향후 옵션 A (ramp_dn p99 사용) 시 추가 감축 가능 (§11).

---

## 8. 통합 운영 구조 및 대시보드

### 8.1 전체 시스템 흐름

```
[D-1] Phase 1 (5-seed ensemble) → μ_p1 (24h baseline)
        ↓
        LNG D-1 계획선 P_DA_u (실측 baseline proxy)

[D-day, 매 1시간] Phase 2 (3-seed) → μ_p2(t+1..일몰) → outage override
        ↓
   ┌── Planner v4 ────────────────────────────────────────────┐
   │ instant_gap   = μ_p1 − pv_actual                          │
   │ effective_gap = max(0, instant_gap − 4 MW)                │
   │ FG1, slope    = μ_p1(t+k) − μ_p2(t+k)                     │
   │                                                            │
   │ Layer A: online 호기 분배 + GT 우선 startup               │
   │ Layer B: GT-only warm-up (P1+2 모드만)                    │
   │ State: KEEP / INCREASE / DELAYED_RELEASE                  │
   └────────────────────────────────────────────────────────────┘
        ↓
   호기별 ΔP_u(t) + 예열 명령 + shortfall / over-commit
```

### 8.2 4 페이지 대시보드 (Streamlit)

| 페이지 | 보여주는 것 | 답하는 질문 |
|---|---|---|
| **🏠 Home** | PoC scope + 운영 흐름 + 최종 KPI 8-card | 전체 한 눈 |
| **📊 예측 비교** | Phase 1 / Phase 2 reissued / 실측 PV (issue 시각별) | Phase 2가 D-1 예측을 어떻게 보정? |
| **📉 실시간 차이** | instant_gap + DEADBAND band + effective_gap + forward_gap + 운영 모드 | 어디까지가 LNG 책임? |
| **⚙ 호기별 백업** | 10 호기 stacked area + Layer A/B 이벤트 + state machine + 일별/전체 모드 토글 | 어떤 호기가 얼마나 책임? 언제 예열? |
| **📈 Phase 2 가치** | P1 단독 vs P1+2 비교 + 월별 KPI + 예열 분포 + 큰 변동일 | intraday 갱신의 정량 가치? |

상세는 `plan/active/pv/dashboard_plan.md`.

---

## 9. 성능 결과 및 기대효과

### 9.1 PV 예측 성능 (test 2025)

| 지표 | Phase 1 | Phase 2 | **Phase 2 + Override** | 비고 |
|---|---:|---:|---:|---|
| 전체 NMAE | 4.747% | 4.432% | **4.220%** | -0.21pp |
| Blackout 시점 | 68.6% | 55.7% | **0.39%** | -55.3pp |
| Normal 시점 | 4.50% | 4.23% | **4.23%** | +0.000pp ✓ |
| 03-23 cloud-pass | 30.0% | 23.3% | **2.89%** | -20.4pp |
| 04-26 cloud-pass | 15.0% | 13.7% | **3.81%** | -9.8pp |
| 광양항 10/10-12 | 28.2% | 15.5% | **0.09%** | -15.4pp |

핵심: **Normal 손상 0pp + Blackout 극적 개선 + cloud-pass 개선 동시 만족**.

### 9.2 LNG planner 성능 (test 2025, daytime 9-17, 365일 portfolio)

| 지표 | 값 | 비고 |
|---|---:|---|
| 부족분 (shortfall) | **442 MWh** | DEADBAND 적용 후 |
| 과대보충 (over-commit) | **640 MWh** | 1순위 KPI |
| 비율 (over/short) | **1.45x** | 균형 |
| 총 추가 발전 | 2,448 MWh | 시간당 평균 0.75 MW |
| 예열 명령 | 20회 (전부 GT) | Layer B |
| 신규 가동 (cold-start) | 0회 | 운영비 절약 |
| 추가 발전 전환 | 941회 | 이미 운전 중 호기 활용 |
| **vs v2 baseline** | **−89.9%** | shortfall 감소 |
| **Phase 2 marginal value** | **−19.1%** | Phase 1 단독 vs P1+2 |

### 9.3 운영 인사이트 (부수 발견)

본 과제는 예측 / 운영 외에도 다음 *site-level 운영 패턴* 을 데이터로 드러냈다:

- 광양항세방 10/10-12 — 3일 종일 site shutdown (16건) 자동 검출
- 고흥만수상 12-14시 정기 outage 패턴 (5 events, 13건) — 점검/정비 의심
- 구미 06-15/16 — 2일 연속 outage 새 발견 (5건)

→ 모델은 예측만이 아니라 **운영 데이터의 이상 패턴을 드러내는 보조 도구** 로도 활용 가능.

### 9.4 정성적 기대효과

1. 예측·운영 분리 X → **end-to-end 운영 지원 구조**
2. weather-driven event vs outage-like event 분리 → 운영 인사이트
3. 단일 호기 X → 10 호기 fleet 기반 → 현실 제약 반영
4. 운영자 시각화 (대시보드 4 페이지)
5. 계통 자체 흡수 vs LNG 직접 대응 *명시적 구분*

---

## 10. 비채택 대안과 한계

### 10.1 비채택 대안 (negative results)

| 시도 | 결과 | 폐기 이유 |
|---|---|---|
| Phase 1 outage-masked retrain | NMAE +0.14~0.16pp 악화 | 마스킹 0.2~0.4% 로 분포 변화 미미 |
| Integrated outage probability head | precision 3.3%, FP 폭증, cloud-pass 악화 | label imbalance + sample 부족 |
| Sparse gate (T<1, τ>0) | 2~3/3 seeds dead | vanishing gradient |
| 단일 호기 (CS2) planner v2 | shortfall 4,366 (v4 442 대비 10배) | 현실 fleet 제약 미반영 |

### 10.2 데이터 측 한계

1. **기상 입력 = perfect-foresight proxy** — 실제 archived forecast 가 아닌 ASOS·GK-2A 실측 (§2.5)
2. **LNG baseline = 2025 actual proxy** — 실제 D-1 thermal schedule 미보유 (§7.2)
3. **SPC 513 MW + 호기별 SU·VC 데이터 부재** — PoC 범위 밖

### 10.3 모델 측 한계

1. **광양항 bias +7%** — post-hoc correction 후보 (미적용)
2. **창원 cap 0.077 MW** — 학습 noise 기여 가능
3. **Phase 2 lead 3h+ 수렴** — recovery shape 만 의미
4. **Outage rule 의존** — rule 못 맞추는 outage type (예: 부분 capacity loss) 은 학습형 detector 필요 (현재 데이터 부족)

### 10.4 planner 측 한계

1. **잔존 over-commit 640 MWh** — baseload 호기 ramp_dn 비대칭 (§6.3)
2. **DEADBAND 가정의 운영 정밀도** — PoC 수준 고정값 4 MW. 실 계통 규칙과 1:1 동일하다고 주장 X
3. **offline startup screening 미구현** — 다음 단계 과제 (cost-aware UC)
4. **multi-step ramp 최적화 미구현** — 1시간 단위 단순 ramp 한도

---

## 11. 향후 확장

### 11.1 단기 확장 (실 데이터 연동 시 즉시 가능)

| 확장 | 현재 상태 | 연동 시 효과 |
|---|---|---|
| 실제 D-1 archived forecast | ASOS·GK-2A proxy | NWP forecast 오차 추가 → 진짜 D-1 NMAE 측정 |
| 실제 D-1 thermal plan | 2025 actual proxy | dispatch reproduction 평가 가능 |
| 호기별 startup cost / variable cost | 데이터 부재 | cost-aware screening + UC 통합 |
| SPC 513 MW 통합 | 미포함 | 전국 portfolio 확장 (~600 MW) |

### 11.2 중기 확장 (구조적 발전)

1. **MOS layer** — Open-Meteo / KMA NWP forecast → site/hour/month 조건부 bias 보정
2. **multi-hour Phase 2** — ST cold-start 1~3h lead 시 ST 도 warm-up 후보로
3. **반응형 portfolio rebalancing** — PV 사이트 추가/제거 자동 적응
4. **shadow operation PoC** — 실제 운영 환경 병행 + 실시간 평가

### 11.3 장기 확장 (시장 환경 변화 대응)

1. **PBP (실시간시장) 전환 대응** — 2026 말 전국 확대 예정. 이중정산 환경에서 예측 오차 = 실시간 SMP 차이 손실
2. **SMP 포함 순이익 최적화** — 시장 신호 추가
3. **계통 전체 제약 통합** — 광역 dispatch 시뮬레이션
4. **실시간 입찰 전략 자동화** — D-1 10시 / 17시 KPX 신고 자동화

---

## 결론

본 과제는 *단일 예측모델의 정확도 경쟁* 이 아니라:

- **Phase 1** 이 D-1 기준선
- **Phase 2** 가 당일 갱신 + outage override
- **LNG Planner** 가 effective_gap (DEADBAND 차감) 만 LNG 책임분으로 보고 Layer A (실 발전) / Layer B (예열만) 분리
- **Dashboard** 가 4 페이지 흐름으로 시연

으로 이어지는 **운영 지원형 예측-의사결정 통합 PoC** 를 제시한다.

본 과제의 차별점:
1. *예측 → 운영* 단일 흐름 (분리 X)
2. weather event vs outage event *명시적 분리*
3. 호기 fleet 기반 (단일 호기 X) — 현실 제약 반영
4. *작은 변동까지 LNG 가 따라가지 않는다* 는 운영 철학 (DEADBAND + over-commit 1순위)
5. *공공 데이터만으로* 도 운영 지원 구조 PoC 가 가능함을 입증

공모전 요구사항 (AI 발전량 예측 + 화력 부하 배분) 을 분리된 두 산출물이 아니라 **하나의 end-to-end 운영 지원 서비스** 로 제시하는 데 의미가 있다.
