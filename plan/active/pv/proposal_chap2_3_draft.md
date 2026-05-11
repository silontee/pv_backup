# [NEW] §2 데이터 구성과 제약 / §3 변수 선정 (draft)

> 본 문서는 proposal 새 구조의 §2, §3 본문 초안이다. user 검토 후 `proposal_outline.md` 에 통합.
> 출처 자료: `plan/active/main/data_strategy.md` (SSOT), `pv/eda_pv_model/D1_data_constraints.png`, `D2_variable_selection.png`.

---

## 2. 데이터 구성과 제약

### 2.1 원천 데이터 — 4년 hourly 공공·실측 자료

본 과제는 **공공 데이터와 공개 실적만으로 운영 지원 PoC 를 구축할 수 있는지** 를 검증한다. 따라서 *내부 운영 데이터 (SPC 발전소, 호기별 운전 비용, D-1 화력 계획) 는 일체 사용하지 않고*, 다음 4종 외부 데이터를 결합하여 학습·평가에 활용한다.

| 구분 | 소스 | 해상도 | 기간 | 용도 |
|---|---|---|---|---|
| **태양광 발전 실적** | 남동발전 홈페이지 (시간별, 사이트·호기별) | 1시간 | 2022-01 ~ 2025-12 (48개월) | 예측 모델 타깃 |
| **LNG 발전 실적** | 남동발전 홈페이지 (분당화력 10 호기) | 1시간 | 2022-01 ~ 2025-12 | LNG planner baseline + 호기 spec 추정 |
| **GK-2A 위성 일사** | 기상청 국가기후데이터센터 (NCDC) SFTP, DSR/ASR/RSR | **10분** (LCC 2km 격자) | 2022-01 ~ 2025-12 | 위성 일사·반사·흡수 |
| **ASOS 지상 기상** | 기상청 11개소 (기온/습도/풍속/강수) | 1시간 | 2022-01 ~ 2025-12 | 지상 기상 측정 |

추가 참고자료:
- `data/info/한국남동발전㈜_발전소 운영 현황_20250728.csv` — 호기별 공식 capacity
- `data/info/한국남동발전㈜_전력거래 입찰 정보 현황_20241231.csv` — KPX 입찰 시점별 SMP 결정 발전기
- `data/info/한국남동발전㈜_(발전공기업 표준) 신재생에너지 사업현황_20221231.csv` — 사이트별 메타 (위치·용량)

### 2.2 PoC 의 명시적 제약

남동발전 측 사전 질의응답 (2026-04-21) 에서 확인된 *외생적 제약* 은 다음과 같다.

| 제약 | 내용 | 본 PoC 의 대응 |
|---|---|---|
| **SPC 태양광 513 MW 데이터 제공 불가** | 별도 법인 (특수목적법인) 의 운영 데이터 미공개 | 자체 설비 22 호기로 학습 한정 |
| **호기별 운전 파라미터 미공개** | 가동 비용 / 기동 비용 / 변동비 / 기동 시간 등 | 4년 hourly 데이터로 *역추정* (P_min, P_max, ramp 등 §6) |
| **D-1 화력 계획 미제공** | KPX 신고 시점의 thermal schedule 부재 | 2025 실측 LNG 를 *baseline proxy* 로 사용 (§4.5) |
| **D-1 기상 archived forecast 미보유** | KMA LDAPS forecast endpoint 별도 신청 + TIGGE 50km 부적합 | ASOS·GK-2A 실측을 **perfect-foresight proxy** 로 사용 |

이 제약은 *PoC 범위* 자체이자 *향후 확장 시 보강 대상* 이다.

### 2.3 태양광 호기 필터링 — 22 → 11

PV 예측 모델이 학습 가능하려면 입력 신호가 *순수 태양광 발전량* 이어야 한다. 다음 두 조건이 이 전제를 무너뜨린다.

**(a) 시간별 데이터 미제공** — 일합계만 공개되어 시간별 학습 불가.
**(b) ESS 혼재** — 낮 발전을 *오후·저녁* 에 방출하여 *피크가 16~18시* 로 이동.

자연 PV 의 피크는 정오 ±2 시간 (10~14시) 이 물리적 정상 영역이다. 따라서 본 과제는 *4년 hourly 실측 데이터에서 호기별 평균 시간 프로필* 을 산출하여 다음 단계로 필터링하였다 (`pv/eda_pv_model/D1_data_constraints.png`).

| 단계 | 기준 | 투입 | 산출 | 제외 호기 |
|---|---|---:|---:|---|
| 원자료 | 자체 설비 전체 | 22 | 22 | — |
| Step 1 | 시간별 데이터 제공 | 22 | 20 | 여수, 탑선 |
| Step 2 | 피크 시각 ≤ 15시 (Pure PV) | 20 | 12 | 삼천포 #4·#5·#6, 영동, 영흥 #3 (3 호기) — 전부 16~18시 피크 |
| Step 3 | 가동 축소·중단 제외 | 12 | **11** | 삼천포 #1 (2025-04 부터 거의 정지) |

ESS 왜곡 호기의 시간 프로필은 *맑은 정오에 발전량이 0 이고 17시에 peak* 인 *역전된 패턴* 이다. 이 데이터로 모델을 학습하면 "태양광 예측 모델" 이 아니라 *ESS 운영 패턴 추정 모델* 이 되며, 본 PoC 의 목적과 직접 충돌한다.

**최종 11 호기 (합계 ~77 MW)** — `data_strategy.md §2.5` 참조.

| # | 사이트 | 용량 (kW) | 비중 | 비고 |
|---:|---|---:|---:|---|
| 1 | **고흥만 수상태양광** | **63,481** | **82.1%** | ★ portfolio dominant. 수상 PV 특이성 (낮은 알베도) |
| 2 | 영흥태양광#5 | 3,500 | 4.5% | |
| 3 | 광양항세방태양광 | 2,993 | 3.9% | 항만 microclimate, bias +7% |
| 4 | 예천태양광 | 2,000 | 2.6% | 영농형, anomaly_zero 4.95% |
| 5 | 삼천포태양광 #2 | 990 | 1.3% | |
| 6 | 영흥태양광 #2 | 993 | 1.3% | |
| 7 | 영흥태양광 #1 | 1,000 | 1.3% | |
| 8 | 구미태양광 (정수장) | 992 | 1.3% | ASOS 구미 icsr 결측 → GK-2A 필수 |
| 9 | 경상대태양광 | 905 | 1.2% | 학교 옥상 |
| 10 | 삼천포태양광 #3 | 350 | 0.5% | 부분 용량 |
| 11 | 두산엔진MG태양광 (창원HSD) | 77 | 0.1% | 경계선 (피크 14시), 노이즈 검증용 |

### 2.4 학습/검증/테스트 분리

| 데이터 | 분리 |
|---|---|
| Phase 1 (D-1 baseline 모델) | 2022-2023 train / 2024 valid / 2025 test |
| Phase 2 (intraday reforecast 모델) | 2024 train (10/15 split) / 2025 test |

근거:
- 4년 데이터 확보. 가장 최신 1년 (2025) 을 test 로 보존 → backtest 정합
- Phase 2 는 *issue 단위* sample 생성하여 sample 수 풍부 → 1년 학습으로 sufficient
- 2024 single-year valid 가 cloud-pass / outage 등 대표 event 포함 → 적합

### 2.5 데이터 가용성 / 결측 처리

**(a) GK-2A 위성 일사** — 48개월 NCDC SFTP 직접 수신 (`data/gk2a_raw/`, 207,958개 NC, 331 GB). 추출본 `data/gk2a_v3/` 166 MB. 시간 무결성·DSR 값 범위·일자별 커버리지 모두 정상.

**(b) ASOS 지상 기상** — 일부 사이트 매핑 거리:
- 고흥만 수상태양광 → 고흥 ASOS 5.3 km
- 광양항세방 → 광양 2.7 km
- 구미·삼천포 → ASOS 거리 수 km, 단 *icsr (일사) 0% 관측* → GK-2A 위성 의존
- 영흥 → 인천 27 km (서해안 대표 관측소)
- 예천 → 안동 24 km (내륙)

**(c) 결측 처리 정책**
| 유형 | 표시 | 학습 처리 |
|---|---|---|
| `exact / interp_*` (정상) | OK | 학습 포함 |
| `zero_seasonal` (계절적 0, 일출 전·일몰 후) | OK | 0 으로 학습 포함 |
| `unrec_archive_gap` (KMA 서버 archive 빠짐) | NaN | 학습 제외 |
| `read_error / out_of_grid` | NaN | 학습 제외 |
| 예천 anomaly_zero (4.95% 행) | OK | *마스킹 없이* 학습 포함 (운영 모듈은 actual=0 신호도 학습 자료로 사용) |

### 2.6 운영 데이터 부재의 *proxy* 처리 (PoC framing)

내부 운영 데이터 부재를 다음 두 가지 proxy 원칙으로 보완한다.

**(a) 기상예보 proxy** — D-1 archived forecast 부재 → ASOS·GK-2A 실측을 *perfect-foresight weather proxy* 로 사용. 즉 본 PoC 의 모든 PV 모델 성능 수치는 *예보 모델 자체가 도달 가능한 상한선* 이다. 실제 운영 D-1 환경에서는 NWP forecast 오차가 추가되어 반드시 더 나쁘다 (`framing.md §2`).

**(b) D-1 thermal plan proxy** — 실제 KPX 화력 계획선 부재 → 2025 실측 LNG hourly 를 *그날의 기준 운전 상태* 로 해석. 이는 dispatch 정답을 복원하는 것이 아니라 *baseline 위 추가 대응량만 계산* 하기 위한 framing (§4.5 in `lng/plan.md`).

이 두 proxy 는 본 PoC 의 한계인 동시에 *PoC 라는 범위에서 합리적인 가정* 이며, 향후 실제 데이터 연동 시 즉시 확장 가능한 구조로 설계되었다 (§11).

---

## 3. 변수 선정

### 3.1 변수 선정 원칙

본 과제의 변수 선정은 **세 가지 기준** 을 따른다.

1. **공공 데이터로 확보 가능** — 향후 D-1 운영 환경에서 *동일 형태로 forecast 입력 변환 가능* 해야 함
2. **cf 와의 *물리적 단조 관계* 또는 *해석 가능 신호*** — black-box feature 미사용
3. **학습-예보 일관성** — 학습 시 입력과 운영 시 forecast 입력의 *형태/단위/해상도* 가 동일해야 함

### 3.2 채택 변수 7종 (`pv/eda_pv_model/D2_variable_selection.png`)

| 변수 | 출처 | cf 와의 corr | 모델 입력 의미 |
|---|---|---:|---|
| **dsr_mean** | GK-2A 위성 (10분 → 시간) | **+0.72 (★ 압도)** | 광역 일사 — 가장 강한 단조 신호 |
| zenith_center | 천문 계산 (lat/lon/time) | -0.61 | 태양 천정각 (daily seasonality) |
| **hm** | ASOS 습도 | -0.54 | 흐림/안개 (산란 손실) |
| **rn** | ASOS 강수 | -0.20 | 비 오는 시간 cf 저하 |
| ws | ASOS 풍속 | +0.22 | 모듈 냉각 (미미) |
| ta | ASOS 기온 | +0.13 | 모듈 효율 (미미) |
| site_oh | one-hot (8 사이트) | — | 사이트별 bias 흡수 (광양항 +7% 등) |

### 3.3 누적 R² — 변수 추가별 cf 설명력 증가

학습셋 8,212 rows (모든 feature 정합 행) 으로 누적 R² 측정 (`D2_variable_selection.png ②`):

| 변수 조합 | 변수 수 | R² (Linear) | R² (Random Forest) |
|---|---:|---:|---:|
| ① zenith only | 1 | 0.05 | 0.23 |
| ② + dsr (GK-2A) | 2 | **0.61** ★ | **0.77** ★ |
| ③ + ASOS 기상 (ta, hm, ws, rn) | 6 | 0.63 | 0.83 |
| ④ + site_oh (8 사이트) | 14 | 0.65 | 0.84 |

**핵심 관찰**:
- **zenith → +dsr 로 R² 0.05 → 0.61 (linear)** — 위성 일사 한 변수만 추가해도 *cf 설명력 12배 증가*
- **+ASOS 기상으로 R² 0.61 → 0.63** — 작지만 명확한 marginal value (특히 RF: 0.77 → 0.83)
- **+site_oh 로 R² 0.63 → 0.65** — 사이트별 bias 흡수의 한계 (개별 사이트는 미세 영향) → AdaLN 의 *동적 conditioning* 으로 보강 필요 (§5)

### 3.4 GK-2A 위성 vs ASOS 지상 — 보완 관계

| 변수 | 위치 | 공간 해상도 | 한계 |
|---|---|---|---|
| GK-2A DSR | 위성 (LCC 2km) | 사이트별 정확 매핑 | 야간 NaN, 픽셀 평균 (구름 미세 변동 제한) |
| ASOS 기상 | 지상 11개소 | 5~30 km 거리 | 직접 사이트 측정 X. 단 광역 패턴은 충분 |
| ASOS icsr (일사) | 일부 사이트 | 0% 관측 (구미·남해) | GK-2A 의존도 ↑ |

→ **결론**: GK-2A 의 광역 일사 + ASOS 의 기상 보조 변수 *조합* 이 단일 소스 보다 우월. 학습-예보 일관성도 만족 (운영 시 GK-2A 실시간 + KMA 단기예보 + Open-Meteo GHI 로 동일 형식 forecast 변환 가능).

### 3.5 변수 선정의 운영 의미

| 변수 | 운영 의미 |
|---|---|
| dsr (GK-2A) | *광역 구름 패턴* — cloud-pass event 의 1~3시간 사전 신호 (Phase 2 의 핵심 입력) |
| hm (습도) | *안개 / 부분 흐림* — DSR 만으로 못 잡는 산란 손실 |
| rn (강수) | *강수 시 cf 저하* — discrete 신호 |
| zenith / hour / month | *daily / seasonal 주기* — AdaLN conditioning 으로 흡수 |
| site_oh | *사이트별 bias / 변동성* — 광양항 +7%, 예천 anomaly_zero 등 사이트 특성 반영 |

→ 이 변수 조합은 *Phase 1 (D-1 baseline)* 과 *Phase 2 (intraday reforecast)* 모두 동일하게 사용된다. 학습-예보 일관성 + 운영 가능성 양축 만족.

### 3.6 사용하지 않은 변수 (명시)

| 변수 | 사용 안 함 이유 |
|---|---|
| 외부 archived weather forecast (Meteologix 등) | 일관된 backtest history 확보 불가 → ASOS·GK-2A 를 *perfect-foresight proxy* 로 대체 (§2.6 a) |
| SMP / 시장 정보 | 본 PoC 평가는 운영 KPI (shortfall / over-commit) 중심이라 시장 신호 미사용 |
| 수동 운량 (cloud cover) | GK-2A DSR 자체가 구름 영향을 반영한 값. 별도 운량 변수 불필요 |
| 장기 기후 변수 (ENSO 등) | Phase 1 baseline 의 monthly conditioning 으로 충분 |

### 3.7 본 변수 조합의 *데이터 내재적* 정당성 요약

`D2_variable_selection.png` 의 4 panel 결과:

1. **dsr (GK-2A) 단일 변수만 추가해도 R² 0.05 → 0.61** — 위성 일사가 cf 의 절대적 설명 변수
2. **dsr 저분위 (Q1-Q4) 영역에서 cf 분포 폭 큼** — 부분 흐림 영역 = Phase 2 event branch 가 잡아야 할 영역
3. **시간대별 corr 안정성** — dsr 은 9~16시 일관적 강함. ASOS hm/ws/ta 는 시간대 의존 (정오 ±2h 에서 marginal value 큼)
4. **site_oh 가 R² 0.012pp 만 추가** — 정적 dummy 만으로 부족. **AdaLN 의 *동적 site/hour/month conditioning* 이 결정적** (§5 motivating EDA M1)

---

## 통합 시 고려사항 (user 검토용)

기존 `proposal_outline.md` 와의 통합 옵션:

1. **§2 위치**: 기존 §4 (데이터 구성) 를 새 §2 (데이터 구성과 제약) 로 *교체* + 본 draft 내용 으로 확장
2. **§3 위치**: 기존 §3 (세부내용 — 시스템 구조) 은 §8 (통합 운영 구조) 로 이동, 본 draft §3 (변수 선정) 을 새 §3 으로 신설
3. **§4 → §4 (PV 데이터 EDA)**: 기존 §5A (PV 모델 motivating EDA) 를 §4 로 이동
4. **§5 → §5 (PV 예측모델 설계 + 채택 근거)**: 기존 §6 (예측모델) + §7 (outage) 통합

전체 chapter renumber 는 user 결정 필요.
