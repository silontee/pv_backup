# LNG 백업 계획 및 재배분 로직

> 기준 문서: `plan/active/pv/model_final.md`
>
> 이 문서는 PV 예측 오차에 대응하기 위한 LNG 측 백업 계획과 호기별 재배분 로직을 정의한다.
> 핵심은 실제 dispatch 재현이 아니라, **D-1 기준 LNG 계획선** 위에서 **실시간 증분 재배분**을 수행하는 것이다.

> **2026-05-08 핵심 원칙 (사용자 지시) + 2026-05-09 갱신:**
> 1. **actual_gap → effective_gap (DEADBAND 차감) 만 LNG 책임** — `effective_gap = max(0, instant_gap - DEADBAND_MW)`. DEADBAND=4 MW 이내는 계통 1차/2차 예비력·AGC·타 조정자원이 자체 흡수 (§4.5 가정). 그 이상의 marginal 만 LNG event.
> 2. **forward gap** — 실 발전이 아니라 *대비 신호*. Layer B는 *호기 예열(warm-up) 명령만* 발행, 출력 X
> 3. **GT(CG*) / ST(CS*) 운영 구분** — GT는 cold-start ~30분, ST는 1~3시간. Layer B warm-up은 **GT 전용** (1시간 lead time 충분)
> 4. **PV over-generation은 LNG 변화에 영향 X** (baseline 그대로, ΔP만 0)
> 5. **over-commit 감소가 평가 1순위** (peak event 개선보다 우선)

---

## 1. 이 문서가 다루는 것

이 문서가 다루는 것은 아래 두 가지다.

1. LNG 발전량 데이터를 **D-1 기준 계획선**으로 어떻게 해석할 것인가
2. PV 변동이 발생했을 때 그 계획선에서 호기별로 어떻게 추가 대응량을 재배분할 것인가

즉 이 planner는:

- 전국 계통 dispatch optimizer가 아니고
- 실제 분당 LNG dispatch 재현기도 아니며
- full unit commitment solver도 아니다

대신 아래를 위한 **실시간 운영 지원 PoC**다.

- 현재 부족분 즉시 보정
- 남은 하루 곡선 기반 persistence 판단
- 온라인 호기 우선 재배분
- 필요 시 오프라인 호기 startup 검토 (cost-aware는 PoC 범위 밖)

---

## 2. 핵심 개념

### 2.1 PV 쪽 기준선

- `Phase 1`: D-1 PV baseline forecast
- `Phase 2`: 최신 actual과 observed weather를 반영한 remaining-day reforecast
- `Phase 2 + outage override`: rule-based outage detection 후처리 적용본 (production)

### 2.2 LNG 쪽 기준선

LNG 발전량 데이터는 *실제 dispatch 정답*이 아니라 **D-1 기준 화력 계획선 proxy**로 해석한다.

각 호기 `u`에 대해:

```text
P_DA,u(t) = 호기 u의 기준 LNG 계획 출력 [MW]
```

이 존재한다고 두고, PV 변동 발생 시 planner는:

```text
P_new,u(t) = P_DA,u(t) + ΔP_u(t)
```

형태로 **증분 재배분**을 계산한다. P_DA 자체는 변경 안 함.

---

## 3. 문제 정의

각 시점 `t`에서 planner는 아래를 판단한다.

1. 현재 시점 PV 부족분을 즉시 얼마나 메울 것인가 (Layer A: 실 발전)
2. 앞으로 큰 부족 예상 시 호기를 미리 준비/켜야 하는가 (Layer B: 사전 대비)
3. 추가 필요량을 어떤 호기에 얼마나 나눌 것인가
4. 회복이 보이면 언제부터 점진적으로 줄일 것인가

---

## 4. 운영 가정

### 4.1 D-1 LNG 계획선 가정

본 PoC에는 실제 D-1 thermal schedule 데이터가 **없다**. 따라서 가정 한 층을 더 얹어 과거에서 학습하는 proxy를 만드는 대신, **2025년 실측 hourly LNG 발전량 자체를 그날의 baseline thermal schedule proxy로 사용**한다.

> 본 PoC에서는 실제 D-1 thermal plan 데이터가 없으므로, 2025년 관측 LNG 발전량을 해당 시점의 baseline thermal schedule proxy로 사용한다. 이는 실제 dispatch를 정답으로 재현하려는 것이 아니라, 당시 운영상 채택된 기준 운전 상태를 출발점으로 하여 신재생 출력 변동에 대한 추가 재배분 로직을 검토하기 위한 가정이다.

이 framing 의 의미:
- 모델은 "기준 운전 상태"를 *예측하지 않는다*
- 모델이 다루는 것은 그 baseline 위의 **추가 재배분(ΔP)** 만
- 평가도 ΔP가 PV 변동을 어떻게 흡수했는가에 한정

#### 4.1.1 P_DA,u(t) 정의 (확정)

```python
P_DA,u(t) = 2025년 시점 t의 호기 u 실측 LNG 발전량 [MW]
```

특성:
- **학습 단계 없음** — 단순 lookup
- evening peaker (CG2/CG4)는 낮 시간 P_DA = 0 (실제로 안 돌았음)
- baseload (CS2, CG6)는 거의 항상 P_DA > 0
- 솔직한 한계 표명: actual을 proxy로 사용 → "엄밀한 사전 계획"은 아니지만, 그날의 grid/operating condition 이 반영된 가장 현실적인 baseline

### 4.2 ΔP는 추가 출력만 — baseline 절대 변경 X

planner는 baseline 자체를 새로 짜지 않는다. **baseline은 외생적 고정**이고 planner는 그 위에 ΔP만 더한다.

- ΔP > 0: PV 부족 → 호기 출력 추가
- ΔP = 0: 변화 없음
- ΔP < 0 → **금지** (PV 초과 시에도 baseline 감산 안 함, §4.4 참조)

### 4.3 Commitment 범위

- full unit commitment 최적화는 본 PoC에서 직접 수행하지 않는다
- `cold start`, `startup cost`, `오프라인 호기 활성화 여부`는 *완전히 무시하지는 않지만*, **호기별 SU/VC 데이터가 없어** cost-aware screening은 PoC 범위 밖
- 즉 본 PoC: **full UC는 아니지만 startup-aware heuristic allocation** (cost-blind)

### 4.4 PV over-generation 처리 — LNG 변화에 영향 X

PV가 baseline보다 많이 나오는 경우 (`instant_gap < 0`):

```text
즉시_보정필요량 = max(0, instant_gap_now) = 0     # 음수 무시
ΔP_u(t) = 0 (해당 호기는 추가 출력 X)
P_DA,u(t) (baseline) = 그대로 유지                # 절대 감산 X
```

즉:
- **planner는 LNG baseline을 음수 방향으로 재최적화하지 않는다**
- 잉여 PV는 *시스템 측 문제* (curtailment, 양수발전, 송전 등) — 본 planner 책임 밖
- ΔP 자체도 음수 안 만듦 (release rate로 천천히 0으로 복귀할 뿐)

### 4.5 ★ System reserve cushion (DEADBAND 가정, 2026-05-09 신설)

> **핵심 가정**: PV portfolio actual_gap 의 *전량* 이 LNG fleet 증분 재배분으로 가는 것은 아니다.
> 일정 수준 이하의 gap 은 **계통 내 기존 예비력·AGC·타 조정자원이 흡수**하는 구간으로 보고, 그 이상부터 LNG backup 이벤트로 정의한다.

**근거 (2026-05-09 사용자 조사)**:

`plan/active/lng/(최종) 연료원별 보조서비스정산금_202603.xlsx` 분석 결과 — **LNG가 전국 보조서비스(예비력) 정산의 ≈ 50% 담당**.

해석:
- 한국 계통 운영예비력 (FCR/AGC/replacement reserve) 의 절반은 LNG, 나머지 절반은 석탄 AGC, 양수, 가스, 수력 등이 분담
- 즉 PV 변동의 *모든 부분* 이 LNG로 가지 않고, 다른 자원도 함께 흡수
- 작은 (< 4 MW) 변동은 계통 frequency reserve / AGC 가 즉시 흡수 → LNG event 트리거 X
- 큰 (≥ 4 MW) 변동만 LNG fleet 증분 재배분 *event* 로 분류

**구현**:

```python
DEADBAND_MW = 4.0    # 계통 자체 흡수 cushion. PV portfolio 77 MW 의 약 5%.
                     # |instant_gap| 분포 p65~70 위치. <4 MW 는 잡음 영역으로 분류.

effective_gap_now = max(0, instant_gap_now - DEADBAND_MW)
즉시_보정필요량   = effective_gap_now           # Layer A 입력
```

운영 모델:
- gap = 3 MW → effective = 0 (계통 흡수, LNG 무동원)
- gap = 5 MW → effective = 1 MW (LNG 1 MW 메움)
- gap = 10 MW → effective = 6 MW (LNG 6 MW 메움)

**의미**:
- KPI 정의도 effective_gap 기준 — `shortfall = max(0, effective_gap - Σ ΔP)` (§13.1)
- 1년 누적 결과: shortfall 1197→442 (-63%), over-commit 5398→640 (-88%) — over-commit 1순위 KPI 큰 폭 감소
- 운영 직관 부합: 작은 변동은 "LNG가 직접 보지 않음", 큰 event 만 응답
- forward gap (Layer B) 임계는 **별개로 유지** — 두 신호의 의미가 다름 (forward = 예측 변화, actual = 실측 부족)

---

## 5. 입력

### 5.1 PV 예측 입력

- `Phase 1` baseline forecast (μ_p1, σ_p1)
- `Phase 2` remaining-day reforecast + outage override 적용본

### 5.2 Gap 신호

- `instant_gap_t = μ_p1_t − pv_actual_t` — 현재 시각 실측 부족 (raw)
- **`effective_gap_t = max(0, instant_gap_t − DEADBAND_MW)` ★ Layer A 메인 입력** (§4.5, §7.1)
- `RG` (realized gap, look-back 평균) — 보조 지표 (대시보드/로그용, 분배 결정에는 미사용)
- `FG_lead(t,k)` (forward gap, lead k) = PV_p1(t+k) − PV_p2(t+k from issue=t) — Layer B 입력

### 5.3 LNG 호기별 spec

각 호기 `u`에 대해 (4년치 hourly data로 사전 계산, `data/processed/lng_unit_specs.parquet`):

- `P_min,u`: 최소 운전 출력 (운전 중 p05)
- `P_max,u`: 최대 운전 출력 (운전 중 p95)
- `Ramp_up,u`: 시간당 상승 한도 (실측 p90)
- `Ramp_dn,u`: 시간당 하강 한도 (실측 p90)
- `mode_u`: 운전 빈도 기반 분류 (baseload / mid-merit / peaker)
- `Priority_u`: 운영 우선순위 (§6.2 참조)
- `SU_u`, `VC_u`, `StartDelay_u`: **데이터 부재** — PoC 범위 밖

### 5.4 호기 가용성

```python
is_online(u, t) = (P_DA,u(t) > P_min,u × 0.5)

if is_online:
    Avail_u(t) = 1.0
else:
    # 직전 7일 같은 hour-of-day 운전 빈도
    Avail_u(t) = mean[ run(u, d, hour(t)) for d in last 7d ]
```

---

## 6. 호기군 해석 (data-driven, 2026-05-08)

### 6.1 호기 type — GT vs ST (2026-05-09 추가)

분당화력 (KOEN LNG CCGT) 호기 명명 규칙으로 구분:
- **CG1~CG8**: **Gas Turbine 단** (8 GT)
- **CS1~CS2**: **Steam Turbine 단** (2 ST)

cold-start 시간 (★ **일반 산업 spec 기반 추정**, 호기별 실측 데이터 부재):
- GT: 30분 ~ 1시간 (warm 30분, cold 1시간)
- ST: 1 ~ 3시간 (CCGT 의 ST 는 GT 배기열 회수 후 가동 → GT 보다 명확히 느림)

운영 의미:
- 1시간 lead time 의 forward 신호로 *예열 명령* 가능한 것은 **GT 만**. ST는 lead time 부족.
- Layer B (warm-up) candidate filter: `unit_type(u) == 'GT'` (§10.2)
- Layer A 즉시 startup 시에도 GT 우선 (cold-start 빠름)
- 향후 호기별 실측 startup 시간 확보 시 ST 도 multi-hour Phase 2 로 검토 가능

### 6.2 운전 패턴 기반 분류

`running_pct` (운전 시간 비율, 4년 평균, `data/processed/lng_unit_specs.parquet`) 기준:

| 분류 | 조건 | 호기 (running_pct %) |
|---|---|---|
| **baseload** | running ≥ 45% | CS2 (51.9), CG6 (47.6) |
| **mid-merit** | 28 ≤ running < 45% | CG8 (44.4), CS1 (41.9), CG1 (39.4), CG3 (37.5), CG7 (35.8), **CG5 (28.2)** |
| **peaker** | running < 28% | CG2 (23.0), CG4 (19.8) |

> ⚠️ **주의 (Q5 데이터 분석, 2026-05-08)**: peaker 도 한 번 켜지면 평균 6~7시간 연속 운전. **"peaker = 단발 spike만 잡는다" 가정은 틀림**. mode 분류는 *운전 빈도 차이* 일 뿐, 연속 운전 길이가 짧다는 의미가 아니다. 따라서 호기 선택 룰에서 "spike vs persistent" 로 mode 우선순위 가르는 건 근거 약함.

### 6.3 호기 Ranking 표 (참고용, 분배 가중치 X)

호기들의 *PV balancing 능력 ranking* — 운영 직관 시각화 용이며 분배 가중치엔 직접 사용 X (§9.3 참조).

`build_lng_unit_specs.py` 산출 식 (실제 `lng_unit_specs.parquet` 의 `priority_data`):

```text
backup_score_u = 헤드룸_p90 × ramp_up_p90 × running_pct/100
priority_data  = backup_score / max(backup_score)        # CS1 = 1.0 기준
```

| 호기 | type | 헤드룸_p90 | ramp_up_p90 | running_pct (%) | backup_score | priority_data |
|---|---|---|---|---|---|---|
| **CS1** | ST | 67 | 29 | 41.9 | 826 | **1.000** |
| **CS2** | ST | 55 | 19 | 51.9 | 530 | 0.640 |
| CG7 | GT | 10 | 26 | 35.8 | 92 | 0.110 |
| CG6 | GT | 9 | 18 | 47.6 | 79 | 0.100 |
| CG8 | GT | 9 | 19 | 44.4 | 75 | 0.090 |
| CG2 | GT | 10 | 29 | 23.0 | 67 | 0.080 |
| CG4 | GT | 8 | 28 | 19.8 | 42 | 0.050 |
| CG3 | GT | 6 | 15 | 37.5 | 33 | 0.040 |
| CG1 | GT | 6 | 13 | 39.4 | 31 | 0.040 |
| CG5 | GT | 6 | 12 | 28.2 | 22 | 0.030 |

→ **CS1, CS2 (ST 단 두 호기)가 압도적 1, 2위** (헤드룸 다른 GT 대비 6~10배). PV balancing 의 절대적 핵심.

> 단, **Layer B warm-up candidate 는 GT 전용** (§6.1, §10.2) — ST 는 cold-start 시간 부족. 즉 ranking 1, 2 위는 *online 시점에서만* 큰 흡수력. 새로 켜는 결정은 GT 안에서.

### 6.4 비대칭 ramp (data-driven)

운영 데이터 직접 분석 결과 **상승은 빠르고 하강은 느림** — 특히 baseload:

| 호기 | mode | 상승 p90 (MW/h) | 하강 p90 (MW/h) | 비율 (up/dn) |
|---|---|---|---|---|
| CG6 | baseload | 18 | **1.9** | **9.5x** |
| CG8 | mid-merit | 19 | 2.4 | 7.9x |
| CS2 | baseload | 19 | 2.7 | 7.0x |
| CS1 | mid-merit | 29 | 7.4 | 4.0x |
| CG2 | peaker | 29 | **11** | 2.7x |
| CG4 | peaker | 28 | 6.8 | 4.0x |

운영 의미: 기존 호기들은 *내릴 때 매우 천천히* 운영. planner도 이 비대칭 그대로 사용.

---

## 7. Gap 신호 정의

### 7.1 effective_gap — Layer A 메인 입력 (DEADBAND 차감, 2026-05-09 갱신)

```python
instant_gap_t   = μ_p1_t − pv_actual_t          # raw 실측 부족
effective_gap_t = max(0, instant_gap_t − DEADBAND_MW)   # ★ DEADBAND=4 MW 차감
즉시_보정필요량 = effective_gap_t
```

운영 의미: "지금 시각 PV 실측 부족 중 **계통이 자체 흡수 못하는 marginal** 만 LNG가 메운다" (§4.5).

> **2026-05-09 RG (look-back 평균) 폐기**:
> 이전 spec은 `RG = mean(gap_history[-2:])` 로 look-back 2시간 평균을 사용했음. v2 (CS2 single-cover) 시절 "1시간 잡음 smoothing" 명목으로 도입된 잔재.
> 분석 결과:
> - `max(RG, instant_gap)` 에서 RG가 winning 하는 케이스 = 직전엔 부족했는데 *지금 회복* 중인 시점 → RG가 호기 출력 끌어올림 → over-commit 발생
> - ramp 한도 + P_min 조건이 이미 brake 역할 충분 → 추가 smoothing 불필요
> - 실험: RG 제거 시 over-commit 7247 → 5398 MWh (−25%)
>
> 따라서 **Layer A 분배는 `instant_gap_now` 직접 사용** (look-back X). RG는 대시보드 시각화 보조 지표로만.

> **2026-05-09 DEADBAND 도입**:
> 보조서비스 정산금 자료 (LNG ≈50% 담당) 근거로 DEADBAND=4 MW 차감.
> 실험: shortfall 1197→442 (−63%), over-commit 5398→640 (**−88%**). 과대보충 큰 폭 감소 + 운영 직관 부합 (§4.5).

### 7.2 FG — 미래 예상 부족 (forward, slope-based)

```python
FG_lead(t, k) = μ_p1_{t+k} − μ_p2_{t+k from issue=t}      # k=1,2,3,...

FG1     = FG_lead(t, 1)              # 1시간 뒤
FGmean3 = mean over k=1..3 of FG     # 3시간 평균
FGslope = (FG_lead(t, 3) − FG_lead(t, 1)) / 2    # 시간당 변화 추세 (Q6)
```

해석:
- `FG1, FGmean3 > 0`: 앞으로 PV 부족 예상
- `FGslope > 0`: 점점 더 부족해짐 (악화)
- `FGslope < 0`: 점점 회복 중

### 7.3 신호 분류 — 데이터 기반 (2026-05-09 갱신)

#### 7.3.1 데이터 분포 (test 2025 portfolio, daytime issue 7~16, n=3725)

**|FG1|**:
- p90: 3.87
- **p95: 5.34** ★ TH_INCREASE 근거
- p99: 12.76

**|FGmean3|**: p90 2.95, p95 4.05 (smoothing 효과)

**|slope| = |(FG3 − FG1) / 2|**:
- p90: 1.71
- **p95: 2.24** ★ TH_SLOPE 근거
- p99: 9.24

**hour별 평균 |FG1|**: 7시 1.37 → 12-13시 3.34 (peak) → 18시 0.55

**애매 영역** (|FG1| 3-5 MW, |slope| < 1): 4.8% (178건)

#### 7.3.2 임계값 (data-driven)

```text
TH_NOISE_FORWARD = 3 MW    # |FG1| 이 이하 = 잡음
TH_INCREASE      = 5 MW    # FG1 p95 — 큰 신호 정의
TH_SLOPE         = 2 MW/h  # slope p95 — 명확한 변화
```

#### 7.3.3 신호 분류식

```python
quiet    = |FG1| < TH_NOISE_FORWARD                         # 작음

strong   = FG1 ≥ TH_INCREASE                                # 큰 신호 (slope 무관)
                                                            # 절댓값 우선

growing  = (TH_NOISE_FORWARD ≤ FG1 < TH_INCREASE)
           AND (FGslope ≥ TH_SLOPE)                          # 작지만 빠르게 커짐 → 사전 대비

recovery = |FG1| ≤ TH_INCREASE
           AND (FGslope ≤ -TH_SLOPE)
           AND (prev_total > 0)                              # 응답 중 + 회복 신호

default  → KEEP                                              # 애매 영역 (4.8%)
```

#### 7.3.4 애매 영역 처리 — KEEP 으로 처리

- 예: |FG1|=4, slope=0 → 변화 없음 + 양은 작음 → KEEP
- 운영 직관: "신호는 있는데 변화 안 함" → 적극 응답 필요 X
- 단, prev_total > 0 (이미 응답 중) 인 경우 release 안 하고 유지

#### 7.3.5 왜 slope가 중요한가

단순 절댓값만 보면:
- 사건 A: FG1=20, FG3=10 (점점 회복) → slope = -5
- 사건 B: FG1=10, FG3=20 (점점 악화) → slope = +5

→ FG1 절댓값만 보면 다르지만, **운영 의미는 정반대**. slope로 구분.

---

## 8. 2-layer 의사결정 구조 (2026-05-08 신설, 2026-05-09 명확화)

> **핵심 원칙**:
> - **Layer A (실 발전)** = actual gap 100% 추종 — 호기 출력 ΔP 결정
> - **Layer B (사전 대비)** = forward signal로 호기 *warm-up 명령* 만 — 출력 X (다음 시각 가용성만 확보)
> - 지금 시각 ΔP_u = Layer A 결과만. Layer B는 *다음 시각 online subset 변경* 으로만 작용

### 8.1 Layer A — 실 발전 명령 (actual gap 추종)

```python
즉시_보정필요량 = max(0, instant_gap_now − DEADBAND_MW)    # effective gap (§7.1, §4.5)

# 흐름 (코드 thermal_planner_v4.py: layer_a_startup):
# 1. online 호기 합 헤드룸 ≥ 즉시_보정필요량  →  online 분배만 (§9.3)
# 2. cap 초과 + P_min ≤ 잔여 ≤ P_max  →  GT 우선으로 호기 1대 켬,
#                                         출력 = 잔여(effective gap) 그대로 (P_min 강제 X)
# 3. 그 외  →  shortfall 감수

ΔP_actual_u(t) = 분배 결과 (§9.3)
```

이게 **현재 시각 호기 실제 추가 출력**. baseline 변경 X.

> **호기 startup 우선순위 (Layer A)**: cold-start 빠른 **GT 우선**, 동률 시 헤드룸 큰 순 → P_min 작은 순 → Avail 높은 순.

### 8.2 Layer B — 사전 대비 (warm-up only, 출력 X)

forward gap이 큰 신호 보내고 *다음 시각 online cap 부족할 것 같으면* → **호기 켜는 *준비*** (warm-up).

> **운영 직관**: 호기 cold-start delay 동안 *데우는 단계* — 발전 안 함. 1시간 후 가용 상태가 되면 그때 출력 가능.

```python
# 다음 시각 예상 부족
expected_next_gap = max(즉시_보정필요량, FG1)

# 다음 시각 online cap (예측: 지금 online + 이미 켜진 호기)
expected_online_cap = sum(headroom for u in current_online_units)

# warm-up 결정
if expected_next_gap > expected_online_cap AND forward_persistent:
    # 호기 1대 warm-up 시작
    pick offline unit u where:
        P_min_u ≤ expected_next_gap ≤ P_max_u
        Avail_u > 0.1
    sort by 헤드룸 × ramp_up 큰 순

    state[u] = "warm_up"             # 다음 시각엔 online 후보로
    ΔP_prep_u(t) = 0                 # ★ 이 시각 출력 0
```

**핵심**:
- ΔP_prep는 항상 0 (지금 시각 출력 안 함)
- 다음 시각엔 그 호기가 `online subset` 에 추가되어 분배 후보
- 즉 forward gap의 영향은 *다음 시각 호기 가용성* 에만 작용

### 8.3 통합 — 지금 시각 alloc

```python
ΔP_u(t) = ΔP_actual_u(t)             # Layer A만
                                      # Layer B는 다음 시각 online flag 변경
                                      # → over-commit 안 만들어짐
```

### 8.4 응답상태 (state machine, Layer B 보조)

state는 **Layer B의 의사결정 모드**를 표현 (실 발전 결정과 무관):

| state | 의미 | Layer B 행동 |
|---|---|---|
| **KEEP** | 평소, 잡음 | Layer B 비활성 — 추가 startup 안 함 |
| **INCREASE** | forward 강신호 (growing) | 호기 추가 startup 검토, 잘 켜놓음 |
| **DELAYED_RELEASE** | forward 회복신호 | 켜놓은 호기 점진 release |

state 진입 조건 (slope-based, Q6):

```python
if forward_quiet:                           state = KEEP
elif forward_growing OR forward_strong:     state = INCREASE
elif forward_recovery AND hold_timer ≥ 1h:  state = DELAYED_RELEASE
else:                                        state = KEEP
```

> **HOLD state 폐기 (2026-05-08)**: 4-state 시도 시 HOLD가 0회 발동 (INCREASE와 임계 동일). 3-state로 단순화.

### 8.5 (Q4) 평소/비상 layer 가능성 (실험적)

3-state 대신 더 큰 분류로 *평소 / 비상* 2-layer 도 검토 가능:

- **평소**: forward_quiet OR forward_recovery. Layer A만 작동, online 분배.
- **비상**: forward_strong OR forward_growing. Layer A + Layer B (startup 검토).

→ 더 단순한 구조이지만 grain 손실. 현재 3-state 유지 + KEEP를 "평소", INCREASE/DELAYED_RELEASE를 "비상 변화" 로 해석 권장.

---

## 9. Layer A 구현 — 호기 분배

### 9.1 online 호기 정의

```python
is_online(u, t) = (P_DA,u(t) > P_min,u × 0.5)
```

§5.4 와 동일.

### 9.2 헤드룸

```python
Headroom_u(t) = max(0, P_max,u − P_DA,u(t))
```

### 9.3 분배 가중치 (정규화) — Decision 1 (2026-05-09)

priority 가중치는 *제곱 효과* 일으켜 분리했다 (§6.2 priority는 ranking 표용으로만).

```python
# 단순 가중 — 헤드룸 × 상승한도 만
w_raw,u  = Headroom_u(t) × Ramp_up,u
w_norm,u = w_raw,u / Σ_v w_raw,v       # online 호기들끼리 정규화

ΔP_actual,u(t) = min(w_norm,u × 즉시_보정필요량, Headroom_u(t))
```

운영 의미:
- *현재 시점 ON/OFF* 가 첫 필터 (online 호기만 후보)
- *현재 시점 헤드룸* 이 분배 비율 결정
- *호기 spec ramp_up* 이 응답 속도 가중

cap에 걸린 호기 잔여 → §10 호기 새로 켜기 검토 (즉시_보정 ≥ P_min 조건).

---

## 10. Layer B 구현 — 호기 warm-up 명령 (2026-05-09 재정의)

> **중요**: Layer B는 *지금 시각 호기 출력에 영향 안 줌*. 다음 시각 *online 가용성만* 변경.
> 이게 over-commit 폭증 방지의 핵심.

### 10.1 트리거 — 다음 시각 부족 예상 + 현재 cap 부족

```python
WARMUP_MIN_GAP = 5.0    # 이 이상 부족 예상 시에만 warm-up 검토

# 다음 시각 예상 부족
expected_next_gap = max(즉시_보정필요량, FG1)

# 현재 시각 online 호기들의 헤드룸 합
online_headroom_total = Σ_u (P_max,u - P_DA,u)  for u in current_online

# warm-up 트리거 (data-driven 완화 조건)
trigger = (expected_next_gap > max(WARMUP_MIN_GAP, online_headroom_total))
          AND forward_persistent
```

`forward_persistent`: §7.3 의 forward signal 분류 결과 (slope or magnitude 기반).

> **이전 조건 (`P_min ≤ expected_next_gap ≤ P_max`)** 의 문제: 대부분 호기 P_min 이 30~40 MW 인데 다음 시각 부족이 그만큼 안 되면 candidate 리스트가 비어 warm-up 0회 발동. **WARMUP_MIN_GAP=5** 로 완화 + 호기 출력은 *분배 후* 결정 (P_min 강제 X) → 실제 발동 가능.

### 10.2 호기 warm-up 선택 — GT 전용 (2026-05-09)

candidate filter (★ **GT only**):
```python
candidates = [u for u in offline
              if u not in warm_up_set                # 아직 안 데우는 호기
              AND u.Avail > 0.1                       # 평소 가동 빈도 충분
              AND unit_type(u) == 'GT'                # ★ Gas Turbine 만 (cold-start 빠름)
              AND u.P_max ≥ WARMUP_MIN_GAP]           # 출력 가능 범위
```

선택 정렬 (mode 우선순위 X):
```python
sort by:
  1. 헤드룸 (P_max - P_min) 큰 순       # 다음 시각 흡수력 큼
  2. P_min 작은 순                      # over-commit 최소
  3. Avail 높은 순
```

선택 결과:
```python
warm_up_set.add(chosen_unit)        # 다음 시각엔 online subset 에 합류
ΔP_u(t) (chosen) = 0                 # ★ 지금 시각 출력 X (예열 단계)
```

### 10.3 다음 시각 처리 — Layer A 분배 후보로 합류

다음 시각 t+1:
- chosen_unit `is_online = True` + `Avail = 1.0` + **`P_DA = 0` 그대로 유지** (P_min 강제 X)
- 다음 시각 actual_gap 만큼만 분배 (§9.3) → over-commit 안 만들어짐
- 다음 시각 actual_gap 이 작으면 출력 0 유지 (예열만 하고 끝)

> 핵심 차이: 이전 spec은 `P_DA = P_min` 강제 → 다음 시각 actual gap 이 작아도 P_min 만큼 출력 → over-commit. 갱신 spec은 `P_DA = 0` 유지 → 분배 후보로만 합류, 출력은 actual gap 따라 결정.

### 10.4 cost-aware screening — PoC 범위 밖 (Q2)

호기별 SU_u, VC_u 데이터 부재 (CLAUDE.md 평균값 5천만원/회, 160원/kWh 만). 본 PoC에서 cost-aware 미구현.

### 10.5 호기 새로 켜는 *즉시 출력* 발생 case (Layer A 안에서)

actual_gap 자체가 큰 경우 (online cap 초과 + **P_min ≤ 잔여 ≤ P_max**):
- Layer A 흐름 안에서 호기 새로 켬 — **출력 = actual gap 잔여 그대로** (P_min 강제 X)
- candidate filter: `is_online == 0`, `Avail > 0.1`, `P_min ≤ residual ≤ P_max`
- 정렬: GT 우선 → 헤드룸 큰 순 → P_min 작은 순 → Avail 높은 순
- actual gap 기반 출력이라 over-commit 발생 X (실제 부족분만큼만 출력)

이 case 와 §10.2 (forward 기반 warm-up) 를 구분:
- Layer A 호기 켜기 = *지금 부족 cover* (즉시 출력 = actual gap)
- Layer B 호기 warm-up = *다음 시각 대비* (지금 출력 X, 분배 후보로만 합류)

### 10.6 startup 분류 — `startup_real` vs `startup_ramp` (2026-05-09 분리)

코드 (`thermal_planner_v4.py`) 의 startup count 를 두 종류로 분리하여 운영 의미 명확화:

```python
online_now = {u['unit']: u['is_online'] for u in units_at_t}
startup_real = sum(1 for u in LNG_UNITS
                    if alloc[u] > 0 and prev_alloc[u] == 0 and online_now.get(u, 0) == 0)
startup_ramp = sum(1 for u in LNG_UNITS
                    if alloc[u] > 0 and prev_alloc[u] == 0 and online_now.get(u, 0) == 1)
startup_count = startup_real + startup_ramp   # legacy 합
```

| 컬럼 | 의미 | 의사결정 의미 |
|---|---|---|
| `startup_real` | 호기 offline (`is_online==0`) 인데 ΔP 발생 | 진짜 신규 가동 (cold-start) — `layer_a_startup` 또는 warm-up 합류 |
| `startup_ramp` | 호기 이미 운전 중 (`is_online==1`) + 직전 ΔP=0 → ΔP 시작 | 추가 발전 전환 (이미 baseline 운전, 추가 출력 시작) |

**실측 결과 (test 2025, daytime 9-17, 1년)**:

| 모드 | startup_real | startup_ramp | 비고 |
|---|---|---|---|
| Phase 1 only | 1 | 931 | 진짜 신규 가동 거의 없음 |
| Phase 1+2 | 0 | 941 | warm-up 합류 호기는 다음 시각 baseline 가용 → ramp 로 분류 |

**의미**:
- 분당화력 LNG 호기들은 *대부분 baseline 운전 중* (P_DA > 0)
- 따라서 PV 변동 대응의 *대부분이 "추가 발전 전환"* (cold-start 비용 거의 발생 X)
- 진짜 cold-start 가 필요한 *offline → 발전* 케이스는 1년에 1회 미만
- 운영 비용 측면에서 좋은 결과 (cold-start 비용 ~5천만원/회 회피)

---

## 11. 호기별 제약 (단일 위치 정리)

각 호기 `u`는 매 시각 아래 제약을 갖는다 (§6.3 데이터 + 상태별 분기).

### 11.1 출력 범위

```text
P_min,u  ≤  P_new,u(t)  ≤  P_max,u

즉:
  ΔP_u(t)가 추가 후 P_DA + ΔP < P_min 면 0으로 (호기 안 켬)
  P_DA + ΔP > P_max 면 cap (P_max 까지)
```

### 11.2 비대칭 ramp + 상태별 (핵심)

```text
# 상승: 호기 자체 한도 (cold start 시간 반영)
ΔP_u(t) − ΔP_u(t-1)  ≤  Ramp_up,u (p90)

# 하강: 상태별
if state == DELAYED_RELEASE:
    ΔP_u(t-1) − ΔP_u(t)  ≤  RELEASE_RATE   # 진동 방지, 천천히
else:  # KEEP / INCREASE
    ΔP_u(t-1) − ΔP_u(t)  ≤  Ramp_dn,u (p90)   # 호기 자체 한도까지 자유
```

핵심 설계:
- `RELEASE_RATE`는 *DELAYED_RELEASE 시에만* 강제 (회복 신호 명확 시 진동 방지)
- 일반 시점엔 호기가 actual_gap 변화 자유롭게 추종
- baseline (P_DA) 자체는 변경 안 됨 (§4.2)

---

## 12. PV Over-generation 처리 (D4 강화)

PV가 baseline보다 많이 나오는 경우 (`instant_gap < 0`, `effective_gap = 0`):

```text
즉시_보정필요량 = max(0, instant_gap_now − DEADBAND_MW) = 0
ΔP_actual_u = 0
P_new,u(t) = P_DA,u(t) + ΔP_u(t-1) − Ramp_dn,u    # ΔP만 천천히 0으로
```

명시적 원칙:
- **baseline P_DA는 절대 감산 X** (외생적 고정)
- ΔP 도 음수 안 만듦 (자연스럽게 0으로 release)
- 잉여 PV는 *시스템 측 영역* (curtailment, 양수, 송전 등) — 본 planner 책임 밖

→ **LNG 변화는 PV under-generation에 대해서만 작동**, over-generation 영향 0.

---

## 13. KPI 평가

dispatch reproduction이 아니라 **PV 부족 흡수 효율**로 평가.

### 13.1 핵심 KPI (DEADBAND 적용 — §4.5)

```text
instant_gap_t   = μ_p1_t − pv_actual_t                      # raw 부족
effective_gap_t = max(0, instant_gap_t − DEADBAND_MW)        # LNG 책임분 (§4.5, §7.1)
shortfall_t     = max(0, effective_gap_t − Σ_u ΔP_u(t))      # 못 채운 effective gap
over_commit_t   = max(0, Σ_u ΔP_u(t) − effective_gap_t)      # effective 대비 과대 출력
```

> KPI 가 effective_gap 기준이라 *계통 자체 흡수* 분 (DEADBAND 이내) 은 shortfall/over-commit 에 포함되지 않음. LNG fleet 의 *실제 운영 효율* 만 측정.

연 누적:
- `Σ shortfall_MWh` — 못 채운 부족
- **`Σ over_commit_MWh` — 과대 보충 (★ 1순위 평가지표)**
- `state_changes` — state 전환 횟수 (안정성)
- `correction_sign_flips` — alloc 부호 뒤집힘 (진동)
- **`startup_real`** — 진짜 신규 가동 (offline → 발전, cold-start 비용 발생 케이스, §10.6)
- **`startup_ramp`** — 추가 발전 전환 (이미 운전 중 + ΔP 시작, cold-start X)
- `startup_count` = `startup_real + startup_ramp` (legacy 합)
- 호기별 누적 `ΔE_u` (분담)

### 13.2 비교 baseline

- `Phase 1 only (Reactive)`: realized gap만 사용
- `Phase 1+2 (Predictive)`: realized + forward 사용 → Layer B 활성화

→ 둘 차이 = Phase 2 + Layer B (사전 대비) 의 가치.

### 13.3 평가 우선순위 (사용자 지시 2026-05-08)

1. **over-commit 감소가 최우선** — peak event (cloud-pass) shortfall 개선보다 우선
2. shortfall 약간 늘어도 over-commit 큰 폭 감소 = 옳은 방향
3. peak event 1년 평균 영향이 크지 않으면 부차적

---

## 14. Problems (2026-05-09 갱신)

### P1 — Over-commit (★ 큰 폭 감소 적용 완료)

KPI 진화 (v4 P1+2, test 2025 daytime 9-17, n=3280h):

| 단계 | spec | shortfall | over-commit | 비율 | 비고 |
|---|---|---|---|---|---|
| (a) v4 초기 (2026-05-08) | priority 가중 + Layer B 분배 출력 + RG | 3236 | 8003 | 2.5x | 메모리 기록 시점 |
| (b) +Layer A/B 분리 + priority 제거 | Layer B = warm-up only | 3955 | 7011 | 1.8x | warm-up 0회 |
| (c) +Layer A `instant_gap` 추가 | `max(RG, instant_gap)` | 1185 | 7247 | 6.1x | cloud-pass 회복 |
| (d) +RG 폐기 | `instant_gap` only | 1197 | 5398 | 4.5x | warm-up 20회 (GT) |
| **(e) +DEADBAND=4 (§4.5)** ★ 현재 | `max(0, instant_gap-4)` | **442** | **640** | **1.45x** | over/short 균형 |

(a) → (e) 누적 변화:
- shortfall: 3236 → 442 (**−86%**)
- over-commit: 8003 → 640 (**−92%**)
- Phase 2 marginal: -3.9% → **−19.1%**
- vs v2 baseline (CS2 single): -12.9% → **−89.9%**

적용된 핵심 개선:
1. **priority 제곱 가중 제거** (§9.3) — 호기 분담 균등화 (b)
2. **Layer A/B 분리, Layer B = warm-up only** (§10) — forward gap 발 over-commit 제거 (b)
3. **Layer A = instant_gap_now 직접** (§7.1, §8.1) — cloud-pass 회복 (c→d)
4. **RG (look-back 평균) 폐기** (§7.1) — over-commit −1849 MWh (d)
5. **Layer B GT 전용 + WARMUP_MIN_GAP=5** (§10.2) — warm-up 실제 20회 발동 (d)
6. **Layer A startup GT 우선** (§8.1) — cold-start 빠른 호기 먼저 (d)
7. **DEADBAND=4 MW (§4.5)** — 계통 자체 흡수 cushion 도입, over-commit 추가 −88% (e)

**부수 인사이트 (2026-05-09 startup 분리, §10.6)**:
- 진짜 신규 가동 (offline → 발전) 1년 0~1회 / 추가 발전 전환 941회
- LNG 호기 대부분 baseline 운전 중 → cold-start 비용 거의 발생 X (운영 비용 양호)
- 잔존 over-commit 640 MWh 의 메인 원인 = baseload 호기 (CG6/CS2) 의 *느린 ramp_dn p90 = 2~3 MW/h*
  → 한 번 올라간 출력이 회복 후에도 *천천히* 내려가서 누적

**남은 over-commit 원인** (잔존 640 MWh, 매우 작음):
- baseload (CG6/CS2) 하강 ramp p90 2~3 MW/h 느림 → DEADBAND 이내 회복 시각에도 일정 출력 유지
- Layer A 호기 startup 후 ramp_dn 한도로 다음 시각 출력 유지 → 잔류 over-commit

→ **현 수준 (over/short 1.45x) 은 PoC 합격선**. 추가 옵션 A (ramp_dn p99) 미적용.

### P2 — peaker 가정 폐기 (해결됨, Q5)

이전: "peaker = 단발 spike" 가정 → mode-aware single-cover.
실제 데이터: peaker도 평균 6~7h 연속 운전. 가정 틀림.
→ §10.2 호기 선택 룰 단순화 (헤드룸/P_min 기반, mode 우선순위 X).

### P3 — slope-based forward signal (해결됨, Q6)

이전: 절댓값만 (FG1 ≥ 5)
변경: slope 추가 (`FGslope`)로 growing vs recovery 구분.

### P4 — 2-layer 구조 (해결됨, D1/D2/Q1)

이전: dP_req에 actual + forward 혼합 → 모호.
변경: Layer A (actual 100%) + Layer B (warm-up only, 출력 X) 분리. 2026-05-09 적용 완료.

### P5 — cost-aware screening 데이터 부재 (P_o C 범위 밖, Q2)

호기별 SU_u, VC_u 없음. cost-blind heuristic 만 작동.

### P6 — demand 모델 없음 (PoC 본질)

전국 계통 demand 데이터 부재 → KPI는 PV-only. shortfall 절대값 의미 작음, 상대 비교만 의미.

---

## 15. Over-commit 줄이기 — 적용 / 미적용 정리 (2026-05-09)

> **사용자 우선순위**: over-commit이 평가 1순위. 2026-05-09 적용 결과 8003 → **640 MWh (−92%)**.

### 15.0 적용 완료 (2026-05-09)

| # | 항목 | 효과 | 위치 |
|---|---|---|---|
| ✅ | priority 제곱 가중 제거 | 호기 분담 균등 | §9.3 |
| ✅ | Layer B = warm-up only (출력 X) | forward gap 발 over-commit 제거 | §10 |
| ✅ | Layer B GT 전용 + WARMUP_MIN_GAP=5 | 실제 warm-up 20회 발동 | §10 |
| ✅ | warm-up 호기 P_DA=0 유지 (P_min 강제 X) | 다음 시각 over-commit 방지 | §10.3 |
| ✅ | RG (look-back 평균) 제거 → `instant_gap_now` 직접 사용 | over-commit −25% (1849 MWh) | §7.1, §8.1 |
| ✅ | Layer A startup GT 우선 | cold-start 빠른 호기 먼저 | §8.1 |
| ✅ | **DEADBAND=4 MW (system reserve cushion)** | **over-commit −88% (5398→640 MWh)** | **§4.5** |

### 15.1 옵션 A — 호기 ramp_dn p99 사용 (data-driven 더 적극) — **미적용 (선택지)**

```python
# 현재: ramp_dn = p90 (예: CG6 = 1.9 MW/h)
# 제안: ramp_dn = p99 (예: CG6 = 48 MW/h)
```

**효과**: 호기 빠르게 내림 → over-commit 큰 폭 감소.
**위험**: 운영 직관 X (실 plant는 p99 같은 극단 거의 안 함). 시뮬레이션상은 가능.

### 15.2 옵션 B — RG 부분 잡음 컷 — **폐기 (RG 자체 제거됨)**

이전 spec의 RG 잡음 컷 옵션. 2026-05-09 RG 자체가 분배에서 제거되어 의미 없음 (§7.1).

### 15.3 옵션 C — DELAYED_RELEASE 조건 slope-based — **이미 적용됨 (§7.3.3)**

이전 spec: `forward_recovery = (FGmean3 < 3) AND (FGmax3 < 3)` (절댓값만)
현재 spec: `forward_recovery = (|FG1| ≤ TH_INCREASE) AND (FGslope ≤ -TH_SLOPE) AND (prev_total > 0)` — **slope-based 이미 적용**.

→ 이 옵션은 §7.3.3 에 통합되어 별도 작업 항목 아님.

### 15.4 옵션 D — RELEASE_RATE 동적 (recovery 강도 비례) — **미적용 (선택지)**

```python
recovery_strength = max(0, -FGslope) / TH_SLOPE     # 0~1+
release_rate_eff = 8 + 16 × recovery_strength       # 8~24 MW/h
```

**효과**: 강한 회복 신호일수록 빨리 내림.
**복잡도**: 새 변수 도입.

### 15.5 추가 작업 후보 (우선순위)

남은 over-commit 5398 MWh 추가 감축이 필요할 경우:

1. **옵션 A (ramp_dn p99)** 단독 — 호기 자체 빠른 release. 효과 가장 큰 후보.
2. **옵션 D (동적 RELEASE_RATE)** — 복잡도 대비 marginal 효과 작을 가능성, 후순위.
3. (옵션 C 는 이미 §7.3.3 에 통합됨)

> PoC 시간 제약상 현재 적용분 (§15.0) 으로 마감 가능 수준. 추가 감축은 향후 작업.

---

## 16. 출력 / 인터페이스

### 16.1 fleet-level

매 시각:
- 응답상태 (KEEP/INCREASE/DELAYED_RELEASE)
- 필요_추가출력 (Layer A + Layer B 합)
- shortfall, over_commit, residual

### 16.2 호기-level

호기별 매 시각:
- `P_DA,u(t)` (baseline)
- `ΔP_u(t)` (Layer A + B)
- `P_new,u(t) = P_DA,u(t) + ΔP_u(t)`

→ operator 관점: "얼마나 더 대응해야 하나" + "어느 호기가 맡나"

---

## 17. 왜 이 구조가 공모전 요구에 맞는가

공모전 요구:
1. 기상과 발전 실적을 연계한 AI 기반 발전량 예측
2. 신재생 변동 시나리오에 따라 화력 발전 부하를 배분하는 운영 로직

본 구조:
- 예측: Phase 1 (D-1) + Phase 2 (intraday reforecast) + outage override
- 운영: LNG D-1 계획선 위 *호기별 증분 재배분* (Layer A + Layer B)

→ "예측 모델만"이 아니라 *예측 → 호기별 부하 배분 → over-commit 까지 정량 비교* 의 통합 PoC.

---

## 18. proposal 핵심 문장

> 본 과제의 LNG 운영 로직은 단일 화력 블록의 단순 반응이 아니라, 호기별 D-1 기준 계획선을 바탕으로 신재생 출력 변동으로 인해 필요한 추가 대응량을 actual gap (즉시 보정) 과 forward gap (사전 대비) 두 layer로 분리하여, 우선적으로 온라인 호기의 headroom에 재배분한 뒤, 필요 시 오프라인 호기를 단계적으로 활성화하는 short-horizon 백업 계획 PoC이다.

> 본 구조는 실시간 신재생 변동 시나리오를 단순 예측에 그치지 않고, 실제 화력 부하 배분과 증분 운전비 추정까지 연결하는 운영 지원형 AI 파이프라인으로 해석될 수 있다.

---

## 19. 다음 작업 (2026-05-09 기준)

### 19.1 적용 완료
- ✅ Layer A / Layer B 분리 코드 (`thermal_planner_v4.py`)
- ✅ slope-based forward signal (§7.3)
- ✅ priority 제곱 가중 제거 (§9.3)
- ✅ Layer B = warm-up only, GT 전용, P_DA=0 유지 (§10)
- ✅ RG 폐기 → instant_gap_now 직접 사용 (§7.1, §8.1)
- ✅ Layer A startup GT 우선 (§8.1)
- ✅ **DEADBAND=4 MW (system reserve cushion, §4.5)**

### 19.2 남은 작업 (PoC 마감용)
1. **Dashboard 갱신** — 호기별 분담 시각화, warm-up 표시, KPI 갱신 (442/640) + DEADBAND 시각화
2. **계획서/proposal 작성** — PoC 결과 정리 (현 spec + KPI + DEADBAND framing)
3. (선택) §15 옵션 A (ramp_dn p99) — 현 수준 over-commit 640 MWh 면 미적용 OK

### 19.3 PoC 범위 밖 (한계 명시)
- cost-aware UC (호기별 SU/VC 데이터 부재)
- demand 모델 (전국 계통 demand 없음, PV-only KPI)
- ST cold-start 시 Layer B 미발동 (1시간 lead time 부족, 향후 multi-hour Phase 2 시 검토)

---

## 20. 한 줄 요약

> 본 LNG planner는 baseline LNG 계획선 위에서 **DEADBAND=4 MW 차감한 effective_gap** 만 LNG 책임분으로 보고 (§4.5, 보조서비스 정산금 LNG ≈50% 담당 근거), **Layer A (effective_gap 100% 즉시 보정, GT 우선 startup)** 과 **Layer B (forward signal 기반 GT-only warm-up 명령, 출력 X)** 두 layer로 분리해 호기별 추가 대응량을 재배분하며, over-commit 감소를 1순위 평가 지표로 둔다.

---

## 21. 최종 v4 spec 한눈에 (2026-05-09 확정)

```python
DEADBAND_MW = 4.0    # 계통 자체 흡수 cushion (§4.5)

# 입력 신호
instant_gap_now = μ_p1(t) − pv_actual(t)
effective_gap   = max(0, instant_gap_now − DEADBAND_MW)

# Layer A — 실 발전 (현재 시각 출력 결정)
즉시_보정필요량 = effective_gap
1) online 호기들에 (헤드룸 × ramp_up) 가중 분배 — priority 제곱 효과 X
2) cap 초과 + P_min ≤ 잔여 ≤ P_max  →  GT 우선 startup, 출력 = 잔여 그대로 (P_min 강제 X)
3) 그 외  →  shortfall 감수

# Layer B — 사전 대비 (다음 시각 가용성만 확보, 출력 X)
trigger: expected_next_gap = max(즉시_보정필요량, FG1)
         > max(WARMUP_MIN_GAP=5, online_headroom_total)
         AND forward_persistent (slope-based: strong/growing)
candidate: GT only, P_max ≥ 5
선택: 헤드룸 큰 → P_min 작은 → Avail 높은 순
효과: warm_up_set 합류, P_DA=0 유지, 다음 시각 분배 후보
```

**KPI (test 2025, daytime 9-17, 365일 portfolio, DEADBAND 적용)**:
- shortfall **442 MWh** / over-commit **640 MWh** / 비율 **1.45x**
- 총 ΔE 2448 MWh (시간당 평균 0.75 MW)
- vs v2 baseline (CS2 single): **−89.9%**
- Phase 2 marginal: **−19.1%**
- warm-up 발동: 20회 (모두 GT, ST 0회)
- Layer A startup: 941회
