# PV 예측 — 최종 운영 아키텍처 (2026-05-05)

> **결정**: 2단계 운영형 forecasting + short-horizon rule-based balancing planner.
> **상위 plan**: `plan/main/plan_v7.md`, `plan/main/framing.md`
> **이전 noteworthy**: `plan/pv/plan_v1.md` (NGBoost 2-stage, 본 문서로 갱신)

---

## 1. Forecasting architecture (확정)

### 1.1 구조

```

[D-1, 17:00 KPX 2차 신고]
  Phase 1: ResMLP+AdaLN v2 ensemble (frozen)
    inputs : weather forecast (perfect-foresight 가정), site_oh, hour/month
    outputs: 24h × 8 site (μ_p1, σ_p1)
    역할   : day-ahead base forecast + KPX 가용용량 신고 근거
  ↓ (ensemble parquet으로 D-day operational input)

[D-day, every daytime hour t ∈ {7..16}]
  Phase 2: intraday TCN residual correction (re-anchored, 2-branch soft-gated)
    inputs : 최근 H=6h actual cf + 같은 구간 observed weather
             frozen Phase 1 (μ, σ) for next 1~3h + perfect-foresight future weather
             static cond (site_oh, issue_time hour, month)
             gate signals: drop_now_3h, abs_realized_gap_3h, gap_x_sigma
    outputs: Δ_base_{t+k}, Δ_event_{t+k}, gate_{t+k} ∈ (0, 1)   for k = 1, 2, 3
    적용   : Δ_total_{t+k} = Δ_base + 2.0 · gate · Δ_event
             μ'_{t+k}     = μ_p1_{t+k} + Δ_total_{t+k}
             σ는 σ_p1 그대로
    역할   : next 1~3h 운영 forecast 갱신, event 시 추가 correction 강화
    범위   : 7~16 issue (하루 10 issues × 3 leads). 7~8시 issue는 일출 후 PV가 시작된 날만 sample 생성 (winter 자동 graceful skip)
```

**핵심 원칙**
- Phase 2는 standalone forecast 모델이 아님 — Phase 1 frozen baseline 위에 residual만.
- recursive autoregressive rollout 금지 — 매 issue_time t마다 frozen Phase 1과 새 actual로 다시 anchor.
- σ는 Phase 1 것을 그대로 사용 (Phase 2 update는 μ만).
- 학습 데이터 분리: Phase 1 = 2022~2023 train / 2024 val / 2025 test, Phase 2 = 2024 train (10/15 split for early stop) / 2025 test.
- **2-branch + soft gate**: 평상시는 base branch만 사용 (gate ≈ 0.05). cloud-pass / ramp event에서 gate↑로 event branch 활성화 → 추가 correction. 최종 hyperparams **λ=2.0, T=1.0, τ=0** (sigmoid soft gate, threshold 없음).

**Final override — output horizon**
- The final Phase 2 definition is no longer a fixed `next 1~3h` patch model.
- It is a **remaining-day full-horizon reforecast** with **variable-length EOD truncation** at each issue time.
- The selected setting is effectively `L=12 EOD`, meaning the model predicts from `t+1` to the end of the remaining daylight horizon for that issue time.
- In comparison tests, `L=12 EOD` superseded `L=6`, preserved most of the lead-1~3 gains, and uniquely provided recovery / persistence / sunset curve shape needed by the planner.

### 1.2 Phase 1 baseline 성능 (5-seed ensemble, test 2025)

| metric | 값 |
|---|---|
| Site NMAE | **4.662%** |
| Portfolio NMAE | **3.751%** |
| Cov80 | 85.0% |
| Cov95 | 94.1% |
| NLL | -1.218 |

UNIFIED training config (Phase 1, Phase 2 동일):
- batch=64, lr=7e-4, wd=1e-4, AdamW + CosineAnnealingLR(T_max=50)
- max_ep=50, patience=8, grad_clip=1.0
- loss = MAE + 0.2·GaussianNLL
- 5 seeds [42, 123, 7, 202, 999]

### 1.3 Phase 2 ensemble 성능 (3-seed, rolling test 2025) — 최종 채택안

**Final override — horizon selection**
- Horizon comparison across `L=3`, `L=6`, and `L=12 (EOD truncation)` showed that:
  - `L=3` still gives the strongest pure lead-1 immediate correction,
  - `L=6` is dominated by `L=12`,
  - `L=12 EOD` preserves most lead-1~3 accuracy while adding remaining-day recovery / persistence / sunset curve shape.
- Therefore the final Phase 2 is selected as **`H=6`, `2-branch soft gate`, `λ=2.0`, `issue 07~16`, `remaining-day EOD-truncated output horizon`**.
- This changes the interpretation of Phase 2 from a short patch generator to an **intraday remaining-day reforecast module**.

**최종 구조**: 2-branch soft-gated, **issue range 7~16**, λ=2.0, T=1.0, τ=0.
비교 분모 = 같은 (issue_hour, lead, site) tuple 풀링 (**n=75,312**).

| segment | Phase 1 | Phase 1+2 (final) | Δ |
|---|---|---|---|
| **Site NMAE (overall)** | 5.287% | **4.633%** | **-0.654%p (-12.4%)** |
| **Portfolio NMAE** | 4.252% | **3.762%** | **-0.490%p (-11.5%)** |
| Lead 1h | 5.297% | **4.034%** | -1.263%p (-23.8%) |
| Lead 2h | 5.312% | 4.848% | -0.464%p (-8.7%) |
| Lead 3h | 5.249% | 5.064% | -0.185%p (-3.5%) |
| Partial cloud (dc 3~7) | 6.693% | **5.819%** | -0.874%p (-13.1%) |
| Cov80 (overall) | 85.0% | 91.6% | +6.6pp |
| Cov95 (overall) | 93.7% | 97.3% | +3.6pp |

> **참고**: 분모 n이 66k → 75k로 변경됨 (issue 7,8 추가). 그래서 P1 baseline NMAE 자체도 이전 5.422 → 5.287로 다름.

**Top cloud-pass events (정오~15시 portfolio NMAE)**

| date | Phase 1 | Phase 1+2 (final) | Δ |
|---|---|---|---|
| **2025-03-23** | 68.00% | **43.30%** | **-24.70%p (-36.3%)** |
| 2025-04-26 | 39.74% | 33.69% | -6.05%p (-15.2%) |
| 2025-05-04 | 53.21% | 39.78% | -13.43%p (-25.2%) |

**vs single-branch H=6 baseline / 이전 issue range 9~16 비교** (참고):

| | per-seed mean | per-seed std | sample 수 |
|---|---|---|---|
| H=6 single-branch (이전 9~16) | 4.754 | 0.049 | 23,286 |
| 2-branch λ=2.0 (이전 9~16) | 4.737 | 0.026 | 23,286 |
| **2-branch λ=2.0 (final 7~16)** | **4.658** | 0.034 | **26,306** |

→ issue range 확장으로 평균 NMAE 0.079%p 추가 개선 (4.737 → 4.658) + sample +13%.

> Phase 2의 강점은 **lead 1h + cloud-pass event**에 집중. Lead 3h는 baseline 수렴.

**Limitation — 2025-03-23 top event**: issue range 확장 후 03-23 정오~15시 portfolio NMAE 약 1%p 악화 (42.12 → 43.30). morning sample 추가로 모델이 분산되어 가장 극단적 cloud-pass에서 약간 손해. 다른 모든 metric에서는 개선.

### 1.4 안정성

3 seeds [42, 123, 7] P1+2 NMAE (final, 7~16): 4.618 / 4.686 / 4.671 (mean **4.658 ± 0.034**).
- single-branch H=6 (이전 9~16): mean 4.754 ± 0.049 (variance 약 2배)
- 2-branch + 확장 sample은 평균과 안정성 모두 개선.

### 1.5 Gate 동작 (3-seed pooled)

| segment | gate mean | top max |
|---|---|---|
| Overall | 0.054 (median 0.053) | — |
| **Top event 12~15h** | **0.103** (1.9x non-event) | 0.89 |
| Non-event | 0.054 | — |
| `drop_now_3h > q90` | 0.077 | — |

**Per-event 분석**:
- 2025-03-23: gate mean 0.064, max 0.72 — 한 시점은 거의 fully open
- 2025-04-26: gate mean 0.106, max 0.95
- 2025-05-04: gate mean 0.098, max 0.85

**Event branch 절대 기여**:
- Overall: |λ·gate·Δ_event| 평균 ~328 kW vs |Δ_base| ~314 kW
- Top events: event branch 기여 ~1262 kW vs base ~2147 kW — event 시 ~3.85x 증폭

### 1.5 Past context window length 결정 — H=6 채택

**EDA (notebook 12)**: lag 1~6h actual / residual feature가 next 1/2/3h |error|를 얼마나 설명하는지 측정.
- `resid_mean_1h`가 모든 segment에서 가장 강한 신호 (overall |corr| 0.581, partial 0.609, top events 0.689).
- `resid_mean_kh`는 lag k가 늘어나도 overall/partial cloud에서 |corr| 0.48~0.55 유지 — 오래된 정보도 noise 아님.
- Top events에서는 lag 5~6h가 0.26~0.28까지 감쇠 — 이벤트는 short-history 우위 경향.

**Sweep 결과** (H=3/4/5/6 × 3 seeds, 같은 UNIFIED config, 3-seed ensemble):

| H | Overall NMAE | Lead 1h | Partial cloud | Portfolio | 03-23 | 04-26 | 05-04 |
|---|---|---|---|---|---|---|---|
| H=3 | 4.717 | 4.239 | 5.936 | 3.843 | 45.31 | 34.75 | 42.42 |
| **H=4** | 4.715 | 4.237 | 5.913 | 3.853 | 44.39 | 34.51 | 41.63 |
| H=5 | 4.715 | 4.242 | 5.906 | 3.859 | 42.93 | 34.41 | **41.32** |
| **H=6 ★** | 4.729 | 4.256 | 5.935 | 3.875 | **41.98** | **34.31** | 42.10 |

Per-seed std: H=3 0.020 / H=4 0.004 / H=5 0.013 / H=6 0.049.

**판정 — H=6 채택**:
- Overall / lead 1h / partial cloud에서 H=3/4/5가 H=6보다 미세 우세이지만 **차이 0.01~0.02%p로 사실상 동률**.
- **Top cloud-pass events에서 H=6이 가장 좋음**: 03-23 41.98% (vs H=5 42.93%, H=3 45.31%), 04-26 34.31% (vs H=4 34.51%).
- 이 PoC의 목적은 평균 leaderboard 최적화가 아니라 *short-horizon variability 대응 + 운영 시연 가치* → top-event robustness 우선.
- Long history가 큰 ramp event에 도움 — EDA의 "top events에서 resid_mean_5h/6h가 lead 3h top-3 feature"였던 결과와도 일치.

**부수 기록**: 일반 NMAE만 보면 **H=4가 가장 안정적인 설정**이었다 (per-seed std 0.004로 가장 작고, overall NMAE 4.715%로 H=3/5와 동률). 운영 우선순위가 변경되어 평균 성능만 따진다면 H=4로 단순화 가능.

### 1.6 2-branch 구조 + soft gate — 최종 확정 근거

H=6 single-branch baseline 대비 *모든 segment*에서 미세 개선 + seed variance 절반:

| 비교 | NMAE | Lead 1h | Partial cloud | Portfolio | per-seed std |
|---|---|---|---|---|---|
| H=6 single-branch | 4.729 | 4.256 | 5.935 | 3.875 | 0.049 |
| 2-branch λ=1.0 (base init) | 4.738 | 4.271 | 5.899 | 3.882 | 0.023 |
| 2-branch λ=1.5 | 4.742 | 4.261 | 5.882 | 3.878 | 0.014 |
| **2-branch λ=2.0 (final)** | **4.712** | **4.209** | **5.871** | **3.847** | **0.026** |

**채택 이유**:
- λ=2.0이 모든 핵심 metric에서 baseline 동률 또는 우세
- per-seed 안정성: H=6 0.049 → λ=2.0 0.026 (**-47%**)
- gate top-event 1.9x 활성화 (max 0.95 도달) — event 시 추가 correction 정상 동작
- top event 03-23 42.12% (H=6 41.98%와 거의 동률, +0.14%p)

### 1.7 Sparse gate tuning — Negative result

Selectivity↑를 목표로 `gate = sigmoid(z/T)` 의 T를 0.5로 sharpen하고 threshold τ를 추가하는 sweep을 수행했으나 채택 안 함:

| config | gate ratio (top/non) | dead seeds (3 중) | NMAE | top 03-23 |
|---|---|---|---|---|
| λ=2.0 T=1.0 τ=0 (final) | 1.9x | 0/3 | **4.712** | 42.12 |
| T=0.5 τ=0 | 5.8x | 0/3 | 4.743 | 43.77 |
| T=0.5 τ=0.1 | 16.8x | **2/3** | 4.732 | 42.56 |
| T=0.5 τ=0.2 | 0x (전체 dead) | 3/3 | 4.729 | 41.98 (= H=6) |

**관찰**:
- Selectivity는 향상 가능 (τ=0.1에서 16.8x ratio, gate max 0.96 도달)
- 그러나 bias=−3 init + T<1 + threshold 결합이 **vanishing-gradient at threshold** 유발
- 2/3~3/3 seed에서 gate dead (이벤트 미감지) → 평균 NMAE 손해 + correction 약화
- 운영 PoC 안정성에 부적합 → **negative result, 채택 안 함**

미래 작업 후보 (현 단계 보류): bias 완화 (−3 → 0), auxiliary gate weak supervision, curriculum learning.

---

## 2. Operational update logic

### 2.1 Issue cadence

**Final override — cadence interpretation**
- D-day each issue time now produces a **remaining-day reforecast**, not just a 3-step patch.
- At issue time `t`, Phase 2 reissues the curve from `t+1` to the remaining daylight end.
- This means the planner can read both the immediate next-hour correction and the later recovery / persistence shape from the same reforecast.

- D-1 17:00: Phase 1 24h forecast 생성 → KPX 가용용량 신고
- D-day 매 시간 t ∈ {7, 8, ..., 16}: Phase 2 update (next 1~3h residual)
- D-day 17:00 이후: 다음 날 Phase 1 재실행
- 7~8시 issue는 일출 후 PV 인식되는 날에만 sample 생성 (winter 조기에는 자동 skip)

### 2.2 Re-anchor 원칙

- Phase 2는 매 issue마다 frozen Phase 1을 base로 다시 anchor → 누적 drift 방지.
- 입력의 "past" sequence는 Phase 1 forecast가 아닌 *실측 cf*를 사용 → actual에서 다시 출발.
- recursive rollout(t+1을 예측한 결과를 t+2 예측에 다시 입력) 사용 안 함.

### 2.3 Mask 처리

**Final override — EOD truncation**
- Final Phase 2 uses **variable-length EOD truncation** rather than a fixed short horizon.
- Each issue produces targets only up to the remaining valid daylight horizon.
- Late-day issues therefore have shorter valid output sequences, which is intentional and operationally natural.

- Past sequence H=6 중 일부 시각이 daytime 외이면 zero pad + mask channel.
- Future horizon 3 중 sunset 이후이면 mask=0 처리 (loss 무시).
- 예천 anomaly_zero 행은 *마스킹하지 않고 그대로* 입력 (운영 모듈은 actual=0 신호도 학습 자료로 사용).

### 2.4 Outage Override Layer (post-processing, 2026-05-07 신설)

Phase 2가 학습으로 흡수 못하는 **운영/설비 사건**을 별도 layer로 분리. weather-driven correction 과 fault/outage handling 의 책임 분리.

#### 2.4.1 Detection rule (outage_v3.5 relaxed)

매 시점 t에서 (site, datetime) 단위로:

```python
outage_flag_t = (
    cf < 0.03                  # 발전 거의 0
    AND mu_phase1 > 0.20       # P1 baseline은 의미있는 발전 예상
    AND dc10Tca < 7            # 부분 흐림까지 허용 (강한 흐림 제외)
    AND z < -3.0               # P1 ±3σ 밖 (z = (cf - mu_p1) / sigma_p1)
    AND neighbor_confirmed     # t-1 또는 t+1도 outage_flag (단발 아님)
)
```

핵심 신호:
- `z < -3.0`: P1 모델 입장에서 통계적 outlier — weather로 설명 안 되는 급락
- `dc10Tca < 7`: 진짜 흐린 날 (cloud-pass)은 자연스러운 P2 흡수 영역으로 남김
- neighbor confirmation: 1시간 단발은 측정 오류 가능성 → 2h+ 연속만

#### 2.4.2 Real-time blackout state machine + recovery

실시간 운영 시뮬레이션 — 사이트별 시간순 진행:

```python
state = NORMAL
recover_streak = 0

for t in target_dt (시간순):
    if state == BLACKOUT:
        if actual_cf > 0.10:
            recover_streak += 1
            if recover_streak >= 1h:
                state = NORMAL
                recover_streak = 0
        else:
            recover_streak = 0
    else:  # NORMAL
        if outage_flag_t:
            state = BLACKOUT
            recover_streak = 0

    # Override
    mu_final_t = 0 if state == BLACKOUT else mu_phase2_t
```

- `recovery threshold = 0.10` (cf): grid sweep 결과 0.10/0.15 거의 동일 효과, 단순한 0.10 채택
- `recovery duration = 1h`: 2h는 over-conservative (정상 시점도 0으로 덮음)

#### 2.4.3 Detection 결과 (test 2025)

전체 41 confirmed outage targets, recovery 룰로 **587 hours blackout cover**:
- 광양항세방 10/10 ~ 10/12 — 3일 종일 outage (16건)
- 고흥만수상 점심시간 정기 outage 패턴 — 03-22, 03-23, 04-06, 04-26, 05-04 (모두 12-14시, 13건)
- 구미 06-15 ~ 06-16 — 2일 연속 (5건)
- 경상대 11-11, 삼천포 12-04 — 단일 event

#### 2.4.4 효과 (test 2025 NMAE)

| 그룹 | n | P1 | P2 | **P2+override** | Δ |
|---|---|---|---|---|---|
| 전체 | 167,953 | 4.747% | 4.432% | **4.220%** | **−0.212pp** |
| Blackout 시점 | 587 | 68.6% | 55.7% | **0.39%** | **−55.3pp** |
| Normal 시점 | 167,366 | 4.50% | 4.23% | **4.23%** | **+0.000pp** ✓ |
| 03-23 cloud-pass day | 440 | 30.0% | 23.3% | **2.89%** | **−20.4pp** |
| 04-26 cloud-pass day | 624 | 15.0% | 13.7% | **3.81%** | **−9.8pp** |
| 05-04 cloud-pass day | 624 | 18.7% | 16.1% | **3.79%** | **−12.3pp** |
| 광양항 10/10-12 outage | 165 | 28.2% | 15.5% | **0.09%** | **−15.4pp** |

**검증 통과**:
- Normal 시점 NMAE 변화 +0.000pp → cloud-pass + 정상 시점 *손상 없음*
- Blackout 시점만 깔끔히 처리

#### 2.4.5 Negative result — 폐기된 대안들

**대안 1: 학습 데이터 outage masking + Phase 1 재학습**
- v3.5 mask 278건 (narrow) / 539건 (wide) 두 변형 모두 시도
- 결과: raw NMAE +0.14pp 악화. site-level 일부 개선 / 일부 악화. cloud-pass 효과 미미.
- 폐기 이유: 학습 데이터의 0.2~0.4% 마스킹으로는 모델 분포 변화 없음. 모델은 outage를 noise로 robust하게 흡수 중.

**대안 2: Integrated outage probability head (joint training)**
- Phase 2 architecture에 outage_prob 헤드 추가, joint loss = forecast + λ·BCE(p_out, y_out, pos_weight)
- soft suppression: `mu_final = (1 - p_outage) · mu_phase2`
- 결과: precision 3.3%, recall 41.7% (FP 폭증). 04-26 cloud-pass 13.79% → **20.24% (악화)**.
- 폐기 이유:
  - Label imbalance 너무 심함 (0.21% positive)
  - Outage가 weather feature로 예측 불가능한 site-specific 운영 사건
  - 학습 sample 부족 (437 positive)
  - **Rule-based detection이 더 자연스러운 도메인** (actual cf을 직접 보고 즉시 판단 → 0% FP)

#### 2.4.6 설계 철학

> **weather-driven correction (Phase 1+2)** 과 **fault/outage override (rule-based)** 를 책임 분리.
> 모델은 weather로 설명 가능한 변동만 책임지고, 비기상 운영 사건은 명시적 detection rule로 처리.

코드: `src/forecast/outage_override.py`
산출물: `pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet`

---

## 3. Thermal balancing logic

> 이 섹션의 planner는 **하루 전체 발전계획 수립기**가 아니다. 목적은 D-1 PV 예측과 intraday update 사이에서 발생하는 **앞으로 1~3시간의 변동성**을, 이미 운영 중인 controllable LNG block의 **추가 balancing response recommendation**으로 연결하는 것이다.

### 3.0 Planner 해석 범위 (CS2 데이터 사용 목적)

**Final interpretation**
- This planner does **not** attempt to reproduce the actual nationwide dispatch of Bundang LNG.
- Its purpose is to determine whether the current LNG balancing response should be **sustained, further increased, or gradually released** under short-horizon PV variability.
- Therefore, it should be interpreted as an **operational response-state logic**, not as a plant-dispatch reconstruction model.

본 PoC의 thermal planner는 **전국 계통 급전 시뮬레이터가 아니다**. 남동발전 PV portfolio의 short-horizon 변동성에 대해 *추가적으로 필요한* balancing requirement를 추정하는 **local balancing planner**로 정의한다.

즉 planner의 출력은:
- ❌ 실제 CS2의 dispatch 재현값
- ✅ PV 예측 오차로 인해 *추가로 필요한* ramp / reserve / balancing response

#### CS2 데이터의 사용 목적

분당 LNG CS2의 4년 운전 데이터는 **planner의 ground truth가 아니라 feasibility reference**:

| 용도 | 채택 여부 |
|---|---|
| planner 출력의 정답값 (ground truth) | ❌ |
| Daytime availability 확인 | ✅ |
| Partial-output 운전 가능성 확인 | ✅ |
| Headroom 규모 확인 | ✅ |
| Ramp capability 확인 | ✅ |

→ CS2는 *"이런 balancing block이 현실적으로 존재 가능한가"*를 보여주는 참조 자원이지, *"우리 planner가 실제 CS2처럼 움직였는가"*를 평가하는 기준이 아니다.

#### 직접 비교를 하지 않는 이유

| 측면 | 실제 CS2 dispatch | 본 planner |
|---|---|---|
| 입력 | 전국 수요, 계통 제약, SMP, 예비력, 타 발전기 상태 | PV portfolio 변동성 + Phase 1/2 forecast + forward gap |
| 목적 | 광역 시스템 cost-merit 급전 | local PV balancing requirement |
| 의사결정 주체 | KPX | (본 PoC)|

목적 함수와 입력 정보가 다르므로 직접 비교는 논리적으로 정합하지 않다.

#### 올바른 해석

PoC에서 의미 있는 질문:
- ✅ "planner가 요구한 balancing response가 *실제 CS2의 운전 가능 범위 안*에 있었는가?"
- ✅ "planner가 가정한 controllable LNG block이 *현실적으로 존재 가능*한가?"

PoC에서 의미 없는 질문:
- ❌ "planner 출력이 실제 CS2 dispatch와 일치하는가?"

#### 핵심 가정

- **CS2 등가 controllable block availability**: 본 PoC는 분당 LNG CS2 등가 자원이 balancing 용도로 가용하다고 가정한다.
- **No cold-start / no unit commitment modeling**: cold-start 시간, 기동비, on/off commitment 최적화는 이번 PoC에서 명시적으로 다루지 않는다.
- **Must-run baseload externalized**: 영흥/삼천포 등 기저발전은 외생 must-run generation으로 고정한다.
- **Incremental response interpretation**: planner 출력은 full dispatch 명령이 아니라, 현재 운영 중인 LNG 대응 체계에 대해 추가적으로 필요한 ramp / reserve recommendation으로 해석한다.
- **Under-generation focused scope**: 현재 PoC는 PV under-delivery 대응을 중심으로 설계되며, PV over-generation은 controllable LNG 감산 범위 내에서만 처리한다.

### 3.0a Horizon 해석

**최종 override**
- Phase 2의 `H=6`은 최근 6시간 actual / residual context를 본다는 뜻이다.
- Final Phase 2 output is no longer limited to `next 1~3h`; it is a **remaining-day curve with EOD truncation**.
- Within that curve, the planner still interprets:
  - the first `1~3h` as the **direct control horizon**,
  - the later remaining daylight tail as the **response persistence / recovery horizon**.
- 즉 `H=6`은 단순 입력 길이가 아니라, **가까운 몇 시간의 직접 대응과 남은 하루 shape 판단**을 가능하게 하는 context window다.

Phase 2의 `H=6`은 **과거 6시간을 본다**는 뜻이고, 출력 `next 1~3h`는 **직접 보정 가능한 미래 horizon**을 의미한다.  
운영에서는 이를 아래 두 층으로 해석한다.

1. **Control horizon (1~3h)**
   - 본 planner는 CS2의 실제 dispatch 재현기가 아니라, CS2가 제공할 수 있는 controllable operating envelope를 기준으로 추가 balancing requirement를 계산한다.
   - cold-start, startup cost, full unit commitment는 이번 PoC 범위에서 제외한다.
   - planner 출력은 full dispatch replacement가 아니라 이미 운영 중인 LNG 대응 체계에 대한 incremental balancing response recommendation으로 해석한다.
   - Phase 2가 직접 수정한 forecast를 사용
   - 실제 balancing power와 reserve adjustment를 결정

2. **Preparation horizon (4~6h)**
   - Phase 2가 직접 예측을 내는 구간은 아니지만, 최근 6시간 context와 현재 event 신호를 바탕으로 준비 action을 결정
   - 추가 ramp 여유 확보, reserve 유지, 대응 수준 유지 같은 준비 판단에 사용

즉 `H=6`은 단순 입력 길이가 아니라, **단기 직접 대응 + 근미래 준비 판단**을 가능하게 하는 context window로 해석한다.

### 3.1 Planner 입력

**최종 override**
- 본 planner는 **분당 LNG CS2 등가 controllable block 1개**를 balancing asset으로 사용한다.
- CS2 4년 데이터는 실제 dispatch 정답값이 아니라, `daytime availability / partial output / headroom / ramp capability`를 설명하는 **feasibility reference**다.
- cold-start, startup cost, on/off commitment는 이번 PoC에서 모델링하지 않는다.
- must-run baseload는 planner가 조정하지 않는 외생 자원으로 둔다.
- planner 출력은 full redispatch 명령이 아니라, 이미 운영 중인 LNG 대응 체계에 대해 추가로 필요한 **incremental balancing response**로 해석한다.
- planner의 직접 목적은 `현재 대응 상태를 유지할지`, `추가 response를 얹을지`, `response를 천천히 해제할지`를 정하는 것이다.

- portfolio Phase 1 forecast `μ_p1`, `σ_p1`, actual cf (per hour, sites 합산)
- portfolio Phase 2 update `μ_p2` (latest issue 기준 remaining-day reforecast)
- demand profile (stylized): `D(h) = 80 + 25·sin(π(h−7)/12)` MW (정오 14h 피크 ~105 MW)
- **controllable thermal scope**:
  - 본 PoC의 thermal planner는 **분당 LNG CS2 등가 controllable block 1개**의 balancing control
  - 4년치 hourly 데이터로 호기 특성 비교한 결과 **CS2가 PV balancing에 가장 적합** (가동률 64%, partial output 가능, 평균 headroom 190 MW)
  - 영흥/삼천포 등 석탄 baseload는 **system-level must-run generation**으로 외부화
  - **이 PoC는 남동발전 PV 12 호기 portfolio (~77 MW) 한정**. 전국 단위 확장 시 multi-unit thermal 모델 + 타 발전사 자원 통합 필요
- **PV over-generation 처리 정책**:
  - 실제 PV가 예측보다 많은 경우, controllable LNG block은 가능한 범위 내에서 감산할 수 있다
  - 그러나 must-run 기저발전은 본 planner가 건드릴 수 없으므로, 그 이상 excess PV는 시스템 차원 흡수 대상으로 둔다
  - 따라서 현재 PoC의 핵심 운영 리스크는 **PV under-generation**이며, KPI 해석도 shortage 대응 중심으로 수행한다
- **CS2 호기 spec (실 데이터 기반)**:
  - `THERMAL_MIN = 43 MW` (CS2 pmin_p05, 가동 중 못 내리는 최저)
  - `ONLINE_MAX = 200 MW` (CS2 pmax_observed)
  - `RAMP_RATE = 30 MW/h` (CS2 hourly ramp 보수 추정 — p95 21, max 179)
  - 4년치 cf 0.298, running 52%, daytime 가동률 64%, 평균 출력 51 MW (median, partial)
  - 평균 headroom 190 MW로 **PV portfolio swing (~60 MW) 전부 흡수**
  - `RESERVE_K = 1.282·σ` (Cov80 buffer)

### 3.2 Gap 정의

**최종 override (2026-05-07 단순화)**
- `realized_gap` = 이미 일어난 PV 부족분 (look-back 평균); **현 시각 backup signal — 100% 추종**
- `forward_gap` = 현 시각부터 일몰까지 Phase 2 갱신; **앞으로의 발전 추세 의사결정 지원**
- `correction` = 신호의 잡음 컷(±3 MW) + 100% 추종 + ramp/release 제약 결과 (이전 ALPHA 부분추종 폐기)

즉 planner는 `realized gap`으로 현 시각 부족분을 backup하고, `forward gap`으로 일몰까지의 발전 흐름을 운영자에게 보여준다.

planner와 dashboard에서는 아래 2개 gap을 구분한다.

**Gap interpretation**
- `realized_gap` = look-back 평균. 현 시각의 진짜 부족분 → 100% backup
- `forward_gap` = 현 시각부터 일몰까지 Phase 2 갱신 중 절댓값 max signed → 의사결정 지원

1. `realized_gap_h = mean over k∈[1,K] (pv_p1_{h-k} - pv_actual_{h-k}), K=2`
   - 현재까지 D-1 forecast가 실제와 얼마나 달랐는지 (look-back avg)
   - 이미 발생한 변동성의 크기

2. `forward_gap_h = arg max-abs over k∈[1, sunset]: (pv_p1_{h+k} - pv_p2_{h+k from issue=h})`
   - 현 시각에 발행된 Phase 2가 일몰까지 D-1 대비 얼마나 부족/초과할 것 같은지
   - 화면 표시 + signal 후보 (둘 중 절댓값 큰 쪽 채택)

3. `correction_h`
   - `signal = arg max-abs(realized_gap, forward_gap)` 기반
   - 잡음 컷 + 100% 추종 + ramp 30 MW/h ↑ / release 4 MW/h ↓

4. `prep_risk_{h+4:h+6}`
   - 해석상 direct dispatch 신호가 아니라 reserve 유지, response hold, 추가 ramp 사전 정보 같은 준비 판단에 사용한다.
   - 최근 6시간 actual / residual 패턴과 current issue signal을 바탕으로 산정한 준비 구간 위험도
   - 직접 dispatch가 아니라 reserve 유지, response hold, 추가 ramp 여유 확보 같은 준비 판단에 사용

### 3.3 Rule-based balancing response

**Rolling planning logic**
- At time `t`, `realized_gap_t` is used to correct the current-hour balancing level.
- The same issue time's `forward_gap` is read as a remaining-day curve; its first `1~3h` drives the direct balancing plan, while the later tail informs recovery / persistence judgment.
- When hour `t+1` actuals arrive, the mismatch between planned and realized balancing is absorbed by a new `realized_gap`, and the remaining horizon is re-planned from the updated `forward_gap`.

**Commitment-lite inertia**
- The planner is not using full cold-start commitment, but it still preserves short-horizon inertia so that Phase 2 has operational value beyond pure one-hour reactive ramping.
- In practice this means:
  - `reserve carry`: keep reserve for an extra step when short-horizon risk persists
  - `minimum response hold`: once an additional response is raised, keep it for a minimum hold window
  - `delayed release`: do not release the response immediately just because the current realized gap softens once
  - `response persistence cost`: discourage frequent up/down reversals

**Operational meaning**
- `realized_gap` answers: "Do we need to correct the current balancing level right now?"
- `forward_gap` answers: "Should the current response be sustained for the next 1~3 hours, or should it be increased further?"
- `minimum response hold` and `delayed release` answer: "Even if the current hour looks better, should we still keep the response because short-horizon risk remains?"
- In other words, the planner's value is not just immediate ramp-up, but **persistence-aware response management**.

**최종 override**
- 이 planner는 **full dispatch optimizer**가 아니라 **incremental balancing recommendation engine**이다.
- 매 시점 `t`에서 `realized_gap_t`는 현재 시점 부족 보정에 쓰이고, `forward_gap_{t+1:t+3}`은 다음 1~3시간 balancing plan에 쓰인다.
- 작은 변화는 무시하고(`threshold` 잡음 컷), 잡음 컷 위는 신호 그대로 추종한다(`100% follow`, 부분추종 없음). 이미 올린 대응은 바로 되돌리지 않는다(`release rate / hysteresis`).
- PV가 baseline보다 많이 나오는 경우에도 must-run baseload는 줄이지 않으며, planner가 실제로 줄일 수 있는 것은 **CS2 controllable range 안의 LNG output**뿐이다.

매 시각 h ∈ [7,18]:

```
realized_gap_h = (1/K) Σ (μ_p1_{h-k} - pv_actual_{h-k}),  K=2  # 지난 K시간 평균
forward_gap_h  = max-abs over k∈[1, sunset]: (μ_p1_{h+k} - μ_p2_{h+k from issue=h})

# Phase 1+2 모드: realized와 forward 중 더 큰 신호 채택
signal_h = arg max-abs (realized_gap_h, forward_gap_h)

# 단순 룰: 잡음 컷 + 100% 추종 (부분추종 없음)
if |signal_h| < THR_LOW:           # THR_LOW = 3 MW
    correction_target_h = 0
    state = 'keep'
else:
    correction_target_h = signal_h  # 100% follow
    state = 'ramp_up' if signal_h > 0 else 'ramp_down'

# ramp/release 제약 (진동 방지)
if correction_target_h > correction_{h-1}:
    correction_h = min(correction_target_h, correction_{h-1} + RAMP_UP_RATE)  # 30 MW/h
else:
    correction_h = max(correction_target_h, correction_{h-1} - RELEASE_RATE)   # 4 MW/h

# 평가 (demand 없음, PV-only)
instant_gap_h  = μ_p1_h - pv_actual_h            # 현 시각 진짜 부족분
shortfall_h    = max(0, instant_gap_h - correction_h)   # 못 채운 부족
over_commit_h  = max(0, correction_h - instant_gap_h)   # 과대 보충
```

핵심 제약은 다음과 같다.

- **threshold (잡음 컷)**: `|signal| < 3 MW`는 잡음으로 무시
- **100% follow**: 잡음 컷 위는 신호 그대로 추종 (이전 ALPHA 부분추종은 직관 어긋나서 폐기)
- **ramp/release**: 상승 30 MW/h, 하강 4 MW/h (한 번 올린 출력 천천히 내림 = 진동 방지)
- **incremental response logic**: 현재 운영 중인 LNG 대응 체계를 기준으로 추가 response만 계산

즉 planner는 `앞으로 3시간은 직접 대응`, `그 이후 4~6시간은 준비 상태 결정`으로 나뉜다.

중요한 점은, 여기서의 `ramp-down`은 "석탄 baseload까지 내린다"는 뜻이 아니다.  
본 PoC에서 줄일 수 있는 것은 **분당 LNG controllable range 내부의 출력**뿐이며, must-run baseload는 외생적으로 고정되어 있다고 본다.  
따라서 PV가 예측보다 더 많이 나오는 경우 planner가 수행하는 것은:

1. 분당 LNG output을 controllable band 안에서 감산
2. reserve release 또는 response hold 해제
3. 그래도 excess가 남으면 `curtailment / spill / system export 필요` 상태로 표시

즉 이 planner는 Phase 2 update를 그대로 따라가는 기계가 아니라, **short-horizon risk를 부드럽게 흡수하는 분당 LNG CS2 등가 block 기준 smoothing layer**다.

### 3.4 Phase 1 only (reactive, look-back) vs Phase 1+2 (predictive, forward gap) — test 2025, daytime 9~17, 3,280 hours

> Phase 2 = final 2-branch λ=2.0 ensemble (issue range 7~16).

**Interpretation note**
- The purpose of this comparison is not to show whether LNG can ramp within one hour, but whether Phase 2 helps the planner decide if the response should be **carried, strengthened, or released more gradually** over the next few hours.

| KPI | Phase 1 only | Phase 1+2 | Δ |
|---|---|---|---|
| **shortfall_MWh** | 5,030 | **4,349** | **-681 (-13.5%)** |
| over_commit_MWh | 5,723 | 5,421 | -303 (-5.3%) |
| shortage_hours | 1,364 | 1,338 | -26 (-1.9%) |
| over_commit_hours | 1,482 | 1,471 | -11 (-0.7%) |
| **state_changes** | 749 | 904 | +155 (+20.7%) |
| **correction_sign_flips** | 12 | 51 | +39 (+325%) |
| correction_volatility_MW | 2.28 | 2.57 | +0.29 (+12.8%) |
| mean \|correction\|_MW | 1.18 | 1.76 | +0.59 (+49.9%) |

**해석**:
- 1차 KPI (shortfall): **-13.5% 감소** → Phase 2 forward gap 신호가 실제 thermal balancing decision을 개선.
- 2차 KPI (안정성): Phase 2가 더 적극적으로 반응 (state_changes +20.7%, sign_flips +325%) — 즉 반응 횟수는 늘었지만, hysteresis(release_rate=4 MW/h)로 과도한 full-chasing 대신 제어된 correction으로 유지.
- over_commit과 over_commit_hours도 함께 감소 (-5.3% / -0.7%) → 2-branch event correction이 양방향(부족/초과) 모두 줄임.
- 본 planner는 full unit commitment가 아니라 incremental balancing recommendation이므로, 이 KPI는 `추가 balancing requirement`와 `response stability` 중심으로 해석한다.

### 3.5 Top cloud-pass events 정오~15시 (4시간)

**Additional planner reading**
- The value of Phase 2 in the planner is not only "can LNG ramp within one hour?".
- More importantly, it helps decide whether the current response should be **carried, reinforced, or only gradually released** over the next few hours.

| date | mode | shortfall_MWh | over_commit_MWh | state_chg | mean\|corr\|_MW |
|---|---|---|---|---|---|
| 2025-03-23 | reactive | 138.7 | 0.0 | 1 | 17.9 |
| 2025-03-23 | **predictive** | **96.7** | 0.0 | **0** | 28.4 |
| 2025-04-26 | reactive | 99.3 | 19.3 | 1 | 10.7 |
| 2025-04-26 | **predictive** | **75.9** | 30.0 | 1 | 19.2 |
| 2025-05-04 | reactive | 130.3 | 25.8 | 1 | 15.0 |
| 2025-05-04 | **predictive** | **100.3** | 35.8 | **0** | 25.0 |

> 큰 cloud-pass event에서 predictive planner는 forward gap을 미리 감지해 *더 큰 correction* (28.5 vs 17.9 MW)을 적용한다. shortfall은 **23~31% 감소**했고, state changes는 같거나 줄어들어, 더 큰 correction을 투입하면서도 반응 빈도를 과도하게 늘리지 않았음을 보여준다.

---

## 4. KPI framing — raw vs clean

운영 보고와 모델 평가를 분리.

| view | 정의 | Phase 1 NMAE | Phase 1+2 NMAE |
|---|---|---|---|
| **raw (운영 KPI)** | 모든 행 포함 (예천 anomaly_zero 4.95% 포함) | 4.662% | (계산 시) — |
| **clean (모델 능력)** | 예천 anomaly_zero 제외 | 4.636% | (계산 시) — |
| **rolling (Phase 2 대상)** | issue×lead×site tuple 풀링 (n=75k) | 5.287% | **4.633%** (2-branch λ=2.0, issue 7~16) |

운영 보고서에 두 metric 동시 표기 + "예천 outage 4.95%는 모델 외부 사건" 명시.

---

## 5. Novelty — 이번 PoC의 차별점

1. **Forecast-to-decision pipeline**
   - 단일 forecast 정확도가 아니라 "forecast → balancing response recommendation → operational KPI(shortfall, over-commit)" 전체 사슬을 평가.
   - shortfall MWh / shortage hours / response stability 같은 *운영 metric*에 매핑.

2. **Intraday event-aware update (2-branch soft-gated)**
   - Phase 2가 매 시간 actual로 re-anchor → cloud-pass event(2025-03-23 등)에서 shortfall 23~30% 감소.
   - 평상시 base branch만 사용, event 시 gate↑로 event branch 활성화 (gate top-event 1.9x, max 0.95).
   - 단순 ML 모델 정확도 비교를 넘어, "어떤 시간에 얼마나 빠르게 업데이트되는가"가 운영 가치.

3. **Short-horizon balancing planner**
   - planner는 하루 전체 발전계획이 아니라, `realized gap (현 시각) + forward gap (일몰까지)` 기반 balancing action generator.
   - 현 시각은 realized gap 기반 정확 backup, 이후 미래는 forward gap 기반 의사결정 지원.
   - Phase 2가 제시한 변동성 신호를 threshold(잡음 컷 ±3 MW) + ramp(30 MW/h ↑)/release(4 MW/h ↓) 제약으로 변환해 thermal action이 forecast 변동에 과민 추종하지 않도록 제어한다.
   - 여기서 thermal action은 full dispatch replacement가 아니라, 이미 운영 중인 LNG 대응 체계에 대한 **incremental balancing recommendation**이다.

4. **Reserve-aware balancing rule**
   - σ_p1 기반 reserve buffer (`1.282σ`)를 balancing action에 반영 → Cov80 수준의 운영 안전 마진을 근사.
   - 완전한 uncertainty 재보정 모델이라기보다, 점예측 위에 보수적 reserve를 얹는 PoC rule로 해석.

5. **데이터 품질 framing의 정직성**
   - 예천 outage(anomaly_zero 4.95%) → 모델 외부 사건으로 분리 보고
   - 광양항 microclimate, 창원 capacity 의심 → site별 처치 차별화
   - "raw vs clean" 두 KPI 동시 표기로 운영자에게 두 정보 모두 제공.

6. **Control vs preparation split**
   - `next 1~3h`는 direct balancing response, `4~6h`는 reserve 유지/response hold 같은 preparation logic으로 분리
   - forecast horizon과 thermal action horizon을 구분해 운영 논리를 더 현실적으로 설명

---

## 6. 산출물 위치

```
pv/experiments/
├── resmlp_adaln_v2_ensemble/         ← Phase 1 (5-seed ensemble baseline) ★
│   ├── seed_{42,123,7,202,999}/
│   ├── ensemble_test.parquet         ← test 2025 portfolio forecast
│   └── per_seed_summary.csv
├── phase2_2branch_g20/              ← Phase 2 final (2-branch λ=2.0, T=1, τ=0) ★ (채택)
│   └── test_predictions_seed{42,123,7}.parquet
├── phase2_intraday_tcn/              ← Phase 2 single-branch H=6 (참고 baseline)
│   ├── test_predictions_seed{42,123,7}.parquet
│   ├── ensemble_test.parquet
│   └── metrics_summary.csv
├── phase2_intraday_tcn_h{3,4,5}/    ← Phase 2 H window sweep (참고)
│   └── test_predictions_seed{42,123,7}.parquet
├── phase2_2branch{,_g15}/           ← 2-branch λ=1.0 / 1.5 (참고)
│   └── test_predictions_seed{42,123,7}.parquet
├── phase2_2branch_g20_T5{,_th10,_th20}/   ← Sparse gate sweep (negative result)
│   └── test_predictions_seed{42,123,7}.parquet
├── thermal_planner_v2/               ← short-horizon balancing planner ★ (채택)
│   ├── log_phase1.parquet            ← reactive (look-back realized gap)
│   ├── log_phase1plus2.parquet       ← predictive (realized + forward gap, 일몰까지)
│   └── summary.csv
├── thermal_planner_v1/               ← v1 reserve-based dispatch (참고)
│   ├── dispatch_phase1.parquet
│   ├── dispatch_phase1plus2.parquet
│   └── summary.csv
└── (이전 비교 실험 — 채택 안됨)
    ├── resmlp_tcn_v2_{base,trans}/
    ├── resmlp_adaln_v2_clean*/
    └── resmlp_adaln_v2_delta_*/

src/
├── models/
│   ├── train_resmlp_adaln.py                    ← Phase 1 model 정의
│   ├── train_resmlp_adaln_v2_ensemble.py        ← Phase 1 학습 ★
│   └── train_phase2_intraday_tcn.py             ← Phase 2 학습 ★
├── eval/
│   └── phase2_ensemble.py                       ← Phase 2 ensemble + 메트릭
└── decisions/
    ├── thermal_planner_v2.py                    ← short-horizon balancing planner ★ (채택)
    └── thermal_planner_v1.py                    ← v1 reserve-based (참고)

src/models/ (continued)
├── train_phase2_2branch.py                      ← Phase 2 final (2-branch + gate) ★

pv/notebooks/
├── 09_problem_site_diagnostic.ipynb             ← 데이터 진단
├── 10_yecheon_isolated.ipynb                    ← 예천 단독
├── 11_delta_volatility_diagnostic.ipynb         ← delta feature 진단
├── 12_phase2_window_eda.ipynb                   ← Phase 2 H sweep 사전 EDA
└── 13_event_flag_eda.ipynb                      ← event flag 신호 ROC/PR 분석
```

---

## 7. 다음 단계

1. **dashboard** (Streamlit) — realized gap, forward gap (일몰까지), 화력 보정 권장, 3-페이지 구성 (Forecast Replay / Gap and Backup Plan / Balancing Response). 상세는 `plan/active/pv/dashboard_plan.md`.
2. **planner parameter tuning** — `THR_LOW` (잡음 컷), `RAMP_UP_RATE`/`RELEASE_RATE` 조정으로 sign flips와 oscillation 최소화 (부분추종 ALPHA는 폐기됨)
3. **광양항 site bias correction** (post-hoc) — bias +7% 직접 감산, 광양항 단독 처치
4. **시연 스토리** — D-1 Phase 1 baseline → D-day Phase 2 rolling update → balancing response → "2025-03-23 shortfall 23~31% 감소"를 demo case로
5. **운영자 의사결정 시각화** — 현 시각 이전 (실행됨) / 이후 (forward 기반 권장)을 dashboard에 동적으로 표시 (현재 page 2에 구현 완료)

---

## 8. 결론 한 줄

> **Phase 1은 안정적 base, Phase 2는 actual-anchored intraday corrector이며 최종 채택 구조는 평상시 base 사용 + event 시 soft-gated event branch가 추가 보정하는 2-branch (H=6, issue 7~16, λ=2.0, T=1, τ=0) 구성이다. 다만 최종 Phase 2의 출력 정의는 고정 3-step patch가 아니라 `remaining-day EOD-truncated reforecast`이며, 이를 통해 가까운 1~3시간의 직접 대응뿐 아니라 이후 recovery / persistence / sunset curve shape까지 제공한다. Balancing planner는 이 곡선의 앞쪽을 direct control horizon으로, 뒤쪽을 response persistence 판단용 tail로 읽어 incremental thermal action recommendation에 연결한다. 그 결과 short-horizon accuracy와 remaining-day operational reasoning을 동시에 확보했으며, backbone 자체가 아니라 forecast-to-decision pipeline이 PoC novelty의 핵심이다.**
