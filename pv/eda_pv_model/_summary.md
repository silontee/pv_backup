# PV 예측 모델 — EDA & 모델 선택 근거

> 본 문서는 *데이터의 성질이 모델 구조를 강제한다* 는 흐름으로 구성된다.
> **1장** 은 PV 데이터가 보여주는 *예측의 난이도 4 가지* — 모델 이름 없이 데이터만.
> **2장** 은 그 난이도 각각에 대해 *왜 이 구조를 골랐는가* — ResMLP / AdaLN / Phase 2 / TCN / Override.
>
> 산출물 (M1~M5, D1~D2 그림·CSV) 은 본 폴더에 그대로 두며, 각 절에서 *근거 자료* 로 인용한다.
> SSOT 인용: `plan/active/pv/model_final.md`, `plan/active/pv/proposal_outline.md`

---

## 1장. 태양광 EDA — 데이터가 보여준 예측 난이도

> 본 장은 **모델 이름을 말하지 않는다.** 데이터에 어떤 *구조* 와 *난이도* 가 있는지만 보여주고, 각 절 끝에서 *"이 데이터가 모델에 요구하는 것"* 한 줄로 다음 장과 연결한다.

### 1.1 평균적으로 일중 패턴은 뚜렷하다

PV 발전은 *아침 상승 → 정오 peak → 저녁 하강* 의 결정적 형태를 가진다. 8 사이트 4년 평균 cf 를 시각별로 집계하면 *정오 ±2h* 구간에 peak 가 모이고, 시각별 분산도 동일 구간이 가장 좁다.

| 산출 | 위치 |
|---|---|
| 시각별 평균 cf + 분산 음영 | `M1_site_pattern_heterogeneity.png` (좌상) |
| 시각별 cf 분포 (boxplot) | `M1_site_hour_cf.csv` 로부터 재집계 가능 |

**관측**: 평균 곡선만 보면 *PV 예측은 24h curve fitting 문제* 처럼 보인다. 즉 *시퀀스 의존성* 보다는 *시각 조건 → 출력값* 의 함수 근사 문제로 다룰 여지가 크다.

**다음 절로의 연결**: 그런데 이 평균 곡선은 *사이트별로 깨진다*. 하나의 함수로 모든 사이트를 fit 하면 잔차가 누적된다.

---

### 1.2 사이트별로 곡선의 형태가 다르다

같은 PV 라도 *용량 · 위치 · 설치 방향 · 지역 microclimate* 에 따라 일중 곡선의 *높이와 형태* 가 다르다. 8 사이트 평균 daily curve 를 겹쳐 그리면 *peak 시각 자체가 다른 사이트* (광양항: 항만 풍 영향), *cf 절대 수준이 +7% bias 인 사이트* (광양항), *영농형으로 산란이 큰 사이트* (예천) 등 구조적 차이가 드러난다.

| 산출 | 위치 |
|---|---|
| 사이트별 일중 cf 곡선 overlay | `M1_site_pattern_heterogeneity.png` (우상) |
| 사이트 × 사이트 daily curve corr heatmap | `M1_site_pattern_heterogeneity.png` (하단) |

**관측**: 사이트 간 daily curve cross-corr 평균 0.98, 최저 0.92 — *높지만 100% 동일이 아닌* 구조. 즉 *공통 곡선 + 사이트별 보정* 이 아니라 *사이트별 곡선 형태 자체가 다른* 문제.

→ **이 데이터가 요구하는 것**: 단순 `site_id` dummy 로 *bias 만* 흡수하면 부족. 사이트 × 시각 × 월 조건에 따라 *함수 형태 자체* 가 달라지는 표현이 필요.

---

### 1.3 당일 구름 통과는 D-1 기준선만으로 잡기 어렵다

D-1 예측은 *평균적 다음날 곡선* 은 만들 수 있다. 그러나 당일 14~17시 cloud-pass 처럼 *수시간 스케일의 출력 급락·회복* 은 D-1 시점에 알 수 없다.

| 산출 | 위치 |
|---|---|
| 맑은 날 vs 부분 흐림 cf curve 비교 | `M2_temporal_scale.png` (좌측) |
| Phase 1 residual 자기상관 (3h scale 까지 양의 corr) | `M5_residual_pattern.png`, `M5_residual_autocorr.csv` |
| 대표 cloud-pass 사례 (03-23, 04-26, 05-04) | proposal §6.2 인용 |

**관측**: Phase 1 residual 의 자기상관이 *1~3h scale 에서 의미 있게 양* — 즉 *부족이 한 시점에만 나오는 게 아니라 수시간 지속* 된다. 이게 의미하는 바: *최근 실측 cf* 를 입력으로 받으면 *남은 시간의 부족분을 미리 좁힐 수 있다*.

→ **이 데이터가 요구하는 것**: D-1 기준선 위에 *당일 실적을 입력으로 받는 재예측 layer* 가 필요.

---

### 1.4 일부 near-zero 출력은 기상으로 설명되지 않는다

`dsr_mean` 이 충분히 높은 시각인데 cf 가 거의 0 인 *high DSR & low CF* 영역이 존재한다. 이 영역은 기상 변수 어느 것으로도 설명되지 않는다.

| 산출 | 위치 |
|---|---|
| DSR × cf 산점도 (우하단 강조) | `M4_outage_separability.png` |
| 사이트 × day × hour outage heatmap | `D_site_outage_heatmap.png` (proposal §4.4.3) |

**관측**: outage 사례는 *site-level shutdown* (광양항 10/10-12 3일 종일 cluster) 처럼 사이트 단위 운영 사건. 빈도 0.2~0.4%, 분포 자체가 *연속/clustered*. 기상 입력에는 어떤 단서도 없음.

**왜 학습으로 풀면 안 되는가**:
- 라벨 imbalance 너무 심함 (positive 비율 ~0.21%, 학습 sample 437개)
- *원인 자체가 weather feature 의 함수가 아님* — site-specific operational event
- outage-masked retrain 시도 결과: NMAE +0.14 ~ +0.16 pp **악화** (분포 변화 미미, mask 가 noise robust 학습을 깸)
- integrated outage_prob head 시도 결과: precision 3.3%, recall 41.7% (FP 폭증), 04-26 cloud-pass 13.79% → 20.24% 악화

→ **이 데이터가 요구하는 것**: outage 는 *예측 모델이 흡수하려 하면 안 되는 영역*. 통계적 outlier 신호 (cf 거의 0 · μ_p1 큼 · z<-3 · 부분 흐림 이하) 로 *분리* 해서 별도 layer 가 처리해야 한다.

---

### 1장 요약

| EDA 관측 | 데이터가 요구하는 것 | 다음 장에서 연결될 결정 |
|---|---|---|
| 1.1 평균 일중 패턴은 결정적 | *시각 → 출력값* 함수 근사 문제로 framing | ResMLP backbone |
| 1.2 사이트별 곡선 형태가 다름 | 사이트 × 시각 × 월 조건부 함수 | AdaLN conditioning |
| 1.3 당일 cloud-pass | 실측을 입력 받는 재예측 layer | Phase 2 + TCN |
| 1.4 high DSR & low CF | 통계적 분리 + 학습 외 처리 | Outage Override (rule-based) |

---

## 2장. 모델 선택 근거 — 왜 이 구조인가

> 본 장은 1장에서 도출한 4 가지 *데이터 요구* 각각에 *어떤 architectural choice* 로 답했는지 정리한다.
> 모델 자체의 sweep 결과는 `_selection_results/` 폴더 + proposal §6 에서 인용.

### 2.1 왜 ResMLP 계열 backbone 인가

PV D-1 예측은 *시퀀스 의존성* 을 길게 따라가는 문제가 아니다. 입력이 *(hour, month, zenith, dsr, ASOS, site)* 로 고정되면 *그 시각의 cf 함수* 는 거의 결정된다 (§3.4: 4 변수군 + site_oh 만으로 RF R² 0.84).

| 시도 모델 | 위치 | 결과 |
|---|---|---|
| LSTM baseline / v2 (α/drop sweep) | `lstm_baseline/`, `lstm_v2_*/` | NMAE ~5.x, calibration 약함 |
| FT-Transformer (tabular) | `ft_transformer/` | 양호하나 무거움 |
| TFT | `tft_baseline/` | 비효율 |
| NGBoost 계열 (baseline/grouped/regime/FiLM) | `ngboost_*/`, `film_ngboost/` | NMAE 양호, ensemble 결합 어려움 |
| **ResMLP+AdaLN v2 ensemble** ★ | `resmlp_adaln_v2_ensemble/` | **채택 — NMAE 4.66% / portfolio 3.75%** |

**채택 이유**:
1. 시각·조건별 *비선형 함수 형태* 학습에 충분 (4 변수군 R² 0.84)
2. ensemble 구조가 `sigma_total` 을 자연스럽게 제공 → Phase 2 의 *frozen anchor* 로 그대로 활용
3. tabular 입력 (ASOS + GK2A + static) 과 정합
4. *다음 단계 (Phase 2) 가 의존할 수 있는 안정적 anchor* — 이것이 결정적

> **발표용 표현**
> D-1 예측은 긴 문맥을 해석하는 문제가 아니라, 시간·일사·사이트 조건에 따른 발전 함수 형태를 안정적으로 근사하는 문제이므로 ResMLP 기반 구조를 채택하였다.

---

### 2.2 왜 AdaLN 조건부 정규화인가

§1.2 가 *사이트별로 함수 형태 자체가 다르다* 를 보였다. 정적 site dummy 로는 부족하다 — 실제로 누적 R² 분석에서 site_oh 추가가 *+0.012pp* 만 기여 (proposal §3.4): bias 만 흡수하고 *시각/계절 패턴 차이* 는 못 잡는다.

**AdaLN 의 역할**: layer normalization 의 *scale (γ)* 과 *shift (β)* 를 *site × hour × month* 조건에 따라 동적으로 결정. 즉 같은 모델이 *광양항의 11시* 와 *고흥만의 11시* 를 서로 다른 normalized representation 위에서 학습한다.

| 비교 | NMAE | 비고 |
|---|---:|---|
| site dummy 만 | (proposal §3.4) +0.012pp | bias 흡수만 |
| AdaLN site × hour × month | 4.66% (5-seed ensemble) | **함수 형태 차별화** |

> **발표용 표현**
> 사이트별 곡선 차이를 단순 dummy 변수로 처리하지 않고, site·hour·month 조건에 따라 모델 내부 표현을 조정하는 AdaLN 을 적용하였다. 이를 통해 하나의 모델 안에서 사이트별 발전 패턴 차이를 분리 반영할 수 있다.

---

### 2.3 왜 Phase 2 재예측인가

§1.3 가 *당일 cloud-pass 는 D-1 기준선만으로 못 잡는다* 를 보였고, Phase 1 residual 자기상관이 1~3h scale 에서 양으로 남는 것도 확인했다 (`M5_residual_pattern.png`). 즉 *실측이 들어오면 남은 시간의 부족분을 좁힐 정보* 가 데이터에 남아있다.

**Phase 2 의 역할**: D-1 baseline (μ_p1) 을 frozen anchor 로 두고, *최근 H 시간 실측 cf* 를 입력으로 *남은 L 시간 (EOD truncation)* 의 residual 을 보정한다. baseline 의 σ 는 그대로 유지 → calibration 보존.

**중요 결정** (§5 sweep 결과):
- **L = 12 EOD truncation**: lead-1 부근 정확도 보존 + 남은 하루 *recovery / sunset curve shape* 추가
- **2-branch + soft gate (λ=2.0)**: event 영역에서 추가 correction 활성, per-seed 안정성 −47%

| 그룹 | n | Phase 1 | Phase 2 | Δ |
|---|---:|---:|---:|---:|
| 전체 (정상) | 167,366 | 4.50% | 4.23% | **−0.27pp** |
| 03-23 cloud-pass | 440 | 30.0% | 23.3% | **−6.7pp** |
| 04-26 cloud-pass | 624 | 15.0% | 13.7% | −1.3pp |

> **발표용 표현**
> D-1 기준선은 평균적인 다음 날 발전 곡선을 제공하지만, 당일 발생하는 구름 통과와 급격한 출력 변화는 실시간 실적 없이는 반영하기 어렵다. 따라서 Phase 2 에서는 당일 실적을 입력으로 받아 남은 시간 예측을 갱신하였다.

---

### 2.4 왜 TCN 인가 (그리고 왜 H=6 인가)

Phase 2 의 *최근 H 시간 실측* 을 어떻게 처리할까. 후보는 LSTM (순차), Transformer (full attention), TCN (dilated causal conv) 셋. PV cloud-pass / 회복 패턴은 *전체 history attention* 이 필요한 문제가 아니라 *직전 수 시간의 추세 + 급변* 이 핵심.

**TCN 이 적합한 이유**:
- *직전 수 시간* 의 local pattern 을 dilated causal conv 로 자연스럽게 잡음
- 순차 처리 없이 병렬 학습 (LSTM 대비 빠름, Transformer 대비 가벼움)
- receptive field 가 H 와 1:1 — H=6 이면 6h 전체가 한 conv stack 에 들어옴

**H=6 선택 (3-seed × H={3,4,5,6} sweep)**:

| H | Overall NMAE | 03-23 (top cloud-pass) | 04-26 | 05-04 |
|---|---:|---:|---:|---:|
| 3 | 4.717 | 45.31 | 34.75 | 42.42 |
| 4 | 4.715 | 44.39 | 34.51 | 41.63 |
| 5 | 4.715 | 42.93 | 34.41 | 41.32 |
| **6 ★** | 4.729 | **41.98** | **34.31** | 42.10 |

- Overall NMAE 는 H={3,4,5,6} 사실상 동률 (±0.02pp)
- *Top cloud-pass event* 에서 H=6 가장 우세 (03-23: 41.98 vs H=5 42.93)
- PoC 목적 = *top-event robustness* 우선 → H=6 채택

> **발표용 표현**
> 구름 통과나 출력 저하는 단발성 노이즈가 아니라 몇 시간 동안 이어질 수 있다. 따라서 직전 6시간 실적을 TCN 입력으로 사용해 당일 변동 흐름을 반영하였으며, H=3~6 sweep 에서 top cloud-pass event 성능이 가장 우세한 H=6 을 선택하였다.

---

### 2.5 왜 학습형이 아닌 Rule-based Outage Override 인가

§1.4 가 *outage 는 weather feature 로 예측 불가능* 임을 보였다. 두 가지 학습형 시도가 모두 실패했다:

| 시도 | 결과 | 폐기 이유 |
|---|---|---|
| Phase 1 outage-masked retrain | NMAE +0.14 ~ +0.16pp 악화 | mask 가 noise robustness 를 깨고, 분포 변화는 미미 |
| Integrated outage_prob head | precision 3.3% / recall 41.7%, 04-26 13.79% → 20.24% | label imbalance + positive sample 부족 |

**채택: Rule-based override** (`outage_v3.5 relaxed`)

```python
outage_flag = (
    cf < 0.03                  # 발전 거의 0
  & μ_phase1 > 0.20            # Phase 1 baseline 의미있는 발전 예상
  & dc10Tca < 7                # 부분 흐림 이하 (강한 흐림은 weather event)
  & z < -3.0                   # Phase 1 ±3σ 밖 통계적 outlier
  & neighbor_confirmed         # 2h+ 연속만 (단발 측정오류 배제)
)
recovery: actual_cf > 0.10 for 1h → NORMAL 복귀
```

**검증 결과 (test 2025)**:

| 그룹 | n | P1 | P2 | **P2+override** |
|---|---:|---:|---:|---:|
| Blackout | 587 | 68.6% | 55.7% | **0.39%** |
| Normal | 167,366 | 4.50% | 4.23% | **4.23%** (Δ=+0.000) |
| 광양항 10/10-12 | 165 | 28.2% | 15.5% | **0.09%** |

- Normal 시점 *NMAE 손상 0.000pp* → override 가 정상 시점을 건드리지 않음
- Blackout 시점만 깔끔 처리

> **발표용 표현**
> 일부 출력 저하는 기상으로 설명되지 않는 site-level 운영 사건이므로, 일반 예측 모델이 이를 학습으로 흡수하면 정상 시점 성능이 손상된다. 따라서 통계적 outlier 규칙으로 outage 를 분리한 뒤 별도 override layer 가 처리하는 구조를 채택하였다.

---

### 2장 요약

| 1장의 데이터 요구 | 2장의 답 | 핵심 sweep / 검증 |
|---|---|---|
| 시각 → 출력 함수 근사 | ResMLP backbone | 5-seed ensemble NMAE 4.66% |
| 사이트별 함수 형태 차별화 | AdaLN site × hour × month | site dummy +0.012pp vs AdaLN 4.66% |
| 당일 cloud-pass 보정 | Phase 2 (L=12 EOD, 2-branch, λ=2.0) | top cloud-pass −6.7~−20.4pp |
| outage 분리 | Rule-based override | Normal 손상 0.000pp / Blackout −55pp |

---

## 부록 A. 인용 산출물 매핑

| proposal 인용 위치 | 본 폴더 산출 |
|---|---|
| §3.4 누적 R² (변수 추가별 cf 설명력) | `D2_variable_selection.png`, `D2_cumulative_r2.csv`, `D2_single_corr.csv`, `D2_hour_corr.csv` |
| §1.1 사이트 선정 (Step 1~3 필터) | `D1_data_constraints.png`, `D1_unit_filter.csv` |
| §4.1 사이트 이질성 | `M1_site_pattern_heterogeneity.png`, `M1_site_hour_cf.csv`, `M1_site_month_cf.csv` |
| §4.x 시간 척도 (Phase 2 H=6 motivation) | `M2_temporal_scale.png`, `M2_autocorr.csv` |
| §3.x 입력 변수 marginal value | `M3_input_marginal.png`, `M3_corr.csv`, `M3_hour_corr.csv` |
| §4.4 outage separability | `M4_outage_separability.png` |
| §x Phase 1 residual 자기상관 | `M5_residual_pattern.png`, `M5_residual_autocorr.csv`, `M5_hour_residual.csv`, `M5_site_residual.csv` |

`_selection_results/` 의 4 장 PNG (사이트 selection / model comparison / Phase 2 architecture / calibration) 는 *결정 후 실험 결과* 자료로 별도 분리.

---

## 부록 B. 본 문서 외 SSOT 위치

- `plan/active/pv/model_final.md` — Phase 1 / Phase 2 / Override 의 모델 spec SSOT
- `plan/active/pv/proposal_outline.md` — 제안서 §3 변수 선정, §4 EDA, §5 모델 결정, §6 architecture
- `pv/notebooks/` — 노트북 14개 (00~13)
- `pv/experiments/` — 60+ 모델 실험 결과 (LSTM / NGBoost / FT-T / ResMLP variants)
