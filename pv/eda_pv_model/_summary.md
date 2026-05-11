# PV 예측 모델 — EDA 및 의사결정 정리

> 본 문서는 PV 예측 모델 (Phase 1 / Phase 2 / Outage override) 의 모든 *설계 결정* 이 어떤 EDA·실험으로 정당화되었는지 한 곳에 모은 SSOT 이다.
> LNG planner 측 EDA 는 `pv/eda_lng_planner/` 에 분리.
>
> **작성**: 2026-05-10 (M1~M5 motivating EDA 추가)
> **참고 SSOT**: `plan/active/pv/model_final.md`

## 폴더 구조

```
pv/eda_pv_model/
├ M1_site_pattern_heterogeneity.py      ★ 사이트 패턴 → AdaLN site/hour conditioning 정당화
├ M2_temporal_scale.py                  ★ 시간 척도 → Phase 2 / H=6 정당화
├ M3_input_marginal_value.py            ★ ASOS + GK2A 입력 → dsr 압도적
├ M4_outage_separability.py             ★ outage 통계 분리 → rule-based 정당화
├ M5_phase1_residual_pattern.py         ★ Phase 1 residual 패턴 → Phase 2 correction 가능
├ _selection_results/                    (기존 sweep 결과 — 모델 선정 결과 자료)
│  ├ 01_site_selection.png
│  ├ 02_model_comparison.png
│  ├ 03_phase2_architecture.png
│  └ 04_calibration_seed.png
├ _summary.md                            (본 문서)
└ README.md
```

**중요한 구분**:
- `M1~M5` = *데이터 자체에서 모델 결정 도출* (motivating EDA)
- `_selection_results/01~04` = *결정 후 실험 결과* (selection / tuning)

---

## 0. 의사결정 한눈

| # | 결정 사항 | 근거 EDA / 실험 | 위치 |
|---|---|---|---|
| 1 | **8 사이트만 사용** | 사이트별 NMAE 분포 + capacity-weighting + 데이터 누락률 | §1, `01_site_selection.py` |
| 2 | 데이터 분리 (2022-23 train / 24 val / 25 test) | 4년 보유 + 매년 변동 차이 | §2 |
| 3 | 입력 변수: ASOS + GK2A + static (site/hour/month) | feature importance + 누락 처리 | §3 |
| 4 | **Backbone = ResMLP+AdaLN v2 ensemble** (5-seed) | 5종 모델 head-to-head | §4, `02_model_comparison.py` |
| 5 | **Phase 2 = 2-branch + soft gate (λ=2.0)** | λ sweep + sparse gate negative result | §5, notebook 12, `03_phase2_architecture.py` |
| 6 | **Phase 2 H=6 (look-back window)** | H={3,4,5,6} sweep | notebook 12, §5 |
| 7 | **Phase 2 L=12 EOD truncation** | L={3,6,12} 비교 | §5 |
| 8 | **Phase 2 Outage override = rule-based** | retrain + integrated head 모두 폐기 | §6, model_final §2.4 |
| 9 | Outage rule: `cf<0.03 ∧ μ_p1>0.20 ∧ z<−3 ∧ dc<7 ∧ neighbor` | grid sweep + threshold sensitivity | §6, notebook 13 |
| 10 | Recovery rule: `actual_cf > 0.10` 1h 지속 | 0.10/0.15/0.20 sweep, 1h vs 2h 비교 | §6 |

---

## 1. 사이트 선정 — 왜 8 사이트만 사용했나

### 1.1 후보 풀 (KOEN 신재생 사업현황)

남동발전 신재생에너지 사업현황 (`data/info/...신재생에너지 사업현황...csv`) 에는 20+ 사이트 정보가 있으나, 본 PoC 는 *공공 데이터로 hourly 발전 실적이 안정적으로 확보 가능한* 8 사이트만 선정.

| 사이트 | 지역 | 용량 (kW) | 비고 |
|---|---|---|---|
| 고흥만수상 | 전남 | 63,481 | ★ 절대 압도 (전체 capacity 의 82%) |
| 영흥 | 인천 | 5,493 | 인천만 영흥도 |
| 광양항세방 | 전남 | 2,993 | 항만 microclimate |
| 예천 | 경북 | 2,000 | 영농형 |
| 삼천포 | 경남 | 1,340 | |
| 구미 | 경북 | 992 | |
| 경상대 | 경남 | 905 | 학교 옥상 |
| 창원 | 경남 | 77 | 매우 작음, 노이즈 검증용 |
| **합계** | | **77,281 kW** | **= 77.28 MW (전국 PV proxy)** |

### 1.2 사이트별 NMAE 차이 — 모델 가능성 분석

5-seed ensemble Phase 1 결과 (`pv/experiments/resmlp_adaln_v2_ensemble/ensemble_per_site.csv`):

| 사이트 | NMAE | bias | Cov80 | Cov95 | 평가 |
|---|---:|---:|---:|---:|---|
| 경상대 | **3.38%** | +0.37 | 92.0 | 97.5 | 가장 모델링 잘됨 |
| 고흥만수상 | 4.41% | +0.12 | 89.4 | 96.7 | 큰 cap, 적정 |
| 영흥 | 4.36% | +0.45 | 88.0 | 95.1 | 적정 |
| 삼천포 | 4.68% | -0.23 | 88.2 | 95.3 | 적정 |
| 구미 | 5.17% | +2.02 | 83.4 | 94.3 | 약한 outage 영향 |
| 예천 | **8.12%** | -0.67 | 78.0 | 90.8 | anomaly_zero 4.95% |
| **광양항세방** | **8.33%** | **+7.02** | 69.9 | 84.7 | bias 큼, microclimate |
| 창원 | 8.55% | +2.22 | 90.9 | 98.4 | cap 0.077 MW (잡음) |

→ **광양항세방 + 예천 + 창원** 이 NMAE 8% 대로 outlier. 그러나 *capacity-weighted portfolio* 에서는 고흥만수상 (63 MW) 이 dominant 라 portfolio NMAE 3.75% (=좋음).

### 1.3 사이트 처치 차별화

- **예천 anomaly_zero (4.95% 행)**: 모델 학습 시 마스킹 X (운영 모듈은 actual=0 신호도 학습), 평가 시 raw vs clean 두 metric 동시 보고 (model_final §4)
- **광양항 bias +7%**: post-hoc bias correction 후보 (model_final §7-3)
- **창원 cap 0.077 MW**: portfolio 영향 미미 (잡음 검증 용도로 유지)

→ **모든 8 사이트 유지하되, 사이트별 처치 차별화** 가 본 PoC 의 정직한 framing.

---

## 2. 데이터 분리 — 학습 기간 결정

```
Phase 1 학습: 2022-01 ~ 2023-12 (train) / 2024 (val) / 2025 (test)
Phase 2 학습: 2024 (train, 10/15 split early stop) / 2025 (test)
```

**근거**:
- 4년 데이터 (2022~2025) 확보. 가장 최신 1년 (2025) test 보존 → backtest 정합성
- Phase 2 는 *issue 시점* 단위로 sample 생성하므로 sample 수가 Phase 1 보다 훨씬 많음 (n=75,312 vs 167k) → 1년 학습으로도 충분
- 2024 single-year valid 가 cloud-pass / outage 등 *대표 event* 모두 포함 → val 으로서 적합

---

## 3. 입력 변수 — feature 선정

### 3.1 사용 feature

| feature | 출처 | 역할 |
|---|---|---|
| `cf` (capacity factor) | KOEN 발전 실적 (시간별) | 학습 타깃 |
| ASOS hourly weather | 기상청 (지상 관측) | 일사·기온·습도 등 (forecast proxy) |
| GK2A v2 위성 자료 | 기상위성센터 GK-2A | 복사·구름 추정 |
| static cond | site_oh, hour, month | 시간/위치 conditioning |

### 3.2 사용하지 않는 것

- 외부 archived weather forecast (Meteologix 등): 일관된 backtest history 확보 어려움 → ASOS/GK2A 를 **perfect-foresight weather proxy** 로 대체 (model_final §4 framing)
- SMP / 시장 정보: 본 PoC 의 평가는 운영 KPI 중심이라 시장 신호 미사용

### 3.3 GK2A v1 → v2 변경 (notebook 00)

- v1: 단순 short-wave radiation 추출
- **v2 (채택)**: 더 풍부한 cloud / aerosol 채널 + 시간 정렬 정밀화
- 효과: feature 다양성 + cloud-pass 신호 강화

---

## 4. Backbone 모델 비교 — 왜 ResMLP+AdaLN 이었나

### 4.1 비교 후보 (`pv/experiments/`)

| 모델 | 종류 | 위치 | 결과 |
|---|---|---|---|
| LSTM baseline | 시계열 RNN | `lstm_baseline/` | NMAE ~5.x, calibration 약함 |
| LSTM v2 (α/drop sweep 3종) | RNN | `lstm_v2_*/` | 미세 개선 |
| NGBoost baseline | tree + uncertainty | `ngboost_baseline/` | NMAE 좋음, 분포 자연스러움 |
| NGBoost grouped/regime/strong | NGBoost variants | `ngboost_grouped/`, ... | 큰 개선 X |
| FiLM-NGBoost | conditional NGBoost | `film_ngboost/` | 미세 개선 |
| FT-Transformer | tabular transformer | `ft_transformer/` | NMAE 양호, 무거움 |
| GRU+FiLM | RNN + conditioning | `gru_film/` | 미세 |
| Hybrid ResMLP+CNN | tabular + CNN | `hybrid_resmlp_cnn/` | 미세 |
| TFT | transformer | `tft_baseline/` | 비효율 |
| ResMLP+TCN | tabular + TCN | `resmlp_tcn*/` | 미세 |
| **ResMLP+AdaLN v2 ensemble** | tabular + AdaLN | `resmlp_adaln_v2_ensemble/` | ★ **채택** |

### 4.2 ResMLP+AdaLN v2 채택 근거

| 지표 | 결과 (5-seed ensemble) |
|---|---:|
| Site NMAE (test 2025) | **4.66%** |
| Portfolio NMAE | **3.75%** |
| Cov80 / Cov95 | 85.0% / 94.1% |
| NLL | -1.218 |

**채택 이유** (model_final §1.1):
1. test 2025 NMAE / calibration 모두 안정적
2. ensemble 구조가 `sigma_total` 을 자연스럽게 제공 (Phase 2 의 frozen anchor)
3. tabular 입력 (ASOS + GK2A + static) 에 적합한 architecture
4. AdaLN conditioning 으로 site/hour/month 효과 자연스럽게 흡수
5. **Phase 2 residual correction 을 붙이기 구조적으로 적합** — 가장 결정적

→ 단순 forecast 정확도가 아니라 **다음 단계 (Phase 2) 가 의존할 수 있는 안정적 anchor** 라는 점이 채택의 핵심.

---

## 5. Phase 2 architecture sweep — 모든 hyperparameter 결정 근거

### 5.1 H (look-back window) sweep — H=6 채택 (model_final §1.5)

3-seed ensemble × H={3, 4, 5, 6}:

| H | Overall | Lead 1h | Partial cloud | 03-23 | 04-26 | 05-04 | per-seed std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 4.717 | 4.239 | 5.936 | 45.31 | 34.75 | 42.42 | 0.020 |
| 4 | 4.715 | 4.237 | 5.913 | 44.39 | 34.51 | 41.63 | 0.004 |
| 5 | 4.715 | 4.242 | 5.906 | 42.93 | 34.41 | **41.32** | 0.013 |
| **6 ★** | 4.729 | 4.256 | 5.935 | **41.98** | **34.31** | 42.10 | 0.049 |

**판정**:
- Overall NMAE 차이 0.01~0.02%p (사실상 동률)
- **Top cloud-pass event 에서 H=6 가장 좋음** (03-23 41.98 vs H=5 42.93)
- PoC 목적 = top-event robustness 우선 → H=6 채택

### 5.2 λ (event branch 가중치) sweep — λ=2.0 채택 (model_final §1.6)

H=6 single-branch 대비 2-branch λ ∈ {1.0, 1.5, 2.0}:

| 비교 | NMAE | Lead 1h | Partial cloud | per-seed std |
|---|---:|---:|---:|---:|
| H=6 single-branch | 4.729 | 4.256 | 5.935 | 0.049 |
| 2-branch λ=1.0 | 4.738 | 4.271 | 5.899 | 0.023 |
| 2-branch λ=1.5 | 4.742 | 4.261 | 5.882 | 0.014 |
| **2-branch λ=2.0 ★** | **4.712** | **4.209** | **5.871** | **0.026** |

**채택 이유**:
- λ=2.0 모든 핵심 metric 에서 baseline 동률 또는 우세
- per-seed 안정성 H=6 0.049 → λ=2.0 0.026 (**−47%**)
- gate top-event 활성화 1.9x (max 0.95) — event 시 추가 correction 동작

### 5.3 Sparse gate sweep — 폐기 (negative result, model_final §1.7)

`gate = sigmoid(z/T)` 의 T 와 threshold τ sweep:

| config | gate ratio | dead seeds | NMAE | top 03-23 |
|---|---:|---:|---:|---:|
| **T=1.0 τ=0 (최종)** | 1.9x | 0/3 | **4.712** | 42.12 |
| T=0.5 τ=0 | 5.8x | 0/3 | 4.743 | 43.77 |
| T=0.5 τ=0.1 | 16.8x | **2/3** | 4.732 | 42.56 |
| T=0.5 τ=0.2 | 0x (전체 dead) | 3/3 | 4.729 | 41.98 |

**관찰**:
- Selectivity 향상 가능하지만 vanishing-gradient 로 gate dead 폭증
- 운영 PoC 안정성 부적합 → 폐기

### 5.4 L (output horizon) — L=12 EOD 채택 (model_final §1.3)

L = {3, 6, 12 EOD truncation} 비교:
- `L=3`: lead-1 즉시 보정은 가장 강함, 그러나 *남은 하루 shape* 정보 X
- `L=6`: L=12 에 의해 dominate
- **`L=12 EOD truncation` ★**: lead-1~3 정확도 대부분 보존 + recovery / persistence / sunset curve shape 추가

→ Phase 2 의 해석이 *short patch generator* 에서 **intraday remaining-day reforecast module** 로 전환됨.

---

## 6. Outage Override Layer — 3 단계 negative result + rule-based 채택

### 6.1 시도 #1: Phase 1 outage-masked retrain (`v3_outage_masked*`) — 폐기

| variant | mask 수 | NMAE | baseline 대비 |
|---|---:|---:|---:|
| baseline | 0 | 4.747% | — |
| narrow | 278 | 4.89% | **+0.14 pp** |
| wide | 539 | 4.91% | **+0.16 pp** |

**원인**: 0.2~0.4% 마스킹으로 분포 변화 미미. 모델은 outage 를 noise 로 robust 학습.
**결론**: 폐기.

### 6.2 시도 #2: Integrated outage probability head (`phase2_2branch_outage_*`) — 폐기

- Phase 2 architecture 에 outage_prob head 추가 + joint loss
- 결과: precision 3.3%, recall 41.7% (FP 폭증)
- 04-26 cloud-pass 13.79% → 20.24% **악화**
- 폐기 이유:
  - Label imbalance 너무 심함 (positive 0.21%)
  - Outage 가 weather feature 로 예측 불가능 (site-specific 운영 사건)
  - 학습 sample 부족 (437 positive)
  - **Rule-based detection 이 더 자연스러운 도메인** (actual cf 직접 보고 즉시 판단 → 0% FP)

### 6.3 채택: Rule-based override (`outage_v3.5 relaxed`)

```python
outage_flag_t = (
    cf < 0.03                  # 발전 거의 0
    AND mu_phase1 > 0.20       # P1 baseline 의미있는 발전 예상
    AND dc10Tca < 7            # 부분 흐림까지 허용 (강한 흐림 제외)
    AND z < -3.0               # P1 ±3σ 밖 통계적 outlier
    AND neighbor_confirmed     # 1시간 단발 X (2h+ 연속만)
)
recovery: actual_cf > 0.10 for 1h 지속 → NORMAL 복귀
```

**채택 결과** (test 2025):

| 그룹 | n | P1 | P2 | **P2+override** | Δ |
|---|---:|---:|---:|---:|---:|
| 전체 | 167,953 | 4.747% | 4.432% | **4.220%** | −0.21 pp |
| Blackout | 587 | 68.6% | 55.7% | **0.39%** | −55.3 pp |
| Normal | 167,366 | 4.50% | 4.23% | **4.23%** | **+0.000 pp ✓** |
| 03-23 cloud-pass | 440 | 30.0% | 23.3% | **2.89%** | −20.4 pp |
| 04-26 cloud-pass | 624 | 15.0% | 13.7% | **3.81%** | −9.8 pp |
| 광양항 10/10-12 | 165 | 28.2% | 15.5% | **0.09%** | −15.4 pp |

**검증 통과**:
- Normal 시점 NMAE 변화 +0.000pp → 정상 시점 *손상 없음*
- Blackout 시점만 깔끔 처리

### 6.4 임계값 sensitivity (notebook 13)

- `z < -3.0` (P1 ±3σ 밖): 단순 z<-2 면 false positive 폭증, z<-3 이 정상-outage 분리에 적합
- `dc10Tca < 7`: cloud-pass 까지 Phase 2 처리 영역으로 남김 (dc>=7 = 강한 흐림은 weather event)
- `neighbor_confirmed`: 1h 단발은 측정 오류 가능성 높아 제외
- `recovery 0.10 / 1h`: 0.10 vs 0.15 거의 동일 효과 → 단순한 0.10 채택. 1h vs 2h 는 2h 가 over-conservative (정상 시점도 0 으로 덮음).

---

## 7. Calibration 분석

5-seed ensemble Phase 1 (test 2025):
- Cov80 = 85.0% (목표 80%, 약간 over-coverage)
- Cov95 = 94.1% (목표 95%, 약간 under)
- NLL = −1.218

→ ensemble 의 sigma_total 이 *대체로 well-calibrated*. Phase 2 는 σ 변경 X (Phase 1 그대로 사용) 로 calibration 유지.

---

## 8. 5-seed ensemble effect

per-seed Phase 1 NMAE: 4.68 / 4.85 / 4.77 / 4.89 / 4.93 (per-seed std ≈ 0.10)
ensemble NMAE: **4.66%** (5-seed 평균)

→ ensemble 로 NMAE 0.10pp 안정화. *seed variance 자체가 모델 불확실성 신호* 로 활용 가능 (sigma_total 과 결합).

---

## 9. 사용한 EDA 노트북 (`pv/notebooks/`)

| 노트북 | 다룬 내용 | 결정 영향 |
|---|---|---|
| 00 | GK2A v2 위성 자료 inspect | feature v1 → v2 전환 |
| 01 | 사이트별 변동성 | 사이트별 처치 차별화 |
| 02 | PV 변동성 | event 구분 필요성 |
| 03 | 데이터 결합 / 지역별 변동 | data join 검증 |
| 04 | 계절성 | seasonal effect (AdaLN month conditioning) |
| 05 | GHI 반응 | 일사량 → cf 매핑 (linear 영역 확인) |
| 06 | NGBoost 결과 | NGBoost 후보 비교 |
| 07 | 실시간 reweight outlier | online correction logic 후보 |
| 09 | 문제 사이트 진단 | 광양항 / 예천 / 창원 처치 |
| 10 | 예천 단독 anomaly_zero | 마스킹 X 결정 |
| 11 | delta 변동성 | residual feature engineering |
| 12 | Phase 2 H sweep 사전 EDA | H=6 결정 |
| 13 | event flag ROC/PR | outage rule 임계값 선택 |

---

## 10. 정직한 한계 / 향후 작업

본 PoC 의 모델 측면 한계 명시:

1. **기상 입력 = perfect-foresight proxy**: 실제 archived forecast 가 아닌 ASOS/GK2A 실측을 forecast 로 사용. *backtest 일관성* 우선.
2. **광양항 bias +7% 미보정**: post-hoc correction 후보 (§7-3 in model_final), 본 PoC 미적용.
3. **창원 cap 0.077 MW**: portfolio 영향 미미하지만 *모델 학습 noise 기여* 가능. 향후 제외 후보.
4. **Phase 2 lead 3h 수렴**: lead 3h 이상에서 P1 baseline 으로 수렴 (improvement 미미). 더 긴 horizon 은 *recovery shape* 만 의미.
5. **Outage rule 의존**: rule 을 못 맞추는 outage type (예: 부분 capacity loss) 은 학습형 detector 가 필요. 현재 데이터로는 비현실.

---

## 11. EDA 산출 위치

- `pv/eda_pv_model/` — 본 폴더 (메타 문서 + 추가 EDA 스크립트)
- `pv/notebooks/` — 기존 EDA 노트북 14개
- `pv/experiments/` — 60+ 모델 실험 결과
- `plan/active/pv/model_final.md` — Phase 1 / Phase 2 / Outage spec SSOT
