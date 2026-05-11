# PV 예측 Plan v1 — 2-Stage (MOS + PV) + 실시간 운영 구조

> 기준일: 2026-05-03 (revised)
> 상위 계획: `plan/current/plan.md` (v8) — 본 v1에서 Stage 1로 흡수
> 결정 근거: `plan/decisions/2026-04-28-pv-mos-stage.md`
> 데이터 SSoT: `plan/current/data_strategy.md`
> 관련 문서: `plan/fuel/v1.md` (Stage 2 LNG 시나리오)
> 상태: **DRAFT** — Gate 통과 후 `current/plan.md` (v9)로 승격

---

## 1. 목적

학습 분포(GK-2A 실측)와 추론 분포(Open-Meteo 예보) 간 distribution shift를 명시적으로 다루는 *2-Stage 예측 파이프라인* + *실시간 운영* 구조 정의.

핵심 가치 명제:
- **D-1**: 24h forecast로 KPX 가용용량 신고 + LNG 사전 계획
- **D-day**: 실시간 PV 실측 vs 예측 *gap* 기반 LNG 출력 조정 (실시간 변동성 대응)

---

## 2. 전체 아키텍처

### 2.1 학습 단계 (offline)

```
┌─────────────────┐     ┌─────────────────────────┐
│  GK-2A 실측 DSR │ ──→ │  Stage 1: PV 모델 학습  │
│  ASOS T/H/W    │     │  (NGBoost / TFT)        │
│  KOEN PV 실측  │ ──→ │  → cf 분포 (μ, σ)       │
└─────────────────┘     └─────────────────────────┘

┌─────────────────────────┐     ┌─────────────────┐
│ Open-Meteo historical-  │ ──→ │ Stage 0: MOS    │
│ forecast (4년치)        │     │ (bias 보정)     │
│ + GK-2A 실측 (정답)     │ ──→ │                 │
└─────────────────────────┘     └─────────────────┘

→ 두 모델 *별도* 학습. 데이터셋 안 섞임.
```

### 2.2 추론 단계 (D-1)

```
Open-Meteo forecast (D+1 24h)
       │
       ▼
┌──────────────┐  사이트별·시간대별
│  Stage 0     │  bias 보정 학습됨
│   (MOS)      │
└──────┬───────┘
       ▼
보정된 GHI + 기상 (≈ GK-2A 실측 분포)
       │
       ▼
┌──────────────┐  Track A: NGBoost (parametric Normal)
│  Stage 1     │  Track B: TFT (quantile head)
│  (PV 모델)   │
└──────┬───────┘
       ▼
24h × 사이트 분포 (μ, σ 또는 quantiles)
       │
       ▼
Monte Carlo 1000 시나리오 풀
       │
       ▼
[Stage 2: LNG 시나리오] (plan/fuel/v1.md)
```

### 2.3 실시간 운영 (D-day)

```
시간 →
[──6h──] [──6h──] [──6h──]
풀 #1     풀 #2     풀 #3      ← Open-Meteo 갱신 (re-inference)
↑↑↑↑↑↑   ↑↑↑↑↑↑   ↑↑↑↑↑↑    ← 매 1h KOEN PV 실측 (re-weighting)
```

| Trigger | 빈도 | 처리 |
|---|---|---|
| Open-Meteo 새 forecast | **6시간** | **Re-inference** (Stage 0 → Stage 1 재실행, 풀 교체) |
| KOEN PV 실측 (시뮬레이션) | **1시간** | **Re-weighting** (기존 풀 내 trajectory 매칭) |

---

## 3. Stage 1 모델 결정 (사이트 분산 분석 결과)

### 3.1 Variance Decomposition Gate (Gate 0) — PASSED

`pv/notebooks/01_site_variance.ipynb`, `02_pv_variance.ipynb` EDA 결과:

| 분산 성분 | DSR 기준 | PV CF 기준 (canonical 필터) |
|---|---|---|
| COMMON (한국 일조·계절) | 51.8% | 35.7% |
| **사이트 systematic** | **0.5%** | **1.7%** |
| within-site (날씨, 사이트 공유) | 47.7% | 62.5% |
| **ICC** | **1.0%** | **1.7~2.7%** |

→ **사이트 거의 동질** (위도 의존 offset ±5%CF 수준만).
→ Bayesian hierarchical / FiLM modulation **정당화 불가**.
→ **Joint 모델 with site_id (categorical) 확정**.

### 3.2 모델 구조 — Hybrid Track 비교

```
Track A: NGBoost (CPU baseline)
  - Distribution: Normal (post-clip ≥ 0)
  - Output: μ, σ per (site, hour)
  - 학습: 분 단위 (CPU)
  - 특성: 단순, 빠름, parametric

Track B: TFT (Temporal Fusion Transformer, GPU 메인)
  - Output: 9 quantile per (site, hour)
  - 학습: 1~3시간 (GPU)
  - 특성: attention 기반, 노벨티 ★, multi-step direct (AR 누적 X)
```

### 3.3 베이스라인 결과 (NGBoost)

```
TEST (2025년, daytime only, 32k 행):
  NMAE (capacity 기준):   6.20%   ← 목표 ≤ 6%
  포트폴리오 NMAE:        5.18%   ← 목표 6% 달성 ✅
  80% interval coverage:  80.6%   ← well-calibrated
  95% interval coverage:  92.9%

사이트별 NMAE: 경상대 3.80% ~ 예천 8.98%
vs VPP 6.2% NMAE (problem.md 기준): outperform
```

**한계 (Caveat)**:
- Test도 GK-2A 실측 input → "**MOS perfect 가정**"의 상한 성능
- 실 운영 시 = test 성능 + MOS 오차 (Stage 0 미구현 단계)

### 3.4 학습-추론 변수 일관성 처리

| 단계 | Input 분포 | 처리 |
|---|---|---|
| 학습 | GK-2A 실측 (정확) | 그대로 사용 |
| 평가 (현재 baseline) | GK-2A 실측 | "MOS perfect" 상한 |
| 추론 (운영 시) | Open-Meteo 예보 | Stage 0 MOS로 GK-2A 분포 보정 후 사용 |

→ MOS Stage 0가 학습-추론 정합 담당.

---

## 4. Stage 0 (MOS) 설계

### 4.1 입력 / 출력

- **입력**: Open-Meteo `historical-forecast-api` (실제 발행된 과거 예보)
  - `shortwave_radiation`, `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`
  - 메타: lead time, issue time
- **출력**: 사이트별 시간별 보정된 GHI + 기상 (GK-2A 실측 분포에 가까움)

### 4.2 학습 데이터 검증 결과 (Gate 1)

`historical-forecast-api` vs `archive-api` (ERA5) 비교:
- 두 API가 *다른 product*임 확인 (값 systematic 차이)
- historical-forecast-api = **실제 발행된 forecast** (MOS 학습용 적합)
- archive-api = ERA5 reanalysis (사후 best-estimate, MOS 학습 부적합)

⚠️ **Lead time caveat**: historical-forecast-api는 짧은 lead time (4~12h) 가능성 → 우리 D-1 24h-ahead use case와 gap 있음.

### 4.3 MOS 모델 후보

- **XGBoost regression** (사이트·시간대 conditioning) ← 권장
- Quantile regression
- Light MLP

선택 기준: 잔차의 R² > 0.3 (시스템적 편향 학습 가능)

---

## 5. 실시간 운영 — Re-inference vs Re-weighting

### 5.1 두 갱신 신호의 *본질적 차이*

| 신호 | 정체 | 모델에서 의미 |
|---|---|---|
| Open-Meteo 6h 갱신 | 새 forecast 변수 (input) | **Input 변경** |
| KOEN PV 실측 (시뮬) | 실측 PV 값 (target) | **Target 관측** (input 아님) |

→ 처리 방식 다름.

### 5.2 Re-inference (Open-Meteo 6h 갱신 시)

새 forecast → input 자체가 바뀜 → 출력 분포 *근본적*으로 달라짐.
- Stage 0 → Stage 1 재실행
- 1000 시나리오 풀 *교체*
- NGBoost 수 초 / TFT 수 초~분 (inference만, GPU)

### 5.3 Re-weighting (KOEN PV 실측 들어올 때)

실측 PV → input은 그대로, *어떤 trajectory 실현 중인지* 정보.
- 기존 1000 시나리오 풀에서 매칭 trajectory weight ↑
- 후속 시간 분포 갱신 (Bayesian filter)
- 밀리초 단위

### 5.4 PoC 시뮬레이션

KOEN 실시간 SCADA 미연동 → **과거 데이터 playback**으로 시뮬:

```python
for t in test_period_hours:
    if t == open_meteo_update_time:  # 매 6h
        pool = re_inference(new_forecast, ngboost)
    
    pv_actual_at_t = test_data[t]  # 시뮬레이션
    pool = bayesian_reweight(pool, pv_actual_at_t, t)
    
    lng_recommendation = compute_lng(pool, t+1, t+24)
```

**검증 metric**: 실시간 갱신 *없는* baseline vs *있는* 시스템의 LNG 비용 차이.

---

## 6. EDA Gate 결과 정리

| Gate | 내용 | 결과 |
|---|---|---|
| 0 | 사이트 분산 (Joint vs Per-site) | ✅ ICC 1.7% → Joint 확정 |
| 1 | Open-Meteo historical-forecast 검증 | ✅ archive와 다른 product, MOS 학습 적합 |
| 2 | MOS bias 학습 가능성 | ⏳ 미실행 |
| 3 | MOS-PV 파이프라인 성능 | ⏳ 미실행 |

---

## 7. 진행 상태 (2026-05-03 기준)

### ✅ 완료

| 작업 | 산출물 | 결과 |
|---|---|---|
| Variance Gate (Gate 0) | `pv/notebooks/01_, 02_*.ipynb` | DSR ICC 1.0%, PV ICC 1.7~2.7% → **Joint LGBM 확정** |
| Data join (PV+GK-2A+ASOS) | `data/processed/training_set.parquet` | 265k 행, 8 사이트 |
| NGBoost baseline (Track A) | `pv/experiments/ngboost_baseline/` | **NMAE 6.20%, Portfolio 5.18%** ★ 목표 달성 |
| 잔차 사이트 cor 측정 | (분석 결과) | mean cor **0.03** → 독립 가정 합산 |
| Portfolio 합산 | `data/processed/portfolio_predictions.parquet` | μ, σ per timestamp |
| Quantile 5 + Trajectory 1000 | `data/processed/scenarios_*.parquet` | ACF 보존 multivariate, 4M trajectories |
| Re-weighting 시뮬 (A vs B) | `pv/experiments/realtime_simulation/` | **1h ahead σ -15.8%**, 6h+ 미미 |
| Thermal 운전 제약 역추정 | `data/processed/thermal_params.csv` | 호기당 90 MW (CLAUDE.md 460 정정) |
| LNG 변동비 검증 | `data/processed/lng_cost_monthly.csv` | 169 원/kWh (CLAUDE.md 160과 일관) |

### 🔄 진행 중

| 작업 | 상태 |
|---|---|
| TFT 학습 (Track B) | GPU 백그라운드, 결과 비교 대기 |

### ⏳ 다음 단계 (우선순위순)

```
1. src/decisions/lng_planner.py
   → 시나리오 → LNG MW 출력 plan (호기 분담, 운전 제약)
   → peaker (CG2,CG4) vs base (CG6,CG8) 자동 선택

2. src/eval/scenario_evaluator.py  
   → A vs B LNG 비용 정량화 (시나리오 가치 입증)
   → "실시간 갱신으로 X% 비용 절감" 핵심 metric

3. src/dashboard/ (Streamlit)
   → Overview KPI / 시간별 fan chart / 시나리오 표 / re-weight 비교 / LNG planner

4. MOS Stage 0 (시간 남으면)
   - src/preprocess/build_mos_dataset.py — Open-Meteo historical-forecast 4년치 수집
   - src/models/train_mos.py — XGBoost bias 보정 학습
   - Gate 1·2·3 검증

5. 최종 통합 + 보고서
   - plan_v1 → current/plan.md (v9) 승격
   - 발표 자료 + 데모 시나리오
```

---

## 8. 핵심 발견 요약

### 모델 품질
- NGBoost 5.18% portfolio NMAE — VPP 6.2% 대비 outperform
- 80% interval coverage 80.6% → 분포 *정확히 calibrated* (LNG 시나리오 신뢰 가능)

### 사이트 동질성
- 한반도 좁아 사이트 ICC 1.7% — *Joint 모델로 충분*
- 잔차 cor 0.03 — *독립 합산* 가능 → 포트폴리오 σ는 √8배만 증가 (위험 분산 효과)

### 실시간 갱신 가치
- 1h ahead σ -15.8% (re-weighting 효과)
- ACF lag-1 0.63 → 1~2h horizon에 가치 집중
- 6h+ horizon은 ACF 감쇠로 효과 미미 → 갱신 주기는 1h가 적정

### LNG 운영 제약 (실측)
- 호기당 Pmax 90 MW (가스터빈), 142/106 MW (증기터빈)
- 분당 plant Pmax 통상 774 MW (spec 920의 84%)
- Min up/down: 7~14h / 13~24h
- Cold start: 호기당 연 ~50회

---

## 8. 모델 자산 위치

```
data/
  processed/training_set.parquet       # PV+GK-2A+ASOS 학습용
  
src/
  preprocess/
    aggregate_gk2a_hourly.py           # v1→v2 시간 집계
    build_training_set.py              # 학습 데이터셋 빌드
  models/
    train_ngboost.py                   # Track A
    train_tft.py                       # Track B
  scenarios/                           # (예정)
    portfolio_aggregate.py             # 포트폴리오 합산
    generate_scenarios.py              # Quantile + Monte Carlo
    bayesian_reweight.py               # 시나리오 가중 갱신

pv/experiments/
  ngboost_baseline/test_predictions.parquet
  tft_baseline/                        # (학습 진행 중)
```
