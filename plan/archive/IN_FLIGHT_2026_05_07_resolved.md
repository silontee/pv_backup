# In-Flight Experiments — 결정 완료 (2026-05-07 새벽 → 06:55 resolved)

> **최종 결정 (2026-05-07 06:55)**:
> - ✅ **실험 B 채택** — Phase 2 + rule-based outage override (recovery 0.10 / 1h)
> - ❌ **실험 A 폐기** — Phase 1 retrain (narrow 278 + wide 539 mask 두 변형, 둘 다 raw NMAE +0.14pp 악화)
> - ❌ **실험 C 폐기** — Integrated outage prob head (precision 3.3%, FP 폭증, 04-26 cloud-pass 망가짐)
>
> 통합 작업 완료:
> - `pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet` (production)
> - `src/dashboard/lib/data_loader.py`, `src/decisions/thermal_planner_v2.py` → override ensemble
> - `plan/active/pv/model_final.md` §2.4 Outage Override Layer 신설
> - `plan/STATUS.md`, `plan/CHANGELOG.md` 갱신
>
> 결과 (Phase 2 + override):
> - 전체 NMAE 4.43 → **4.22%** (−0.21pp)
> - Cloud-pass: 03-23 23→**2.89%**, 04-26 13.7→**3.81%**, 05-04 16.1→**3.79%**
> - 광양항 10/10-12: 28.2→**0.09%**
> - Planner shortfall: 4527 → **4366** (Phase 2 가치 −16.6%, 이전 −13.5%에서 추가 −3.1pp)
>
> 본 문서는 archive 보존 — negative result (실험 A, C) 도 PoC 가치 있음.

---

# (이하 원본 — 진행 중 시점 기록 보존)

# In-Flight Experiments — 미반영 (2026-05-07 새벽)

> ⚠️ **중요**: 본 문서의 항목들은 **실험/검증 단계**입니다. 결과 종합 후 채택 여부 결정.
> Production (`src/dashboard/`, dashboard에 보여지는 model) 은 아직 변경 X.
> 결과 보고 좋으면 반영, 아니면 폐기.

---

## 🟡 Production 상태 (현재 dashboard에 연결된 것)

| 컴포넌트 | 위치 | 상태 |
|---|---|---|
| Phase 1 (frozen) | `pv/experiments/resmlp_adaln_v2_ensemble/` | ✅ 활성, **변경 X** |
| Phase 2 (final) | `pv/experiments/phase2_2branch_g20_L12/ensemble_test.parquet` | ✅ 활성, **변경 X** |
| Thermal planner | `src/decisions/thermal_planner_v2.py` (CS2 단일) | ✅ 활성 |
| Dashboard | `src/dashboard/{app, pages}.py` (3-page) | ✅ 활성 |
| Plan | `plan/active/{main, pv, lng}/*.md` | ✅ v3 spec까지 반영 |

---

## 🔬 실험 1 — Phase 1 outage-masked retrain (실험 A)

### 가설
> outage_v3.5 mask 로 학습 데이터의 운영 사건을 제거하면, Phase 1 모델이 weather-driven baseline 만 더 깨끗하게 학습.

### 방법
**Mask 정의 (옵션 B)**:
- Train period (~2024-10-16): 5 사이트 (고흥만수상, 광양항세방, 경상대, 삼천포, 구미) + cf<0.03 + dsr>500 + zenith<55 + cloud<5 + hour 10-15 + 3h 연속 → **236건**
- Val period (2024-10-16~): full v3.5 z-rule (cf<0.03 + mu>0.20 + cloud<7 + z<-3 + neighbor) → **84건**
- 합계: **278 rows** (학습 셋의 0.226%) — `data/processed/phase1_outage_mask.parquet`

### 실행 위치
```
src/models/train_resmlp_adaln_v3_outage_masked.py
→ pv/experiments/resmlp_adaln_v3_outage_masked/
  └ seed_{42,123,7,202,999}/{val_predictions,test_predictions}.parquet
  └ per_seed_summary.csv
```

### 진행 상태
- Background `b00w0mn7o` 실행 중 (5 seeds, 시작 05:37 KST)
- 예상 완료 ~05:50

### 비교 metric (결과 나오면 확인)
- Test 2025 raw NMAE
- Clean NMAE (mask 시점 제외)
- Cov80, Cov95 (calibration)
- Site-level NMAE
- Reference: 기존 v2 ensemble NMAE 4.747% (P1)

---

## 🔬 실험 2 — Phase 2 + outage override + recovery (실험 B, 완료)

### 가설
> outage 감지 시점부터 actual recovery까지 mu_phase2 = 0 으로 강제 → outage event NMAE 압도적 개선, normal 시점 손상 없음.

### 방법
- Detection: outage_v3.5 (cf<0.03 + mu>0.20 + cloud<7 + z<-3 + neighbor) → 51 confirmed targets
- Recovery: `actual_cf > 0.10 && 1h 지속` → blackout 해제
- Override: blackout 동안 mu_p2 := 0
- 비교: thr 0.10/0.15, hrs 1/2 sweep

### 결과 (recovery 0.10/1h 채택)

| 그룹 | n | P1 | P2 | **P2+override** | Δ |
|---|---|---|---|---|---|
| 전체 | 167,953 | 4.747% | 4.432% | **4.220%** | **−0.212pp** |
| Blackout 시점 | 587 | 68.6% | 55.7% | **0.39%** | **−55.3pp** |
| Normal 시점 | 167,366 | 4.50% | 4.23% | **4.23%** | **+0.000pp** ✓ |
| 03-23 전체 | 440 | 30.0% | 23.3% | **2.89%** | **−20.4pp** |
| 04-26 전체 | 624 | 15.0% | 13.7% | **3.81%** | **−9.8pp** |
| 05-04 전체 | 624 | 18.7% | 16.1% | **3.79%** | **−12.3pp** |
| 광양항 10/10-12 | 165 | 28.2% | 15.5% | **0.09%** | **−15.4pp** |

### 산출물
- 코드: `src/forecast/outage_override.py`
- 데이터: `pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet`
- recovery threshold 0.10 vs 0.15 효과 거의 동일 (둘 중 단순한 0.10 채택)
- recovery hours 2 너무 보수적 (4.293%, 정상 시점도 0으로 덮음)

### 검증 통과
- Normal 시점 NMAE 변화 +0.000pp → cloud-pass + 정상 시점 손상 X
- Blackout 시점만 깔끔히 처리

---

## 🔬 실험 3 — Thermal planner v3 (state machine, 4-state)

### 가설
> 단순 룰 → KEEP/INCREASE/HOLD/DELAYED_RELEASE state machine + reserve carry 가 Phase 2 가치 메커니즘 (hold/release timing) 활용.

### 결과 (grid search 완료)
- Best: TH_ON=5, TH_HOLD=5, MIN_HOLD=1, RELEASE=10
- shortfall 4527 → 3984 (−12%) but over_commit +14%
- HOLD 발동 0회 (transition rule 검토 필요)
- 사용자 의견: plan v3 spec 자체는 OK, 코드 마이그레이션 보류

### 산출물
- 코드: `src/decisions/thermal_planner_v3_test.py`, `thermal_planner_v3_grid.py`
- 결과: `pv/experiments/thermal_planner_v3_test/`

---

## 🔬 실험 4 — Thermal planner v4 (multi-unit fleet, 완료)

### 가설
> CS2 단일 → 10 LNG 호기 fleet allocator. P_DA proxy = 2025 actual 직접 사용.

### 결과
- shortfall 4527 → **4324 (−4.5%)** — fleet allocation 효과
- sign_flips 58 → 0
- Cloud-pass 03-23: 28.7 → 12.2 MWh (−57.5%) ★
- Unit 분담: CS1 1288 > CS2 932 > CG7 332 > CG8 302 (Headroom dominant)

### 산출물
- 코드: `src/decisions/thermal_planner_v4_test.py`, `build_lng_baseline.py`
- 데이터: `data/processed/lng_baseline_test.parquet`, `pv/experiments/thermal_planner_v4_test/`

---

## 📋 채택 결정 펜딩 — Phase 1 retrain 결과 보고 통합 결정

### Decision matrix

| 시나리오 | Phase 1 retrain 좋음 | Phase 1 retrain 안 좋음 |
|---|---|---|
| **Override (B) 좋음** ✓ | 통합 채택 (Phase 1 v3 + override + planner v3/v4 검토) | Phase 1 그대로 + override만 채택 |
| Override 안 좋음 | 거의 불가능 — A,B 함께 작동 | rollback (변화 없음) |

### 통합 시 갱신 대상

만약 결과 좋아서 채택하면:
1. Plan 문서:
   - `plan/active/pv/model_final.md` — outage detection layer 신설 (§3.x)
   - `plan/active/lng/plan.md` — outage_v3.5 spec 정식 §
2. Dashboard:
   - Phase 2 path → `ensemble_test_overridden.parquet` 사용
   - Page 1, Page 2 에 outage flag 시각화
   - KPI 카드 "운영 사건 자동 검출 N건"
3. Planner v2 입력도 override 적용본 사용
4. STATUS.md / CHANGELOG.md 갱신

### 보존 의무
- 옛 production 결과 (`ensemble_test.parquet`) 백업
- Plan 변경은 history 로 보존

---

## 🗂️ 새로 생성된 파일 (production 미반영)

```
src/
├── models/
│   └── train_resmlp_adaln_v3_outage_masked.py     ← Phase 1 retrain
├── decisions/
│   ├── thermal_planner_v3_test.py                  ← 4-state machine
│   ├── thermal_planner_v3_grid.py                  ← grid search
│   ├── thermal_planner_v4_test.py                  ← multi-unit fleet
│   └── build_lng_baseline.py                       ← P_DA proxy
└── forecast/
    └── outage_override.py                          ← Phase 2 후처리

data/processed/
├── lng_baseline_test.parquet                       ← LNG hourly + Avail
└── phase1_outage_mask.parquet                      ← 278 rows mask

pv/experiments/
├── phase2_2branch_g20_L12/
│   └── ensemble_test_overridden.parquet            ← override 적용
├── thermal_planner_v3_test/                        ← state machine 결과
├── thermal_planner_v4_test/                        ← fleet 결과
└── resmlp_adaln_v3_outage_masked/                  ← 진행 중
```

---

## ⚠️ 절대 안 건드릴 것

- `pv/experiments/resmlp_adaln_v2_ensemble/` (Phase 1 frozen production)
- `pv/experiments/phase2_2branch_g20_L12/ensemble_test.parquet` (Phase 2 production — overridden 본 별도 파일)
- `src/dashboard/lib/data_loader.py` (현재 path 그대로 유지)
- `src/decisions/thermal_planner_v2.py` (production planner)

---

## 🎯 다음 액션

1. **Phase 1 retrain 결과 대기** (b00w0mn7o, ~05:50 완료 예상)
2. 결과 분석:
   - vs 기존 v2 ensemble NMAE
   - calibration (Cov80/95)
   - site-level
3. 통합 결정:
   - Override 후처리만 (B): 보수적
   - Phase 1 retrain + Override (A+B): 적극적 (양쪽 다 좋아야)
   - 둘 다 안 좋음: rollback
4. 결정 후 plan + dashboard + STATUS 일괄 갱신

이 문서는 통합 결정 후 폐기 또는 archive 이동.
