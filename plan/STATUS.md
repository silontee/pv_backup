# 프로젝트 현재 상태 (2026-05-07 새벽 갱신)

> **2026-05-07 04:00~07:00 update**: outage detection layer (rule-based override) 정식 채택.
> Phase 1 retrain (실험 A) 폐기. Integrated outage prob head (실험 C) 폐기. **Phase 2 + post-processing override (실험 B) 채택**.

> 한 화면에 보는 진척도. 자세한 건 `plan/active/`, `plan/CHANGELOG.md`.

---

## 큰 그림

```
[Phase 1]  D-1 baseline forecast (ResMLP+AdaLN ensemble, 5-seed, frozen)
              ↓
[Phase 2]  Intraday re-anchored correction (2-branch TCN, 일몰까지 reforecast)
              ↓
[LNG Planner]  Fleet-level state machine + 호기별 재배분 (CS1+CS2+CG1~8)
              ↓
[Dashboard]  3-page Streamlit (Forecast Replay / Gap and Backup Plan / Balancing Response)
```

운영 (D-day):
- D-1 17시: Phase 1 baseline 발행 (frozen)
- D-day 매 시간 (07~18): Phase 2 갱신 (현재 시각부터 일몰까지 reforecast)
- Planner: 매 시각 realized + forward gap → 4-state (KEEP/INCREASE/HOLD/DELAYED_RELEASE) → 호기별 ΔP 배분

---

## ✅ 완료된 작업

### Phase 1 — Day-Ahead Baseline (frozen)
- [x] ResMLP+AdaLN v2 ensemble (5 seeds): `pv/experiments/resmlp_adaln_v2_ensemble/`
- [x] Test 2025 NMAE 4.75%, Cov80 85%, Cov95 94%
- [x] Output: 24h × 8 site (μ_p1, σ_total)

### Phase 2 — Intraday Reissued Forecast
- [x] 2-branch TCN with soft gate (λ=2.0, T=1.0, τ=0)
- [x] L=12 EOD truncation (issue 시점부터 일몰까지)
- [x] Issue range: **7~18시** (확장 완료, 4월 26일 19시 PV까지 cover)
- [x] 3-seed ensemble: `pv/experiments/phase2_2branch_g20_L12/`
- [x] Test NMAE 4.43% (Phase 1 대비 −0.32pp)
- [x] **+ Outage Override Layer (rule-based, 2026-05-07 신설)**
  - Detection: `cf<0.03 + mu_p1>0.20 + cloud<7 + z<-3 + neighbor confirm`
  - Recovery: `actual_cf > 0.10 && 1h 지속` → blackout 해제
  - 41 confirmed outage events → 587h blackout cover
  - **전체 NMAE 4.43 → 4.22% (−0.21pp), Cloud-pass days 압도적 개선**
- [x] **최종 Production: `ensemble_test_overridden.parquet`** (mu_phase2 = override 적용본)

### Thermal Planner — 진화 history
- [x] **v1** (deprecated): reserve-based dispatch with stylized demand
- [x] **v2** (현재 production, dashboard 연결): CS2 단일 호기, 단순 룰 (잡음 컷 + 100% 추종 + ramp/release)
  - Test 2025: shortfall 4527 MWh, Phase 2 가치 −13.5% vs Phase 1만
- [x] **v3 test** (`thermal_planner_v3_test.py`, 코드 마이그레이션 X):
  - 4-state machine (KEEP/INCREASE/HOLD/DELAYED_RELEASE) + reserve carry
  - Grid search 완료 (TH_ON/TH_HOLD/MIN_HOLD/RELEASE_RATE)
  - 결과: shortfall −22% 가능 (TH_ON=4) but over_commit +21% trade-off
  - **HOLD 발동 거의 X** (state 흐름 INCREASE → DELAYED_RELEASE 직행)
- [x] **v4 test** (`thermal_planner_v4_test.py`, 코드 마이그레이션 X):
  - **Multi-unit fleet allocator** — 10 LNG 호기 (CS1, CS2, CG1~8)
  - P_DA proxy: **2025 actual = baseline** (학습 단계 없음, plan §4.1.1)
  - Online allocation: w ∝ Headroom × Ramp × Priority (plan §9.3)
  - 결과 (P1+2 vs v2): SF 4527 → **4324 (−4.5%)**, sign_flips 58 → 0, state_chg 868 → 276
  - Cloud-pass: **03-23 28.7 → 12.2 MWh (−57.5%)**, 04-26 −13.7%, 05-04 −20.9%
  - Unit 분담: CS1 (1288 MWh) > CS2 (932) > CG7 (332) > CG8 (302) > CG6 (190) > 나머지
  - HOLD 발동 0회 (v3와 동일)
  - 잠재 offline 영역 (residual) 835 MWh

### Dashboard (`src/dashboard/`, Streamlit)
- [x] **Home (`app.py`)**: 시스템 흐름 + balancing rule expander + 4계절 + cloud-pass preset
- [x] **Page 1 — Forecast Replay**: Phase 1 vs Phase 2 vs actual + 일몰까지 small multiples
- [x] **Page 2 — Gap and Backup Plan**: 3 gap subplot + 실시간 운영계획 (현 시각 기반 동적)
- [x] **Page 3 — Balancing Response**: 1년 KPI + Reactive vs Predictive 비교
- [x] 사이드바: 4 평일 + 3 cloud-pass preset, 현 시각 7~18 슬라이더
- [x] 한글 라벨 (state: 유지 / 출력 상승 / 출력 하강)
- [x] 그래프 자동 y-range, cliponaxis, 옅은 색 = 권장(앞으로)

### 데이터 / 인프라
- [x] LNG hourly (2022~2025) parsing: `data/thermal_hourly/`
- [x] Thermal unit profile (10 LNG 호기 spec): `data/processed/thermal_unit_profile.csv`
- [x] LNG baseline + Avail (2025): `data/processed/lng_baseline_test.parquet`
- [x] PV training set (Phase 1 frozen + Phase 2 train + test 분리)

### 계획 문서 (`plan/active/`)
- [x] `main/framing.md` — 전체 PoC framing
- [x] `pv/model_final.md` — PV 모델 spec (2-stage residual cascade)
- [x] `pv/dashboard_plan.md` — dashboard SSOT, 현재 구현 반영
- [x] `pv/proposal_outline.md` — 공모전 outline
- [x] `lng/plan.md` — fleet planner spec (v4 spec, P_DA = 2025 actual proxy 확정)

### 메모리 (다음 세션 보존)
- [x] PV 시나리오 철학, data_strategy SSOT, 노트북 EDA 룰
- [x] PoC 컨텍스트 (KOEN 공모전), Codex 협업 패턴
- [x] **가정 추가 대신 실측 proxy 사용 원칙** (P_DA = 2025 actual 결정 계기)

---

## 🔄 진행 중 / 미완

- 정식 코드 마이그레이션 (v2 → v3 → v4): user pause 상태
- v4 결과를 dashboard에 반영 X (현재 dashboard는 v2 기반)

---

## 📋 다음 작업 후보

### A. HOLD state 미발동 원인 분석 (간단, 1시간)
- v3, v4 모두 HOLD 발동 0회 (KEEP → INCREASE → DELAYED_RELEASE 흐름)
- plan §8.2/§8.3 transition rule 점검
- 의도면 plan에서 HOLD 제거 또는 진입 조건 완화

### B. §10 Offline Startup Screening 도입 (중간, 4~6시간)
- v4 P1+2 잠재 residual 835 MWh
- evening peaker (CG2/CG4) 가동 시 흡수 가능
- plan §10에 정의된 screening rule (E_need, Cost_start, Benefit) 구현
- C_shortfall, H_commit 등 외부 변수 결정 필요
- 필요 시 plan §10 추가 정량화 (StartupPenalty 식 등)

### C. Priority 데이터 기반 재산출 (간단, 2시간)
- 현재 plan §11.2 priority 임의값 (CS2=1.0, CG6/7/8=0.7~0.8 등)
- v4 결과 보면 CS1이 가장 많이 분담 (headroom 40 MW 효과)
- backup_score 또는 headroom 비례로 priority 자동 산출
- A/B test로 KPI 차이 확인

### D. Over-commit 줄이기 (중간, 3~4시간)
- v3/v4 모두 over_commit +14~21% 증가 (v2 대비)
- 원인: 음수 dispatch 안 만듦 + 보수적 release
- 옵션:
  - RELEASE_RATE 호기별 차등 (peaker는 빨리, baseload는 천천히)
  - 음수 RG 시 release 가속 strength 증가 (현재 1.5x → 2~3x)
  - 호기별 turn-down 명시적 허용

### E. Dashboard에 호기별 분담 시각화 (중간, 4~6시간)
- v4 fleet allocator 결과를 dashboard에 보이게
- 옵션:
  - Page 2 "실시간 운영계획"에 호기별 막대 추가
  - 새 페이지 "Fleet Allocation" — 일별 호기별 timeline + 누적 KPI
  - Sankey/stacked bar로 ΔP_req → 호기별 ΔP 흐름

### F. v4 정식 코드 마이그레이션 (큰 작업, 8~12시간)
- 전제: v4가 채택 결정
- thermal_planner_v2.py 폐기 → v4 신규
- dashboard data_loader 갱신 (호기별 컬럼 처리)
- planner KPI 재계산 + 모든 페이지 갱신
- 회귀 테스트 (이벤트 day 결과 보존)

### G. Plan 문서 v4 결과 반영 (간단, 2시간)
- v4 결과 (SF −4.5%, sign_flip 0, 03-23 −57.5%) 를 plan/active/lng/plan.md §16 KPI 섹션에 추가
- proposal_outline 업데이트
- model_final.md "다음 단계"의 dashboard 항목 v4 반영

### H. Cloud-pass 04-26 v4 vs v2 일별 비교 분석 (조사, 1~2시간)
- 04-26은 v4가 −13.7% 만 개선 (다른 두 이벤트 −20~57% 대비 작음)
- 일별 timeline 그려서 어디서 v4가 부족한지 진단
- 인사이트로 plan 정교화 가능

### I. 경제효과 (NetBenefit) 계산 (중간, 4~6시간) — plan §14
- 현재 KPI: shortfall, over_commit (MWh)
- 추가: 운영비 (Σ ΔP × VC), startup 비용 (호기 켜기 시), 절감 효과
- 한국 SMP 또는 신재생 페널티 정보 필요
- 결과: "v4 도입 시 LNG 운전비 X% 감소" 같은 직접적 PoC value

### J. 추가 KOEN 공모전 산출물 정비
- proposal 본문 작성 (proposal_outline.md 기반)
- dashboard 화면 캡처
- 발표용 슬라이드

---

## 핵심 산출물 위치

```
data/processed/
├── training_set.parquet                  # PV 학습 세트
├── lng_baseline_test.parquet             # ★ 신규 — 2025 LNG actual = baseline proxy
├── thermal_unit_profile.csv              # 10 LNG 호기 spec
└── thermal_params.csv                    # 분당 LNG 운전 제약

pv/experiments/
├── resmlp_adaln_v2_ensemble/             # ★ Phase 1 (5 seeds, frozen)
├── phase2_2branch_g20_L12/               # ★ Phase 2 final (3 seeds, issue 7~18, L=12)
├── thermal_planner_v2/                   # 현재 dashboard 연결 (CS2 단일)
├── thermal_planner_v3_test/              # 4-state grid search 결과 + best 후보 csv
└── thermal_planner_v4_test/              # ★ fleet allocator (10 LNG, online only)

plan/active/
├── main/framing.md
├── pv/model_final.md                     # 2-stage residual cascade spec
├── pv/dashboard_plan.md                  # 현재 dashboard SSOT
├── pv/proposal_outline.md
└── lng/plan.md                           # ★ v4 fleet planner spec (P_DA = actual)

src/
├── models/
│   ├── train_resmlp_adaln_v2_ensemble.py # Phase 1
│   └── train_phase2_2branch.py           # Phase 2 final
├── decisions/
│   ├── thermal_planner_v2.py             # 현재 production (CS2 단일)
│   ├── thermal_planner_v3_test.py        # 4-state, single unit (TEST)
│   ├── thermal_planner_v3_grid.py        # grid search
│   ├── thermal_planner_v4_test.py        # ★ fleet allocator (TEST)
│   └── build_lng_baseline.py             # ★ P_DA + Avail 빌더 (학습 X)
└── dashboard/
    ├── app.py
    ├── lib/data_loader.py
    └── pages/{1_Forecast_Replay, 2_Gap_and_Backup_Plan, 3_Balancing_Response}.py
```

---

## 핵심 발견 (PoC 평가/논문 hook)

1. **2-stage residual cascade**: Phase 1 frozen (안정) + Phase 2 re-anchored (즉응) — 두 단의 분업이 핵심
2. **Phase 2 일몰까지 reforecast**: short-horizon patch (L=3) → EOD truncation (L=12) 가 cloud-pass 흡수에 결정적
3. **단순 룰의 가치**: 부분추종(α) 폐기 + 잡음 컷 + 100% 추종 + ramp/release만으로 충분, 운영자 직관과 일치
4. **Fleet allocation이 단일 호기보다 우수**: 03-23 같은 큰 cloud-pass에서 −57.5% (단일 호기로는 ramp 못 따라감, fleet으로 흡수)
5. **Headroom > Priority**: CS1이 priority 0.7임에도 분담 1등 (headroom 40 MW). 운영 직관과 부합
6. **sign_flips 완벽 제거**: v2 58 → v3/v4 **0**. 운영 안정성 압도적 개선
7. **데이터 기반 threshold**: forward gap p75 → TH_LOW=3, p99 → TH_ON 결정 (v3 grid 검증)
8. **가정 최소화 원칙**: D-1 plan, demand 같은 부재 데이터를 학습 proxy로 만들지 말고 actual 직접 사용 + framing 명시

---

## 진행 명령 (다음 세션)

```bash
# 현 상태 dashboard 띄우기 (v2 + ngrok)
streamlit run src/dashboard/app.py

# v4 fleet 결과 재현
python src/decisions/thermal_planner_v4_test.py

# baseline 재빌드 (LNG 데이터 갱신 시)
python src/decisions/build_lng_baseline.py
```

---

## Pause point (2026-05-07)

- LNG planner spec → v4 fleet allocator 완성 + 실험 검증
- v4가 v2 대비 모든 핵심 지표에서 개선 (SF, state_chg, sign_flips, cloud-pass) 확인
- 코드 마이그레이션은 보류 (사용자 결정 대기)
- 다음 세션에서 위 [A~J] 후보 중 우선순위 결정 후 진행
