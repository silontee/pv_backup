# Plan Changelog

## 2026-05-09 — LNG planner v4 spec 확정 (RG 폐기 + GT/ST 분리 + warm-up 발동)

**Layer A 입력 단순화** — `RG` (look-back 2시간 평균) 폐기, `instant_gap_now = μ_p1 - pv_actual` 직접 사용.
- 진단: `max(RG, instant_gap)` 에서 RG winning = 직전 부족 후 회복 중 → over-commit 발생
- 결과: over-commit 7247 → 5398 MWh (**−25%, −1849 MWh**), shortfall 1185 → 1197 (+1%, noise 수준)
- RG 는 대시보드/로그 보조 지표로만 잔존

**Layer B = warm-up only + GT 전용**
- 이전: Layer B 가 사실상 발동 0회 (P_min 30~40 MW 조건이 너무 빡셈)
- 변경: `WARMUP_MIN_GAP=5` 완화 + GT(`CG*`) 전용 candidate filter (ST 는 1시간 lead time 부족) + warm-up 호기 `P_DA=0` 유지
- 결과: warm-up 20회 발동 (전부 GT, ST 0회), Phase 2 marginal value −13.1% 가시화

**Layer A startup 도 GT 우선** — cold-start 빠른 호기 먼저 켜기

**최종 KPI** (test 2025, daytime 9-17, 365일 portfolio):
- shortfall **1197 MWh** / over-commit **5398 MWh**
- vs v2 baseline (CS2 single): −72.6%
- Phase 2 marginal: −13.1%

→ `plan/active/lng/plan.md` 갱신 (§4.4, §5.2, §6.1, §7.1, §8, §10, §13, §14, §15, §19, §21).

**2026-05-11 dashboard Home 페이지 제거**:
- `app.py` 의 PoC scope / KPI summary / 페이지 안내 등 *전체 콘텐츠 제거*
- 진입 시 자동으로 Page 1 (`📊 예측 비교`) 로 redirect (`st.switch_page`)
- 각 페이지에 CSS 추가하여 sidebar 에서 'app' 항목 숨김
- 이전 Home 콘텐츠는 `_app_old_backup.py` 에 보관
- dashboard_plan.md §0 / §4 / §10 갱신 (Home 제거 반영)

**2026-05-10 (밤 후속) dashboard_plan.md 현재 코드 기준 동기화**:
- §0 변경 이력에 2026-05-10 entry 추가
- §4.x 파일명 한국어로 정정 (`1_📊_예측_비교.py` 등 — 일부 영문 잘못 적혀있던 것 수정)
- §6.3 마커 컨벤션 정정 — code 와 일치:
  - 신규 가동 = ● red circle (이전 plan ▲ 잘못 적혀있던 것 수정)
  - 추가 발전 전환 = ▲ orange triangle
  - 예열 명령 = ★ red star (호기명 텍스트 동반)
  - Phase 2 발행 = ◆ royalblue diamond
  - 운영 모드 = ■ square
- §5 용어 매핑에 `forward_gap (FG1)` → "향후 1h 예상 차이" + `FGslope` → "변화율" 추가
- §8 데이터 흐름에 모든 헬퍼 함수 + 상수 (PRESET_DAYS / EVENT_DAYS / ALL_SITES) 명시
- §7 대표 날짜에 EVENT_DAYS 선정 기준 명시
- §10 코드 위치에 line count 명시 (총 1,618 lines)

**2026-05-10 (밤 후속) proposal §3, §4 디테일 보강**:
- §3 변수 선정 (3.1~3.8) — 약 3배 확장
  - 3.1 변수 선정 원칙 (3가지) 풀어쓰기
  - 3.2 사용 원천 데이터 4종
  - 3.3 PV 입력 변수군 — 기상 / 시간 / 사이트 / 실적 4 군 분리, 각 변수 *물리적 의미* + cf corr + 운영 시점 의미
  - 3.4 누적 R² (이미 있음) — 해석 4 항목으로 확장
  - 3.5 LNG 입력 변수군 — PV gap / baseline / 호기 spec 3 군
  - 3.6 사용하지 않은 변수 — 6 항목, *왜 안 썼는지* 풀이
  - 3.7 학습-예보 일관성 — 운영 swap 시나리오
  - 3.8 변수 선정의 운영 의미 종합
- §4 PV 데이터 EDA (4.1~4.7) — 약 3배 확장
  - 4.1 사이트 이질성 — capacity 분포 / hour 패턴 / 월 패턴 / cross-corr 4 subsection
  - 4.2 기상-출력 관계 — dsr 분위 box / 변수별 corr / 모델 영역 분리
  - 4.3 시간적 지속성 — fat tail / cf residual autocorr / cloud-pass 지속 / Phase 1 residual autocorr / 시간대별 변동성
  - 4.4 이벤트 분리 — z-score 분포 / dsr 무관 / 사이트별 outage heatmap / 함의
  - 4.5 calibration — 5-seed / 사이트별 / outage detector 정당성
  - 4.6 portfolio aggregation — NMAE 3 단계 변환 / outlier 영향 / 부정합 위험
  - 4.7 종합 — 데이터 관찰 → 모델 구조 매핑 표

**2026-05-10 (밤) proposal v2 — 11 chapter 구조 완전 재배치**:
- 기존: §1~§13 섞여 있음 (EDA / 모델 결정 / planner 가 하나로)
- 새 구조 (user 분류 기준):
  1. 필요성 및 목적
  2. 데이터 구성과 제약
  3. **변수 선정** (NEW — EDA 에서 분리)
  4. **PV 데이터 EDA** (성질만 — 사이트 이질성, 기상-출력, 시간적 지속성, 이벤트 분리, calibration, portfolio)
  5. **PV 예측모델 설계 및 채택 근거** (모델 비교 / sweep / 비채택 — EDA 에서 분리)
  6. LNG 데이터 EDA (호기 패턴 / spec / ramp 비대칭 / DEADBAND 근거 / over-commit 중요성)
  7. LNG 백업 계획 및 재배분 로직 (planner 설명)
  8. 통합 운영 구조 및 대시보드
  9. 성능 결과 및 기대효과
  10. 비채택 대안과 한계
  11. 향후 확장
- 기존 백업: `proposal_outline_v1_backup.md` (1341줄)
- 새 proposal_outline.md 약 580줄 — 정보량 유지하되 분류 명확

**2026-05-10 (저녁) proposal §2 (데이터 구성과 제약) + §3 (변수 선정) 신규 작성**:
- 새 chapter draft: `plan/active/pv/proposal_chap2_3_draft.md`
- 보조 EDA 2개:
  - `pv/eda_pv_model/D1_data_constraints.png` — 22 → 11 호기 필터링 + ESS 왜곡 + 4년 결측
  - `pv/eda_pv_model/D2_variable_selection.png` — 단일 corr / 누적 R² / dsr→cf 관계 / 시간대별 corr
- 핵심 메시지:
  - §2: 자체 22 호기 → 11 호기 (시간별 미제공 / ESS 왜곡 / 가동 축소 단계 제외) + perfect-foresight proxy framing
  - §3: 누적 R² zenith only 0.05 → +dsr 0.61 → +ASOS 0.63 → +site_oh 0.65 — dsr (GK-2A) 단일 변수만으로 12배 증가
- 출처: `plan/active/main/data_strategy.md` (SSOT 462줄)

**2026-05-10 (재작성) PV 모델 motivating EDA 5개 신설 — 데이터 자체에서 모델 결정 도출**:
- 기존 `01~04` (모델 선정 결과) 는 *결과* 였지 *근거 EDA* 가 아니었음 → 별도 `_selection_results/` 부록 폴더로 이동
- 새 motivating EDA 5개 신설 — *데이터 내재 특성에서 모델 구조 도출*:
  - **M1** 사이트 패턴 이질성 → AdaLN site/hour conditioning 정당화 (daily curve cross-corr 0.92~0.98)
  - **M2** PV 시간 척도 (residual autocorr / cloud-pass 지속) → Phase 2 / H=6 정당화 (lag 6h corr 0.42, cloud-pass median 1h)
  - **M3** 입력 변수 marginal value → ASOS + GK2A 조합 정당화 (dsr corr +0.85 압도)
  - **M4** outage / cloud-pass 통계 분리 → rule-based override 정당화 (z deep tail 분포)
  - **M5** Phase 1 residual 자기상관 → Phase 2 residual correction 가능 근거 (lag 1h corr 0.76)
- proposal §5A 전면 재작성 — *모델 선정 결과* 가 아니라 *데이터 내재적 모델 동기* 로
- §5A.7 부록: `_selection_results/01~04` 가 motivating EDA 와 *일치 방향* 임을 명시

**2026-05-10 (후속) proposal §5 두 part 재구성 + 폴더 이름 명확화**:
- `pv/eda_v4/` → `pv/eda_lng_planner/` 이름 변경 (LNG 임을 명확히)
- 두 EDA 폴더에 `README.md` 신설 (그림/스크립트/csv 목록 + proposal 인용 위치)
- proposal §5 를 두 part 로 재구성:
  - **§5A. PV 예측 모델 측 EDA** (NEW, eda_pv_model 4 그림 깊이 인용)
    - 5A.1 사이트 선정 / 5A.2 Backbone 비교 / 5A.3 Phase 2 sweep / 5A.4 Calibration & 5-seed / 5A.5 Outage 처리 (요약)
  - §5B. LNG planner 측 EDA (기존 §5.1~§5.11 → §5B.1~§5B.11)
- §5B.11 EDA 기반 설계 결정 요약 표를 PV 모델 + LNG 두 part 분리
- §10.5 EDA 산출물 위치 표를 두 폴더 분리

**2026-05-10 PV 모델 EDA 정리 폴더 신설**:
- `pv/eda_pv_model/` 폴더 신설 — *우리가 이미 한* PV 모델 측면 EDA / 의사결정 정리
- `_summary.md` 메타 문서 (10 챕터):
  - 사이트 선정 / 데이터 분리 / 입력 변수 / Backbone 비교 / Phase 2 architecture sweep / Outage 3단계 negative+rule-based / Calibration / 5-seed ensemble / 사용 노트북 14개 / 정직한 한계
- 신규 EDA 스크립트 4개 + 그림 + csv:
  - 01 사이트 선정 (NMAE / bias / Cov / capacity 분포)
  - 02 모델 비교 (LSTM/NGBoost/FT-T/ResMLP single/ensemble)
  - 03 Phase 2 architecture sweep (H, λ, gate, L)
  - 04 calibration + 5-seed ensemble 효과
- `proposal_outline.md §6.0` 추가 — 모델 측면 의사결정 한눈 표

**2026-05-09 (밤 후속) PoC 입증 EDA 8개 추가 + proposal 보강**:
- `pv/eda_lng_planner/` 폴더 신설 — 8 개 EDA 스크립트 + 그림 + csv
  - A. DEADBAND 민감도 (sweep 0~8) — 4 MW 정당화
  - B. 호기 ramp 비대칭 (4년 데이터) — over-commit 원인 입증
  - C. Forward gap 분포 + 임계값 정당화
  - D. 사이트별 outage 패턴 heatmap (8 사이트 × 365일 × 24h)
  - E. 시간대별 운영 부담
  - F. startup 분리 (real vs ramp)
  - G. Phase 2 일별 가치 분포
  - H. 호기 운전 패턴 (peaker 가정 폐기 근거)
- `proposal_outline.md` 대폭 보강:
  - §5 EDA 섹션 — 11 subsection 으로 확장 (각 EDA 결과 + 설계 결정 인과)
  - §7.5 outage 비채택 정량 결과 추가 (NMAE 표)
  - §8.3 DEADBAND 근거 (외부 정산 + 내부 sweep 양축)
  - §8.6 KPI 표 startup 분리 추가
  - §8.8 한계 — over-commit 메인 원인 (ramp_dn 비대칭) 명시
  - §10.4 planner spec 변화 5 단계 추적 표 추가
  - §10.5 EDA 산출물 위치 표

**2026-05-09 (밤) startup 분리 — startup_real vs startup_ramp**:
- planner v4: `startup_count` 단일 카운터 → `startup_real` (offline 호기 → 발전, 진짜 cold-start) + `startup_ramp` (이미 운전 중 + ΔP 시작) 분리
- 의미: 라벨 "신규 가동" 의 *진짜 의미* (cold-start) 를 구별. 이전 startup_count 는 두 케이스 혼합.
- 실측: **startup_real 1년 0~1회**, startup_ramp 931~941회 — 진짜 신규 가동은 거의 발생 X (분당 LNG 호기들이 대부분 baseline 운전 중)
- 운영 비용 인사이트: cold-start 비용 (~5천만원/회) 거의 회피 — 양호한 운영 패턴
- plan §10.6 신설 + §13.1 (KPI) / §14 P1 (부수 인사이트) 갱신

**2026-05-09 (저녁) DEADBAND=4 MW 도입 — system reserve cushion**:
- 근거: `(최종) 연료원별 보조서비스정산금_202603.xlsx` 분석 — LNG 가 전국 보조서비스(예비력) 정산의 ≈50% 담당
- 가정: PV portfolio actual_gap 의 4 MW 이내는 계통 1차/2차 예비력·AGC·타 조정자원이 자체 흡수. LNG는 그 이상의 marginal 만 event로 응답.
- 구현: `effective_gap = max(0, instant_gap - 4)`, Layer A 입력 + KPI 식 일괄 변경
- 결과: shortfall 1197→**442** (−63%), over-commit 5398→**640** (**−92%**), 비율 4.5x → **1.45x**
- vs v2 baseline: −72.6% → **−89.9%**, Phase 2 marginal −13.1% → **−19.1%**
- plan §4.5 신설 + §5.2/§7.1/§8.1/§13.1/§14/§15/§19/§20/§21 갱신

**2026-05-09 (오후) plan-vs-code mismatch 정정**:
- §6.2: CG5 분류 peaker → mid-merit (running_pct 28.22% ≥ 28% 코드 boundary)
- §6.3: priority 식 정정 — `헤드룸 × ramp_up × running_pct/100` (이전 plan은 running_pct 누락) + 표 값 실측 데이터로 갱신
- §8.1, §10.5, §21: Layer A 호기 startup 표현 "P_min 강제 출력" → "출력 = actual gap 잔여 그대로" (코드 `layer_a_startup` 가 `give = actual_gap_remaining`)
- §6.1: GT/ST cold-start 시간 "일반 산업 spec 추정" 명시 (호기별 실측 데이터 부재)
- §15.3: 옵션 C (slope-based recovery) — "이미 §7.3.3 에 적용됨" 으로 상태 변경
- §14 P1: KPI 진화 4단계 표 추가 ((a) v4 초기 → (d) 현재) — 단일 "이전→현재" 압축에서 누적 단계 framing 으로

다음: dashboard 갱신 → 계획서 작성.

---

## 2026-05-07 (오전 후속) — Outage Detection Layer 정식 채택

**가설 검증 — outage 분리 실험 (3가지 시도)**:
1. ❌ **Phase 1 retrain (실험 A)**: outage masked retrain 두 변형 (narrow 278건, wide 539건)
   - 결과: raw NMAE +0.14~0.16pp 악화. site-level mixed.
   - 원인: 0.2~0.4% 마스킹으로는 모델 분포 변화 미미. 모델은 outage를 noise로 robust하게 학습 중.
   - **폐기**.
2. ❌ **Integrated outage probability head (실험 C)**: Phase 2 architecture에 outage prob head 추가, joint loss
   - 결과: precision 3.3%, recall 41.7% (FP 폭증). 04-26 cloud-pass 13.79% → 20.24% 악화.
   - 원인: label imbalance (0.21% positive), outage가 weather feature로 예측 불가능한 site-specific 사건, 학습 sample 부족.
   - **폐기**.
3. ✅ **Rule-based outage override (실험 B) 채택**:
   - Detection: `cf<0.03 + mu_p1>0.20 + cloud<7 + z<-3 + neighbor confirm`
   - Recovery: actual_cf > 0.10 && 1h 지속 → blackout 해제
   - 결과: 41 confirmed outages → 587h blackout cover
   - **전체 NMAE 4.43% → 4.22% (−0.21pp), Normal 시점 손상 0.000pp**
   - **Cloud-pass: 03-23 23→2.89%, 04-26 13.7→3.81%, 05-04 16.1→3.79%**
   - **광양항 10/10-12 outage: 28.2% → 0.09%**

**Production 통합**:
- `pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet` 신설 (mu_phase2 = override 적용본, mu_phase2_raw 보존)
- `src/forecast/outage_override.py` — detection + recovery state machine
- `src/dashboard/lib/data_loader.py` — Phase 2 path 변경
- `src/decisions/thermal_planner_v2.py` — override ensemble 입력
  - shortfall 4527 → 4366 (−16.6% vs Phase 1, 이전 −13.5%에서 추가 −3.1pp)
  - 03-23/04-26/05-04 shortfall 각 1.26 / 5.29 / 1.36 MWh
- `plan/active/pv/model_final.md` §2.4 신설 (Outage Override Layer)
- `plan/STATUS.md`, `plan/CHANGELOG.md` 갱신
- `plan/IN_FLIGHT.md` 결정 기록 후 archive

**핵심 인사이트 (PoC novelty)**:
> weather-driven correction (Phase 1+2) 과 fault/outage override (rule-based) 를 책임 분리.
> 모델은 weather로 설명 가능한 변동만 책임지고, 비기상 운영 사건은 명시적 detection rule로 처리.
> Cloud-pass 와 outage 가 통계적/도메인적으로 분리 가능 (z-score deep tail vs cloud cover).

**부수 발견 (운영 보조 가치)**:
- 광양항세방 10/10-12 — 3일 종일 site shutdown (16건) 자동 검출
- 고흥만수상 12-14시 정기 outage 패턴 (5 events, 13건) — 점검/정비 의심
- 구미 06-15/16 — 2일 연속 outage 새 발견 (5건)

---

## 2026-05-07 — LNG fleet planner v4 + dashboard rule 단순화

**Phase 2 모델 확정**: 2-branch TCN, λ=2.0/T=1/τ=0, **L=12 EOD truncation**, **issue 7~18시 확장**
- 3-seed ensemble (`pv/experiments/phase2_2branch_g20_L12/`)
- Test NMAE 4.43% (vs Phase 1 4.75%, −0.32pp)
- Cloud-pass 정오~15시: 03-23 −15pp, 04-26 −6pp, 05-04 −9pp
- 4월 26일 19시 PV까지 reforecast 가능 (이전 16시 cutoff 해결)

**Thermal planner 단순화 (v2 production)**:
- 부분추종(ALPHA=0.6) **폐기** — 직관 어긋나고 Phase 2 가치 메커니즘 약함
- demand 항목 KPI에서 제외 — 실제 수요 데이터 부재 + PV 변동 흡수 효과만 평가
- 단순 룰: `|signal|<3 → 유지`, 그 외 100% 추종, ramp 30/release 4 MW/h
- KPI: shortfall = max(0, (μP1−actual) − correction), demand-free PV-only

**Thermal planner v3 (test)** — 4-state machine + reserve carry:
- KEEP / INCREASE / HOLD / DELAYED_RELEASE
- TH_NOISE=3, TH_HOLD=5, TH_ON=5, MIN_HOLD=1h, RELEASE=10 (균형형, grid 검증)
- Grid search 48 combos: shortfall −22% 가능 (TH_ON=4) but over_commit +21% trade-off
- HOLD 발동 거의 X (state 흐름 INCREASE → DELAYED_RELEASE 직행)
- 코드 마이그레이션 보류

**Thermal planner v4 (test) ★** — multi-unit fleet allocator (10 LNG):
- **P_DA proxy 결정**: 2025 actual LNG = baseline thermal schedule proxy (학습 X)
  - 부재 데이터를 가정 추가로 추정하지 않고 실측 직접 사용 + framing 명시
  - "true D-1 plan unavailable, actual baseline proxy" 한계 표명
- **Online allocation**: w ∝ Headroom × Ramp × Priority, normalize, cap by Headroom
- 결과 (v2 대비): SF 4527 → **4324 (−4.5%)**, sign_flips 58 → **0**, state_chg 868 → 276
- Cloud-pass: 03-23 28.7 → **12.2 (−57.5%)**, 04-26 −13.7%, 05-04 −20.9%
- Unit 분담: CS1 (1288 MWh) > CS2 (932) > CG7/8 (300+) — Headroom dominant
- Residual 835 MWh (offline screening 잠재 영역)

**Dashboard 3-page 갱신**:
- "Gap & Risk" → **"Gap and Backup Plan"**
- "Event Cases" 페이지 삭제 (small multiples + cloud-pass preset에 통합)
- "issue time" → "현 시각" (운영자 직관)
- 한글 라벨: 유지 / 출력 상승 / 출력 하강 (peaker/start/standby 제거)
- Page 2: 신호+correction 그래프가 현 시각에 따라 동적 (실선=실행, 점선=권장)
- Phase 2 EOD: small multiples [7, 10, 13, 15, 17] 시각 보임

**Plan 문서 active 디렉토리 정비**:
- `plan/active/pv/dashboard_plan.md` 전면 재작성 (현재 dashboard SSOT)
- `plan/active/pv/model_final.md` partial response 폐기 + demand 제거 반영
- `plan/active/lng/plan.md` v4 spec 확정 (P_DA proxy = 2025 actual, fleet allocator)
- `plan/active/pv/proposal_outline.md` 새 용어 반영

**메모리 추가**:
- `feedback_minimize_modeling_assumptions.md` — 부재 데이터를 학습 proxy로 만들지 말고 실측 사용

---

## 2026-05-03 - Stage 1 + 시나리오 파이프라인 완성, 실시간 가치 측정

**Stage 1 PV 모델 baseline 확보**:
- NGBoost 학습 완료 (`src/models/train_ngboost.py`)
  - TEST NMAE 6.20% (capacity 기준), 포트폴리오 NMAE 5.18% — **PoC 목표 6% 달성**
  - 80% coverage 80.6%, 95% coverage 92.9% — calibration 정확
  - 분포 출력 (Normal: μ, σ) — 시나리오 sampling 직접 가능
- TFT 학습 진행 중 (`src/models/train_tft.py`, GPU 백그라운드)

**EDA Gate 0 (사이트 분산 분해) PASSED**:
- DSR ICC = 1.0%, PV ICC = 1.7~2.7% (canonical 필터)
- 사이트 거의 동질 → **Joint LGBM with site_id 확정**, Bayesian/FiLM 기각
- 노트북: `pv/notebooks/01_site_variance.ipynb`, `02_pv_variance.ipynb`

**Data 파이프라인 정비**:
- `src/preprocess/build_training_set.py` — PV+GK-2A+ASOS unified table
- `src/preprocess/estimate_thermal_params.py` — 분당 LNG 운전 제약 4년 역추정
  - 호기당 Pmax 87~90 MW (가스터빈), 142/106 MW (증기터빈)
  - **CLAUDE.md 460 MW/호기는 오류 — 정정 필요**
- `src/preprocess/estimate_lng_cost.py` — LNG 효율 42.6%, 변동비 169 원/kWh (CLAUDE.md 160과 일관)

**시나리오 생성 파이프라인**:
- `src/scenarios/portfolio_aggregate.py` — 8 사이트 → 포트폴리오 분포 합산
  - 잔차 cor 측정 결과 0.03 → **독립 가정 합산**으로 정밀화
  - σ²_total = Σ σ_i² (이전 보수적 합 8 → √8 = 2.83배만 증가, 위험 분산 효과)
- `src/scenarios/generate_scenarios.py` — Quantile 5 + Multivariate trajectory 1000
  - ACF lag-1 0.63 보존하는 multivariate normal sampling
  - 4M trajectory rows, marginal calibration 정확 (90% interval 88%)
- `src/scenarios/bayesian_reweight.py` — 실시간 PV 실측 시뮬레이션 + A vs B
  - **핵심 발견**: 1h ahead σ -15.8%, MAE 미세 (+3.9%)
  - σ 감소가 LNG 비용 절감의 진짜 가치 (확신도 ↑ → 보수적 마진 줄임)
  - 6h+ horizon은 ACF 감쇠로 개선 미미 → re-weighting 가치는 *1~2h ahead*에 집중

**운영 아키텍처 확정**:
- Open-Meteo 6h cycle → re-inference (NGBoost 재실행, 시나리오 풀 교체)
- KOEN PV 1h cycle (시뮬레이션) → re-weighting (Bayesian filter on trajectories)
- KPX 신고 (D-1 forecast) vs 실시간 LNG 운영 (D-day gap-based) 분리

**plan 문서 갱신**:
- `plan/pv/plan_v1.md` — Stage 1 결과 + 운영 흐름 반영
- `plan/fuel/v1.md` 신설 — Stage 2 LNG 시나리오 + thermal params

## 2026-04-28 - PV 모델 옵션 C(MOS+PV 2-Stage) 채택 검토 시작

- `plan/pv/plan_v1.md` 신설 — Stage 0 MOS + Stage 1 PV(v8 Hybrid 2-Track) 2-Stage 구조 초안.
- v8 Hybrid 2-Track(LightGBM + Bayesian)은 Stage 1으로 흡수, 자산 보존.
- 결정 근거: `plan/decisions/2026-04-28-pv-mos-stage.md` (옵션 A/B/C 비교, GK-2A NaN 분석, 영흥 2024-01-15 검증 결과).
- `current/acceptance.md` 폐기 — v8 Hybrid 2-Track 종속, MOS 추가 후 재작성 예정.
- `current/plan.md`, `current/data_strategy.md` 상단에 PIVOT 배너. 본문 갱신은 EDA 3개 Go/No-Go 게이트 통과 후 일괄.
- 게이트: (1) Open-Meteo historical-forecast-api dawn/dusk 커버리지, (2) MOS 편향의 학습 가능성, (3) MOS-PV 파이프라인 성능. 세 게이트 모두 통과 시 plan_v1 → current(v9) 승격.

## 2026-04-22 - plan 디렉토리 재구조화

- `current/`와 `history/`를 분리해 살아있는 SSoT와 과거 버전을 구분했다.
- 원칙을 plan=Markdown, paper=LaTeX, raw=원문 보존으로 정리했다.
- EDA 결과를 계획으로 환류하기 위한 `plan/pv/eda_findings.md`를 신설했다.
- KOEN 회신과 초기 질문 브레인스토밍은 `raw/`에 보존하고, 참고자료는 `references/`로 분리했다.

## 2026-04-21 - KOEN 회신 수신

- KOEN 회신을 통해 SPC 상세자료와 화력 발전기 세부 파라미터 제공이 어렵다는 점을 확인했다.
- 프로젝트 성격을 운영 시스템 완성보다 제한된 데이터 기반 PoC와 의사결정 지원 가능성 검증으로 정렬했다.
- 공개 데이터, 역산 가능한 파라미터, 시나리오 기반 검증을 중심으로 계획을 조정해야 한다.
