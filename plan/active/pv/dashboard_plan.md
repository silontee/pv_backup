# Dashboard Plan (2026-05-10 갱신)

> 목적: `Phase 1 + Phase 2 + LNG planner v4 (Layer A/B + DEADBAND)` 구조를 PoC 시연/평가에 바로 연결할 수 있도록 Streamlit dashboard 의 화면 구조 + 데이터 흐름 + 한국어 용어를 확정한다.
>
> **본 문서는 `src/dashboard/` 코드의 SSOT (source of truth) 로, dashboard 수정 시 반드시 함께 갱신한다.**

---

## 0. 변경 이력

- **2026-05-11** — Home 페이지 (app.py) 제거 — 진입 시 자동으로 Page 1 (예측 비교) 로 redirect
  - app.py 의 PoC scope / KPI summary / 페이지 안내 등 *전체 콘텐츠 제거*. `_app_old_backup.py` 에 백업 보관.
  - app.py 는 `st.switch_page("pages/1_📊_예측_비교.py")` redirect 만
  - 각 페이지에 CSS 추가하여 sidebar 에서 'app' 항목 숨김 — sidebar 는 4 페이지만 표시
- **2026-05-10** — 본 문서를 *현재 dashboard 코드 기준* 으로 일괄 동기화
  - §4.x 의 파일명을 *한국어 파일명* 으로 정정 (`1_📊_예측_비교.py` 등)
  - §6.3 마커 정정 — 신규 가동 = ● (red circle), 추가 발전 전환 = ▲ (orange triangle), 예열 명령 = ★ (red star), Phase 2 발행 = ◆ (royalblue diamond)
  - §5 용어 매핑에 *향후 1h 예상 차이* (FG1) 표기 통일
- **2026-05-09 (밤)** — startup 분리 + Page 3 보기 모드 토글 + 호기 구성 표
  - planner v4: `startup_count` → `startup_real` (offline → 발전, 진짜 신규 가동) + `startup_ramp` (이미 운전 중 + ΔP 시작) 분리
  - 진짜 신규 가동은 1년 0~1회 / 추가 발전 전환 931~941회 — 대부분 이미 운전 중인 호기가 새로 발전 시작하는 케이스
  - Page 3 호기 구성 표 (가스터빈 CG1~8 / 증기터빈 CS1~2 + 기동시간)
  - Page 3 sidebar 보기 모드 토글: **일별 (선택한 날짜의 09~17시 흐름)** / **전체 (1년 누적)** *(2026-05-19: 라벨 명확화 — '시간별 timeline' 이 hour selector 가 있는 줄 오해 → 의미 그대로 풀어씀; 호기 이벤트 row 와 겹치던 범례 y=-0.05 → y=-0.18 + margin b=110)*
  - 전체 모드: 1년 KPI + 호기별 누적 + 월별 호기 stacked + 호기 운전 통계
- **2026-05-09 (저녁 후속)** — sidebar 라벨 한국어화 + pie/bar 색 통일
  - 페이지 파일명 한국어 (`1_📊_예측_비교.py` 등) — sidebar 한글 표시
  - 호기 색 컨벤션 `UNIT_COLORS` 를 `data_loader` 로 끌어올림 (단일 SSOT)
  - LNG Fleet Backup pie chart + 1년 누적 bar chart 색을 stacked area 와 통일 (호기별 동일 색)
- **2026-05-09 v4** — 전체 재구조화. v3 → v4 spec 반영 (DEADBAND, Layer A/B, 호기별 재배분, 한국어 용어)
  - 4 페이지 구조 확정 (예측 비교 / 실시간 차이 / 호기별 백업 / Phase 2 가치)
  - LNG planner v4 결과 (442 / 640 MWh) 반영
  - 모든 라벨 한국어화 (instant_gap → 실시간 차이, DEADBAND → 자체 흡수폭, etc.)
- 2026-05-07 v3 — Phase 1 + Phase 2 + balancing 구조 (CS2 single-cover 시절)

---

## 1. 목표

dashboard 가 답해야 할 4 가지 질문:

1. **Phase 2 가 D-1 예측을 어떻게 보정하는가** — 현 시각마다 일몰까지 갱신
2. **어디까지가 LNG 책임이고 어디부터 자체 흡수인가** — 자체 흡수폭 (DEADBAND) 가정
3. **어떤 호기가 얼마나 책임지고 언제 예열되는가** — 10 호기 fleet, Layer A/B
4. **당일 갱신 예측의 정량 가치는** — Phase 1 단독 vs Phase 1+2

핵심 메시지 흐름:
```
D-1 예측 → 당일 갱신 예측 → 자체 흡수폭 차감 → 호기별 재배분 + GT 예열
```

---

## 2. 구현 방식

- **Framework**: Streamlit + Plotly (PoC 단계, 운영 배포 X)
- **데이터 소스**: parquet (Phase 1 / Phase 2 ensemble + planner v4 log)
- **theme**: light / white background (`.streamlit/config.toml`)
- **port**: 8501 (단일 인스턴스)

### 실행
```powershell
# 프로젝트 루트에서
streamlit run src/dashboard/app.py
```

브라우저: `http://localhost:8501`

---

## 3. 대상 사용자

- 1차: PoC 심사자, 내부 의사결정자
- 2차: 향후 thermal balancing 운영자 (D-day 의사결정 보조)

---

## 4. 페이지 구조

```
app.py (진입점, 콘텐츠 없음)
   └ st.switch_page("pages/1_📊_예측_비교.py")  — 자동 redirect

📑 Pages (sidebar 한글 표시 — 파일명 한글이 sidebar 에 그대로 표시됨)
   ├ 1. 📊 예측 비교        — pages/1_📊_예측_비교.py  ★ 진입 시 자동 이동
   ├ 2. 📉 실시간 차이       — pages/2_📉_실시간_차이.py
   ├ 3. ⚙ 호기별 백업       — pages/3_⚙_호기별_백업.py
   └ 4. 📈 Phase 2 가치     — pages/4_📈_Phase2_가치.py
```

### 4.1 app.py 진입점 (Home 제거됨)

이전에 있던 Home 페이지 (PoC scope / KPI summary / 페이지 안내 등) 는 **2026-05-11 제거**.

```python
# app.py 핵심
st.set_page_config(...)
st.markdown("""<style>
[data-testid="stSidebarNav"] ul li:first-child {display: none;}
</style>""", unsafe_allow_html=True)
st.switch_page("pages/1_📊_예측_비교.py")   # 자동 redirect
```

각 페이지에도 동일 CSS 가 들어가 sidebar nav 에서 'app' 항목이 *완전히 숨김* 처리됨.

이전 Home 콘텐츠는 `src/dashboard/_app_old_backup.py` 에 보관 (필요 시 복원 가능).

### 4.2 Page 1 — 예측 비교 (`1_📊_예측_비교.py`)

**목적**: Phase 2 가 D-1 예측을 어떻게 보정하는가 + 정지 호기 검출은 언제

- **상단 메인 차트**:
  - σ band (Phase 1 ±80% 신뢰구간, gray dotted)
  - Phase 1 baseline (회색 line)
  - Phase 2 현재 발행 (royalblue diamond ◆) — issue=h 시각의 lead 1+α reforecast
  - Phase 2 직전 발행 (orange dashed) — issue=h-1 의 reforecast (직전 1h 갱신)
  - 실측 (검정, 현 시각까지만)
  - 현 시각 vline (royalblue dashed)
- **사이트 선택**: Portfolio (전체 8 사이트) / 단일 사이트
- **issue hour slider**: 7~18시
- **추가 차트**: 실시간 forecasting 지표 (rolling lead-1h Phase 2)

### 4.3 Page 2 — 실시간 차이 (`2_📉_실시간_차이.py`)

**목적**: 어디까지가 LNG 책임이고 어디부터 자체 흡수인가

#### 상단 (standalone, 480px)
- **포트폴리오 PV** — Phase 1 baseline + Phase 2 (현 시각 갱신, ◆) + 실측 + σ band
  - Forecast Replay 와 동일 스타일 (단, 직전 발행 orange 선 제외)

#### 아래 3-row subplot (720px)
- **② 실시간 차이** — instant_gap 막대 + 자체 흡수폭 ±4 MW 음영 + 빨강(부족>4)/파랑(초과<-4)/회색(이내) 색
- **③ LNG 대응 차이** — DEADBAND 차감 후 양수 부분만 (빨강 막대)
- **④ 향후 예상 차이** — Phase 2 갱신 forward gap 막대 + 운영 모드 markers (KEEP/INCREASE/DELAYED_RELEASE)
- 모든 row 에 issue 시각 vline (royalblue dashed)

#### 하단
- **색 / 운영 모드 가이드 박스**
- **일별 KPI 5-card**: 실시간 차이 합 / 계통 자체 흡수 / LNG 대응 차이 / LNG 응답 / 부족·과대보충
- **시간별 상세 표** (expander)

### 4.4 Page 3 — 호기별 백업 (`3_⚙_호기별_백업.py`)

**목적**: 어떤 호기가 얼마나 책임지고 언제 예열되는가

#### 4.4.0 상단 공통
- **호기 구성 표** — 2 행 (가스터빈 GT: CG1~CG8 / 증기터빈 ST: CS1~CS2 + 기동시간)
- **2-단계 의사결정 설명** (Layer A 실 발전 / Layer B 사전 대비)
- **Sidebar 보기 모드 토글** — 📅 일별 / 📈 전체

#### 4.4.1 일별 모드 (기본)

- **일별 KPI 5-card**: 부족분 / 과대보충 / 총 추가 발전 / 예열 명령 / 신규 가동(real) / 추가 발전 전환(ramp)
- **4-row 메인 차트** (900px):
  - ① **호기별 추가 출력 stacked area** — ST(파랑 2색, 아래) + GT(주황 8색, 위)
  - ② **차이 vs 응답** — 실시간 차이 / LNG 대응 차이 (빨강) / LNG 응답 (파랑) + 자체 흡수폭 음영
  - ③ **운영 모드 + 향후 1h 예상 차이** — KEEP/INCREASE/DELAYED_RELEASE markers + FG1 line
  - ④ **호기 이벤트 (3종)**:
    - **신규 가동 ●** (red circle, y=1) — `startup_real`
    - **추가 발전 전환 ▲** (orange triangle, y=2) — `startup_ramp`
    - **예열 명령 ★** (red star, y=3 + 호기명 텍스트) — Layer B warm-up
- **일별 호기별 추가 발전 표** (호기 / 종류 / dE / peak / 운전시간) + **비중 pie chart**
- **시간별 상세 expander** — 운영 모드 / 향후 신호 / 호기별 ΔP 컬럼 모두

#### 4.4.2 전체 모드 (1년 누적)

- **1년 KPI 5-card** (동일 구성)
- **호기별 1년 누적 추가 발전** 표 + bar chart
- **월별 호기별 stacked bar** — 어떤 달에 어느 호기가 활약했나
- **호기별 1년 운전 통계** 표 — 추가 발전 / 최대 출력 / 평균 출력 (운전 중) / 활성 시간 / 운전율

### 4.5 Page 4 — Phase 2 가치 (`4_📈_Phase2_가치.py`)

**목적**: 당일 갱신 예측의 정량 가치는

- **1년 KPI 비교 4-card**: 부족분 / 과대보충 / 총 추가 발전 / 예열 명령 (P1 vs P1+2)
- **상세 비교 표** (6 행): 부족분, 과대보충, 총 추가 발전, 신규 가동, 예열 명령, 추가 대응 모드
- **월별 비교 2-row bar** (Phase 1 단독 vs Phase 1+2)
  - row 1: 월별 부족분
  - row 2: 월별 과대보충
- **GT 예열 명령 분석** — 호기별 횟수 표 + bar chart
- **시간대별 예열 명령** bar chart (시간 분포)
- **큰 변동일 비교 표** (3 day) — 부족 / 과대 / 예열 명령
- **한 줄 결론 박스** — Phase 2 의 정량 가치 요약

---

## 5. 한국어 용어 매핑 (★ 일관성 유지)

| 영어 / 코드 명 | 한국어 (UI 표시) |
|---|---|
| instant_gap | 실시간 차이 |
| effective_gap | LNG 대응 차이 |
| forward_gap (FG1) | 향후 1h 예상 차이 / 향후 예상 차이 |
| FGslope | 변화율 |
| DEADBAND | 자체 흡수폭 |
| state | 운영 모드 |
| KEEP | 유지 |
| INCREASE | 추가 대응 |
| DELAYED_RELEASE | 점진 해제 |
| shortfall | 부족분 |
| over-commit | 과대보충 |
| ΔP / ΔE | 추가 출력 / 추가 발전 |
| total_alloc | LNG 응답 / LNG 실제 응답 |
| startup_real | 신규 가동 (offline 호기 → 발전 시작, 진짜 cold-start) |
| startup_ramp | 추가 발전 전환 (이미 운전 중 호기 + ΔP 시작) |
| Layer B warm-up | 예열 명령 |
| issue time | 현 시각 / 발행 시각 |
| Cloud-pass event | 큰 변동일 |
| Phase 1 only | Phase 1 단독 |
| Phase 1+2 | Phase 1+2 |
| outage override | 정지 호기 보정 |
| online cap | 여유 출력 |
| ramp_up / ramp_dn | 상승 한도 / 하강 한도 |
| cold-start | 기동시간 |
| custom 직접 선택 | 직접 선택 |

기술 명칭 (Phase 1 / Phase 2 / GT / ST / Layer A / Layer B) 은 그대로 (식별자 역할).

---

## 6. 색 / 시각 컨벤션

### 6.1 색 의미

| 색 | hex | 의미 |
|---|---|---|
| 빨강 | `#D32F2F` | PV 부족 / 추가 대응 모드 / 부족분 / LNG 대응 차이 |
| 파랑 | `#1976D2` | PV 초과 / 점진 해제 / Phase 2 발행 / LNG 응답 |
| 회색 | `#9E9E9E` | 자체 흡수폭 이내 / 유지 모드 / Phase 1 baseline |
| 주황 그라데이션 | `#FFC107~#8B0000` | GT 호기 8 색 (CG1→CG8) |
| 파랑 진하기 | `#1976D2`, `#0D47A1` | ST 호기 2 색 (CS1, CS2) |
| 검정 | `black` | 실측 PV |

### 6.2 호기별 색 (★ 모든 차트 통일)

`data_loader.UNIT_COLORS` dict 가 SSOT. stacked area / pie / bar 모두 같은 호기 = 같은 색.

```python
GT_COLORS = ['#FFC107','#FFB300','#FFA000','#FF8F00',
             '#FF6F00','#E65100','#BF360C','#8B0000']   # CG1~CG8 gradient
ST_COLORS = ['#1976D2','#0D47A1']                       # CS1, CS2
UNIT_COLORS = {**{u: GT_COLORS[i] for i,u in enumerate(GT_UNITS)},
               **{u: ST_COLORS[i] for i,u in enumerate(ST_UNITS)}}
```

stacked area 순서:
- ST 먼저 (아래) — CS1 / CS2 (파랑 진하기, 큰 헤드룸)
- GT 나중 (위) — CG1~CG8 (주황 gradient)

### 6.3 마커 / 심볼 컨벤션

| 마커 | 심볼 | 색 | 의미 | 코드 컬럼 |
|---|---|---|---|---|
| ● circle | red | `#D32F2F` | 신규 가동 (offline → 발전) | `startup_real` |
| ▲ triangle-up | orange | `orange` | 추가 발전 전환 (이미 운전 중) | `startup_ramp` |
| ★ star | red-orange | `#E65100` | 예열 명령 (Layer B, GT 전용) — 호기명 텍스트 동반 | `warm_up_unit` |
| ◆ diamond | royalblue | `royalblue` | Phase 2 발행 (현 시각 reforecast) | (Page 1, 2) |
| ■ square | gray/red/blue | state 색 | 운영 모드 (KEEP/INCREASE/DELAYED_RELEASE) | `state` |

---

## 7. 대표 날짜 (모든 페이지 공통)

`data_loader.PRESET_DAYS` / `EVENT_DAYS` 에 정의. 변경 시 본 문서도 갱신.

### 7.1 안정 평일 (4 계절, Phase 1 NMAE 1% 내외)

```python
PRESET_DAYS = {
    "봄 — 2025-03-20":   "2025-03-20",
    "여름 — 2025-08-22": "2025-08-22",
    "가을 — 2025-10-20": "2025-10-20",
    "겨울 — 2025-12-22": "2025-12-22",
}
```

### 7.2 큰 변동일 (Phase 2 가치 다양성)

```python
EVENT_DAYS = {
    "★ 2025-09-06": "2025-09-06",   # Phase 2 가치 극대 (-61%, GT 예열 3회)
    "2025-08-20":   "2025-08-20",   # 여름 medium (예열 1회)
    "2025-03-23":   "2025-03-23",   # 1년 최대 cloud-pass (eff_gap 200 MWh)
}
```

선정 기준 (2026-05-09):
- 가을 (09-06): Phase 2 가치 극대 + warm-up 시각화 효과
- 여름 (08-20): warm-up 발동 + INCREASE 모드 medium
- 봄 (03-23): 1년 최대 effective_gap (Phase 2 한계 사례 — 정직 시연)

---

## 8. 데이터 흐름

```
data/processed/
  ├ phase1: pv/experiments/resmlp_adaln_v2_ensemble/ensemble_test.parquet
  ├ phase2: pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet
  ├ planner P1: pv/experiments/thermal_planner_v4/log_phase1.parquet
  ├ planner P1+2: pv/experiments/thermal_planner_v4/log_phase1plus2.parquet
  └ summary: pv/experiments/thermal_planner_v4/summary.csv
       ↓
src/dashboard/lib/data_loader.py
  ├ load_phase1, load_phase2                          # 모델 결과 로딩
  ├ load_planner(mode), load_planner_both             # planner log
  ├ load_planner_summary                              # 1년 요약 csv
  ├ portfolio_phase1, portfolio_phase2_at_issue,
  │ portfolio_phase2_at_issue_with_fallback           # 시간별 portfolio 변환
  ├ kpi_year(log)                                     # 1년 누적 KPI dict
  ├ kpi_monthly(log)                                  # 월별 KPI 집계
  ├ warmup_distribution(log)                          # 호기별 예열 횟수
  ├ daily_unit_dE(log, date_sel)                      # 특정 날짜 호기별 ΔE
  ├ fleet_share_year(log)                             # 호기별 1년 누적 + 비중
  ├ monthly_unit_dE(log)                              # 월별 × 호기별 ΔE (long format)
  ├ unit_year_stats(log)                              # 호기별 1년 운전 통계
  ├ find_normal_day(df_p1)                            # 가장 안정 발전일 자동 검출
  └ 상수: DEADBAND_MW=4, PV_PORTFOLIO_MW=77.28, LNG_FLEET_MW=920,
          GT_UNITS, ST_UNITS, LNG_UNITS,
          GT_COLORS, ST_COLORS, UNIT_COLORS,
          PRESET_DAYS, EVENT_DAYS, ALL_SITES
       ↓
페이지별 streamlit chart (plotly)
```

---

## 9. 수정 규칙 (★ 본 문서 갱신 약속)

dashboard 코드 수정 시 본 문서를 **반드시 함께 갱신** 한다:

- 페이지 추가/삭제 → §4 페이지 구조
- chart row / column 변경 → §4.x 해당 페이지 항목
- 새 한국어 용어 도입 → §5 매핑 표
- 색/마커 컨벤션 변경 → §6
- 대표 날짜 변경 → §7
- 데이터 소스 / loader 함수 변경 → §8

각 변경마다 §0 변경 이력에 한 줄 추가.

---

## 10. 코드 위치

```
src/dashboard/
├ app.py                          # 진입점 (auto redirect to page 1, ~22 lines)
├ _app_old_backup.py              # 이전 Home 콘텐츠 (참고용 백업, 185 lines)
├ lib/
│  └ data_loader.py               # 모든 데이터 로딩 + 상수 + 헬퍼 + UNIT_COLORS (290 lines)
├ pages/
│  ├ 1_📊_예측_비교.py             # Forecast Replay (~366 lines, sidebar CSS 포함)
│  ├ 2_📉_실시간_차이.py            # Gap & DEADBAND (~276 lines)
│  ├ 3_⚙_호기별_백업.py            # LNG Fleet Backup, 보기 모드 토글 (~334 lines)
│  └ 4_📈_Phase2_가치.py           # Phase 2 Value (~190 lines)
└ .streamlit/config.toml          # light theme (프로젝트 루트)
```

---

## 11. 한 줄 요약

> 본 dashboard 는 `D-1 예측 → 당일 갱신 → 자체 흡수폭 차감 → 호기별 재배분 + GT 예열` 흐름을 4 페이지로 시각화하며, **dashboard 수정 시 본 문서를 함께 갱신** 한다 (§9).
