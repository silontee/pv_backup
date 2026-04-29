# 태양광 변동성 대비 화력 백업 최적화 PoC

> 한국남동발전(KOEN) 자체설비 태양광 12 호기를 대상으로, 발전량 예측 불확실성을 정량화하고 LNG(분당) 백업 의사결정을 지원하는 PoC. AI·공공데이터 활용 경진대회 산출물.

## 1. 문제 정의

- **태양광은 변동비 0원 → 무조건 최대 발전, 전량 입찰 원칙**.
- 핵심 문제는 "내일 얼마나 나올지 모른다" — 예측 오차 10~30%, 급변 시 50%+.
- 화력은 이미 최소출력(~6,400MW)으로 24시간 가동 → 정확하지만 느림.
- LNG(분당 920MW)만 빠른 출력 조절 가능 → **태양광 변동성을 LNG로 흡수**.
- 따라서 핵심은 **태양광 예측 정확도 + 불확실성 정량화**, 그 결과로 LNG 기동/대기 추천 + KPX 가용용량 신고 최적화.

자세한 정의: [`plan/current/problem.md`](plan/current/problem.md)

## 2. 접근 — 2-Stage (MOS + PV)

학습 데이터(GK-2A 위성 실측)와 추론 데이터(Open-Meteo 예보)의 distribution shift를 명시적으로 다루기 위해 2-Stage 구조 채택.

```
Stage 0 (MOS):  Open-Meteo 예보 → bias-corrected GHI
                ↑ 학습: historical-forecast vs GK-2A 실측

Stage 1 (PV):   bias-corrected GHI + 기상 → PV 발전량 + 불확실성
                ↑ Hybrid 2-Track:
                  Track A — LightGBM (기준선)
                  Track B — Bayesian Hierarchical + FiLM-GRU (메인)
```

- 결정 근거: [`plan/decisions/2026-04-28-pv-mos-stage.md`](plan/decisions/2026-04-28-pv-mos-stage.md)
- PV 세부 계획: [`plan/pv/plan_v1.md`](plan/pv/plan_v1.md)
- 상위 계획 (v8, Stage 1로 흡수 중): [`plan/current/plan.md`](plan/current/plan.md)

## 3. 데이터 현황

### 학습용 (실측, 2022-01 ~ 2025-12, 48개월)

| 변수 | 소스 | 해상도 | 상태 |
|------|------|--------|------|
| **PV 발전량** | KOEN 홈페이지 크롤링 | 시간별, 호기별 | ✅ |
| **GHI (위성)** | GK-2A SWRAD (DSR) | 10분 → **시간 집계 완료** | ✅ |
| **기온/습도/풍속** | KMA ASOS 12 관측소 | 시간별 | ✅ |
| **LNG 발전량 (분당)** | KOEN 크롤링 | 시간별, 호기별 | ✅ |

### 예보용 (서비스 단계 입력)

| 변수 | 소스 | 비고 |
|------|------|------|
| GHI 예보 | Open-Meteo (ECMWF/GFS) | 무료, 좌표 기반 |
| 기상 예보 | KMA 단기예보 API | 5km 격자 |

### 사이트 (좌표 중복 collapse 후 11개 픽셀)

영흥, 삼천포, 여수, 영동, 예천, 구미, 탑선, 경상대, 고흥만수상, 광양항세방, 창원

데이터 SSoT: [`plan/current/data_strategy.md`](plan/current/data_strategy.md)

## 4. GK-2A 데이터 처리 (확정)

10분 위성 데이터를 시간 단위로 집계:

- **라벨 컨벤션**: hour-ending. 라벨 N = `[N-1:00, N:00)` 6 슬롯 평균. (KOEN solar_hourly와 정합)
- **DSR 집계**: `mean of non-NaN slots`. NaN은 측정 안 됨 (야간/위성 outage).
- **품질 플래그**: 검증 결과 `dsr_dqf == 1.0 ⇔ dsr.notna()` 100% 동치 → DQF 컬럼 제거. `sw_dqf`도 zenith로 99.98% 재현 가능 → 제거.
- **태양 천정각**: pvlib로 시간 중심(라벨 - 30분) 기준 사이트별 계산. 사이트 lat/lon 차이로 일출/일몰 시각 자동 분리 (예: 17:30 KST 영흥 89.2° vs 고흥 88.5°).
- **NaN 처리**: 야간/outage row는 학습 시 mask 또는 drop. weight magic number(0.5 등)는 사용 안 함 — site×time 차이는 모델(FiLM-GRU)이 자동 학습.

스크립트: [`src/preprocess/aggregate_gk2a_hourly.py`](src/preprocess/aggregate_gk2a_hourly.py)

### v2 출력 스키마 (`data/gk2a_v2/YYYYMM.csv`)

```
datetime_kst | site | dsr_mean | dsr_n_valid | n_slots | zenith_center | lat | lon
```

- 385,495 hourly 행 (11 사이트 × 4년)
- 49 월별 CSV (boundary 1개 포함)

## 5. 디렉토리 구조

```
pv_backup/
├── plan/                  # 계획 문서 (versionless current + history)
│   ├── current/           # SSoT: problem.md, plan.md, data_strategy.md
│   ├── decisions/         # ADR (방향 전환 기록)
│   ├── pv/                # PV 세부 계획 (plan_v1.md, eda_*.md, model_notes.md)
│   ├── fuel/              # LNG 백업 계획
│   ├── history/           # 과거 plan 버전
│   ├── references/        # 참고자료
│   └── CHANGELOG.md
├── pv/                    # PV 예측 연구
│   ├── notebooks/         # EDA 노트북
│   ├── paper/             # 모델 논문 (LaTeX)
│   ├── src/               # 모델·학습·평가 코드 (TBD)
│   └── experiments/
├── src/
│   ├── crawl/             # KMA, KOEN, GK-2A 크롤러
│   ├── preprocess/        # 데이터 집계 (gk2a v1→v2 등)
│   └── diagnose/          # 일회성 진단 스크립트
├── data/
│   ├── gk2a_raw/          # GK-2A NC 원본 (~47GB)
│   ├── gk2a_v1/           # 10분 사이트별 추출 CSV
│   ├── gk2a_v2/           # 시간 집계 + zenith CSV (★ 학습 입력)
│   ├── asos_hourly/       # 기상 관측
│   ├── solar_hourly/      # KOEN PV 발전량
│   ├── thermal_hourly/    # KOEN 화력 발전량
│   └── fuel_*/            # 연료 소비·조달
├── CLAUDE.md              # 프로젝트 컨텍스트 (Claude/AI 협업용)
└── README.md
```

## 6. 진행 상태 (2026-04-29)

- ✅ KOEN 회신 수신, PoC 범위·산출물 확정 (12 호기, 대시보드 포함)
- ✅ 데이터 수집 완료 (PV/LNG 발전량, GK-2A 4년, ASOS)
- ✅ GK-2A v2 시간 집계 완료 (zenith 포함)
- 🔄 **EDA 진행 중** (현재 단계)
- ⏳ MOS 검증 (`plan/pv/plan_v1.md` Gate 1~3)
- ⏳ Track A/B 학습 → 비교
- ⏳ 대시보드 + 시나리오 데모

## 7. 참고

- 모델 논문: [`pv/paper/pv_predict.tex`](pv/paper/pv_predict.tex) (Bayesian + FiLM-GRU)
- 변경 로그: [`plan/CHANGELOG.md`](plan/CHANGELOG.md)
- 외부 참고자료: [`plan/references/`](plan/references/), [`references.md`](references.md)
