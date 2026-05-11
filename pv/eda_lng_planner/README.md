# LNG Planner EDA

> **proposal §5 (EDA 결정) + §8 (LNG 백업 계획)** 에 인용되는 LNG 화력 백업 측면 EDA 자료.
> PV 예측 모델 측 EDA 는 별도 폴더 `pv/eda_pv_model/` 참조.

## 그림 (proposal 인용)

| 파일 | 내용 | proposal 인용 위치 |
|---|---|---|
| `A_deadband_curve.png` | DEADBAND 0~8 sweep — 4 MW 최적 plateau | §5.8, §8.3 |
| `B_ramp_asymmetry.png` | 호기별 상승/하강 한도 비대칭 (4년 데이터) | §5.7, §8.8 (over-commit 원인) |
| `C_forward_gap_distribution.png` | \|FG1\| / \|slope\| 누적분포 + 임계값 위치 | §5.4 (forward signal 임계값 정당화) |
| `D_site_outage_heatmap.png` | 사이트 × 365일 × 시각 outage heatmap | §5.2 (사이트별 outage 패턴) |
| `D_site_cf_violin.png` | 사이트별 cf 분포 violin | §5.2 |
| `E_hourly_burden.png` | 시간대별 부족분 / 응답 / 예열 명령 | §5.5 (운영 부담 시간대) |
| `F_startup_split.png` | 진짜 신규 가동 vs 추가 발전 전환 | §5.10, §8.6 (cold-start 회피) |
| `G_phase2_marginal.png` | Phase 2 일별 가치 분포 + Top 10 | §5.9 (Phase 2 가치 분포) |
| `H_unit_pattern.png` | 호기별 운전율 vs 평균 연속 운전 길이 | §5.6 (peaker 가정 폐기) |

## 스크립트

| 파일 | 입증 메시지 |
|---|---|
| `A_deadband_sensitivity.py` | DEADBAND=4 MW 정당화 (sweep 결과 + 보조서비스 정산금 근거) |
| `B_ramp_asymmetry.py` | 잔존 over-commit 의 메인 원인 = 호기 ramp_dn 비대칭 |
| `C_forward_gap_distribution.py` | TH_INCREASE=5, TH_NOISE=3, TH_SLOPE=2 데이터 기반 |
| `D_site_variance_outage.py` | outage 가 site-level 비기상성 사건임을 입증 |
| `E_hourly_burden.py` | 정오 ±2h 가 운영 부담 peak (예열 1~2h 선제) |
| `F_startup_split.py` | startup_real (cold-start 비용) 1년 0회 — 비용 효율 |
| `G_phase2_marginal_value.py` | Phase 2 가치는 평균이 아니라 *큰 변동일 집중* |
| `H_unit_operation_pattern.py` | peaker 도 평균 6~7h 연속 → mode 가정 폐기 정당화 |

## 데이터 (csv)

| 파일 | 내용 |
|---|---|
| `A_deadband_results.csv` | DEADBAND ∈ {0,2,4,6,8} sweep KPI |
| `A_deadband_posthoc.csv` | DEADBAND post-hoc 비교 |
| `B_ramp_stats.csv` | 호기별 상승/하강 ramp 통계 (p50/p90/p95/p99) |
| `D_site_stats.csv` | 사이트별 cf 분포 + outage rule 발동 통계 |
| `E_hourly_stats.csv` | 시간대별 1년 누적 KPI |
| `G_phase2_daily.csv` | 일별 Phase 2 marginal value |
| `H_unit_pattern.csv` | 호기별 운전율 / 연속 운전 길이 |

## 공통 헬퍼

`_common.py` — 한글 폰트, 호기 색 컨벤션 (`UNIT_COLORS`), 경로.

## 실행

```powershell
# 프로젝트 루트에서
python pv/eda_lng_planner/A_deadband_sensitivity.py
python pv/eda_lng_planner/B_ramp_asymmetry.py
# ... C ~ H
```
