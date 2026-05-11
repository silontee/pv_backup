# PV 예측 모델 EDA

> **proposal §6 (예측모델 설계) + §5 (EDA 결정)** 에 인용되는 PV 모델 측면 EDA 자료.
> LNG planner 측 EDA 는 별도 폴더 `pv/eda_lng_planner/` 참조.

## 그림 (proposal 인용)

| 파일 | 내용 | proposal 인용 위치 |
|---|---|---|
| `01_site_selection.png` | 사이트별 NMAE / bias / Cov / capacity 비중 | §6.0, §1.1 (사이트 선정) |
| `02_model_comparison.png` | LSTM / NGBoost / FT-T / ResMLP+AdaLN 비교 + 5-seed variance | §6.0, §6.1 (Backbone 채택) |
| `03_phase2_architecture.png` | H sweep / λ sweep / gate sweep / L horizon | §6.0, §6.2~§6.3 (Phase 2 설계) |
| `04_calibration_seed.png` | Cov80 / Cov95 / per-seed NMAE / 사이트별 calibration | §6.0, §6.4 (calibration) |

## 스크립트

| 파일 | 산출 |
|---|---|
| `01_site_selection.py` | 사이트별 NMAE / bias / capacity 분석 |
| `02_model_comparison.py` | 모델별 test_predictions.parquet 로 NMAE 산출 |
| `03_phase2_architecture.py` | H/λ/gate/L sweep 결과 정리 (model_final §1.5~1.7 인용) |
| `04_calibration_seed_ensemble.py` | Cov80/95 + per-seed variance |

## 데이터 (csv)

| 파일 | 내용 |
|---|---|
| `01_site_metrics.csv` | 사이트별 NMAE / bias / Cov80 / Cov95 / capacity |
| `02_model_comparison.csv` | 5종 모델 NMAE 비교 |
| `03_H_sweep.csv` | Phase 2 H={3,4,5,6} sweep |
| `03_lambda_sweep.csv` | 2-branch λ={1.0,1.5,2.0} sweep |
| `03_gate_sweep.csv` | Sparse gate (T, τ) sweep |

## 메타 문서

`_summary.md` — 모든 모델 측면 의사결정의 SSOT (11 챕터).

## 실행

```powershell
# 프로젝트 루트에서
python pv/eda_pv_model/01_site_selection.py
python pv/eda_pv_model/02_model_comparison.py
python pv/eda_pv_model/03_phase2_architecture.py
python pv/eda_pv_model/04_calibration_seed_ensemble.py
```
