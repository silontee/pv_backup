# PV 예측 연구 (Solar Power Forecasting)

태양광 발전량 예측 모델 연구 및 개발을 위한 디렉토리.

## Plan
- 세부 계획은 `plan/pv/` 에서 버전 관리 (v1, v2, ...)
- 현재 최신: `plan/pv/v1`

## 핵심 모델
베이지안 계층모형 + FiLM-조건부 GRU decoder-only (pv_predict.md)

## 디렉토리 구조
```
pv/
├── README.md          ← 이 파일
├── plan_v1.md         ← PV 예측 계획 (최신)
├── pv_predict.md      ← 모델 논문 (베이지안+FiLM-GRU 수식)
├── crawl_solar.py     ← 남동발전 태양광 크롤링 스크립트
├── notebooks/         ← EDA, 실험 노트북
├── src/
│   ├── data/          ← 데이터 로딩, 전처리
│   ├── models/        ← 모델 정의 (1단계 PyMC, 2단계 PyTorch)
│   ├── training/      ← 학습/검증/평가
│   └── utils/         ← 공통 유틸리티
└── experiments/       ← 실험 결과, 체크포인트
```
