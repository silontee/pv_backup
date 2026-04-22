# Plan v8 Acceptance Checklist

> 기준일: 2026-04-22
> 참조: `plan/main/plan_v8.md`

## A. Data Integrity

- [ ] 시간 인덱스 연속성 검증 완료
- [ ] 중복률 임계치 검증 완료
- [ ] 결측률 임계치 검증 완료
- [ ] DQF 클래스 분포 검증 완료
- [ ] 사이트별 최소 유효 샘플 기준 충족

## B. EDA Outputs

- [ ] 사이트별 lag/상관/분산분해 리포트 생성
- [ ] DQF 클래스별 민감도 표 생성
- [ ] 유닛 A/B/C 등급표 생성

## C. Track A (Baseline)

- [ ] 홀드아웃 최소 성능 하한 충족
- [ ] 입력 피처/품질 마스크 정책 고정
- [ ] 사이트별 오차 편향 점검

## D. Track B (Bayesian)

- [ ] 캘리브레이션 지표 충족
- [ ] 예측구간 커버리지 충족
- [ ] 저데이터 사이트 안정성 개선 확인

## E. Operations

- [ ] 재수집 후 Stage A~D 재실행
- [ ] 데이터 버전별 비교표 생성
- [ ] 최종 채택 규칙(Track 결합) 판단 로그 기록
