# Plan Directory Guide

이 디렉토리는 프로젝트 계획, 의사결정, 참고자료, 원문 기록을 분리해 관리합니다.

## 먼저 읽을 문서

1. `current/problem.md`: 현재 문제 정의와 범위
2. `current/plan.md`: 현재 실행 계획 SSoT
3. `current/data_strategy.md`: 데이터 전략과 사용 가능성 판단
4. `current/acceptance.md`: 수용 기준과 체크리스트
5. `CHANGELOG.md`: 날짜순 주요 변경 요약

## 어디에 무엇을 적나

- `current/`: 지금 살아있는 계획만 둡니다. 버전 번호를 붙이지 않습니다.
- `pv/eda_findings.md`: 노트북 EDA 결과가 모델/계획에 주는 함의를 적습니다.
- `pv/model_notes.md`: PV 모델 설계 요약과 paper source 연결 메모를 적습니다.
- `fuel/`: LNG/화력 파라미터 역산 등 fuel 관련 계획을 둡니다.
- `decisions/`: 되돌리기 어렵거나 이후 작업에 큰 영향을 주는 결정만 ADR로 남깁니다.
- `references/`: 사람이 참고하는 문서, PDF, CSV, 링크 메모를 둡니다.
- `raw/`: KOEN 회신 등 가공 전 원문을 보존합니다.
- `history/`: 과거 plan 버전과 폐기된 초안을 보존합니다.

## 포맷 원칙

- Plan 문서는 Markdown을 사용합니다.
- 논문/최종 PDF 원본은 LaTeX를 유지합니다.
- Raw 문서는 원문 보존을 우선합니다.

## 운영 규칙

- 현재 계획을 바꿀 때는 `current/`를 수정하고 `CHANGELOG.md`에 날짜순으로 요약합니다.
- 큰 방향 전환은 `decisions/YYYY-MM-DD-topic.md`에 Context/Decision/Consequences로 기록합니다.
- 노트북에서 발견한 내용은 바로 plan 본문에 섞지 말고 `pv/eda_findings.md`에 먼저 정리합니다.
- 과거 버전 파일을 `current/`에 다시 만들지 않습니다.
