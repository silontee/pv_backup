# ADR: SSoT와 versionless current 문서

## Context

`plan/main/`에 `plan_v1`부터 `plan_v8`까지 여러 버전이 병존했다.
현재 기준 문서가 무엇인지 파일명만으로 판단하기 어려웠고, 변경 이유도 추적하기 어려웠다.
사용자는 plan이 흩어져 있고 실시간 중요 변경을 기록할 곳이 없다는 pain point를 제기했다.

## Decision

살아있는 계획은 `plan/current/` 아래에 버전 번호 없이 둔다.
과거 버전은 `plan/history/`로 이동해 참고용으로만 보존한다.
현재 SSoT는 `problem.md`, `plan.md`, `data_strategy.md`, `acceptance.md` 조합으로 관리한다.
주요 변경은 `CHANGELOG.md`에 날짜순으로 요약한다.

## Consequences

현재 문서를 찾는 비용이 줄어든다.
과거 문서는 남아 있으나 기본 작업 경로에서는 제외된다.
새 버전을 만들기보다 `current/`를 직접 갱신해야 하므로 변경 요약 기록이 중요해진다.
