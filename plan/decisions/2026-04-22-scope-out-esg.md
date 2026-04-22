# ADR: ESG 대기질 모듈 scope-out

## Context

초기 PRD 시절 ESG 대기질 모듈을 위한 `plan/esg/` 디렉토리가 존재했다.
이후 계획은 LNG 백업, PV 변동성, 제한된 공개 데이터 기반 PoC로 축소되었다.
현재 `plan/esg/`는 비어 있으며 살아있는 계획과 연결되지 않는다.

## Decision

ESG 대기질 모듈은 현재 프로젝트 범위에서 제외한다.
빈 `plan/esg/` 디렉토리는 제거한다.
프로젝트 초점은 PV 예측 불확실성과 LNG/화력 백업 의사결정 지원으로 둔다.

## Consequences

문서 트리에서 사용하지 않는 영역을 제거해 탐색 비용을 줄인다.
향후 ESG 모듈을 재개하려면 새 scope decision과 plan을 작성해야 한다.
현재 PoC는 데이터 확보 가능성과 운영 영향 검증에 집중한다.
