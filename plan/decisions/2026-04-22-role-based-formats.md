# ADR: 역할 기반 문서 포맷

## Context

프로젝트 문서에 Markdown과 LaTeX가 혼재했다.
특히 `pv_predict.md`는 확장자는 Markdown이지만 실제 내용은 LaTeX였다.
수식, 표, bibliography, 최종 PDF 가능성을 고려하면 모든 문서를 Markdown으로 변환하는 것은 위험하다.

## Decision

문서 포맷은 역할 기준으로 정한다.
계획 문서는 Markdown을 사용한다.
논문 또는 최종 PDF 원본은 LaTeX를 유지한다.
KOEN 회신 등 raw 자료는 원문 보존을 우선한다.

## Consequences

확장자와 실제 내용의 불일치를 줄일 수 있다.
LaTeX 문서의 수식과 제출 가능성을 보존한다.
계획 문서는 Markdown으로 통일되어 리뷰와 수정이 쉬워진다.
