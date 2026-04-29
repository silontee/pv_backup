# Plan Changelog

## 2026-04-28 - PV 모델 옵션 C(MOS+PV 2-Stage) 채택 검토 시작

- `plan/pv/plan_v1.md` 신설 — Stage 0 MOS + Stage 1 PV(v8 Hybrid 2-Track) 2-Stage 구조 초안.
- v8 Hybrid 2-Track(LightGBM + Bayesian)은 Stage 1으로 흡수, 자산 보존.
- 결정 근거: `plan/decisions/2026-04-28-pv-mos-stage.md` (옵션 A/B/C 비교, GK-2A NaN 분석, 영흥 2024-01-15 검증 결과).
- `current/acceptance.md` 폐기 — v8 Hybrid 2-Track 종속, MOS 추가 후 재작성 예정.
- `current/plan.md`, `current/data_strategy.md` 상단에 PIVOT 배너. 본문 갱신은 EDA 3개 Go/No-Go 게이트 통과 후 일괄.
- 게이트: (1) Open-Meteo historical-forecast-api dawn/dusk 커버리지, (2) MOS 편향의 학습 가능성, (3) MOS-PV 파이프라인 성능. 세 게이트 모두 통과 시 plan_v1 → current(v9) 승격.

## 2026-04-22 - plan 디렉토리 재구조화

- `current/`와 `history/`를 분리해 살아있는 SSoT와 과거 버전을 구분했다.
- 원칙을 plan=Markdown, paper=LaTeX, raw=원문 보존으로 정리했다.
- EDA 결과를 계획으로 환류하기 위한 `plan/pv/eda_findings.md`를 신설했다.
- KOEN 회신과 초기 질문 브레인스토밍은 `raw/`에 보존하고, 참고자료는 `references/`로 분리했다.

## 2026-04-21 - KOEN 회신 수신

- KOEN 회신을 통해 SPC 상세자료와 화력 발전기 세부 파라미터 제공이 어렵다는 점을 확인했다.
- 프로젝트 성격을 운영 시스템 완성보다 제한된 데이터 기반 PoC와 의사결정 지원 가능성 검증으로 정렬했다.
- 공개 데이터, 역산 가능한 파라미터, 시나리오 기반 검증을 중심으로 계획을 조정해야 한다.
