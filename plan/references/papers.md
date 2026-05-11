# 참고 논문 — Probabilistic Forecasting × LNG 백업 의사결정

> 우리 PoC의 *연결성* 강화 학술적 근거. Tier별로 우선순위.
> 마지막 갱신: 2026-05-04

## 우리 의문 ↔ 논문 매핑

| 의문 | 답 있는 논문 |
|---|---|
| "scenario ↔ decision 연결성 약한 이유" | **Donti 2017** ★ |
| "real-time PV 정보 가치 한계" | Pinson 2012, 2013 |
| "1000 시나리오 어떻게 다루나" | Heitsch 2003, Bouffard 2008 |
| "분산 사이트 합산·correlation" | Bessa 2017 |
| "trajectory-based vs marginal quantile" | Bouffard 2008 §3 |
| "scenario reduction 표준 알고리즘" | Heitsch 2003 |

---

## Tier 1 — 필수 (3편)

### 1. Pinson (2013) — *큰 그림*

**제목**: "Wind energy: Forecasting challenges for its operational management"
**저널**: *Statistical Science*, 28(4), 564-585
**오픈 액세스**: https://projecteuclid.org/journals/statistical-science/volume-28/issue-4

**왜 첫 번째**: probabilistic forecast → 의사결정 frame의 학술적 표준 overview. *연결성* 문제의 본질 정리.

**포커스 섹션**:
- §3 — Uncertainty propagation (예측 분포 → 운영)
- §4 — Decision making under uncertainty

**우리 PoC 적용**:
- "분포 정확도와 의사결정 가치 분리" 정당화
- 예: 우리 NGBoost 80% coverage = 분포 정확도, LNG 비용 절감 = 의사결정 가치

---

### 2. Bessa et al. (2017) — *Solar + 분산 사이트*

**제목**: "Probabilistic solar power forecasting in smart grids using distributed information"
**저널**: *International Journal of Electrical Power & Energy Systems*, 88, 86-97
**DOI**: 10.1016/j.ijepes.2016.12.018

**왜 두 번째**: 우리 케이스와 *가장 비슷*. 분산 PV 사이트 + probabilistic + 운영 통합.

**포커스 섹션**:
- §2 — Probabilistic forecasting framework
- §4 — Case study (우리 실험과 비교)

**우리 PoC 적용**:
- 8 사이트 합산 시 cor 처리 (우리 잔차 cor 0.03 → 독립 합산과 비교)
- Quantile forecast → operational use

---

### 3. Donti, Amos, Kolter (2017) — *연결성 정답* ★

**제목**: "Task-based End-to-End Model Learning in Stochastic Optimization"
**학회**: *NeurIPS 2017*
**arXiv**: https://arxiv.org/abs/1703.04529 (오픈)
**코드**: https://github.com/locuslab/e2e-model-learning

**왜 세 번째 (실은 가장 중요)**: 예측 정확도 ≠ 의사결정 비용 — 정확히 우리 *연결성* 문제 정형화. Decision-focused learning의 시초.

**포커스 섹션**:
- §2 — Problem formulation (E2E model)
- §4 — Electrical grid scheduling case (우리와 정확히 같은 problem)

**우리 PoC 적용**:
- 현재 NGBoost는 *분포 정확도*에 학습됨
- Donti 방식: forecast 모델을 *LNG 비용 regret 최소화*로 직접 학습
- 향후 enhancement (Phase 2)의 핵심 기법

---

## Tier 2 — 깊이 (3편, 선택)

### 4. Bouffard & Galiana (2008) — *SUC 표준*

**제목**: "Stochastic Security for Operations Planning With Significant Wind Power Generation"
**저널**: *IEEE Transactions on Power Systems*, 23(2), 306-316

**왜**: Stochastic Unit Commitment (SUC) 고전적 reference. 시나리오 기반 발전 운영의 학술적 baseline.

**우리 PoC 적용**:
- LNG 시나리오 결정 로직의 학술적 grounding
- 우리 단순 결정 ↔ SUC 정교화 비교

**단점**: 비교적 무거운 MIP 수식

---

### 5. Pinson (2012) — *Bayesian re-weighting 정확히*

**제목**: "Adaptive calibration of (u, v)-wind ensemble forecasts"
**저널**: *Quarterly Journal of the Royal Meteorological Society*, 138(666), 1273-1284

**왜**: 우리 Bayesian re-weighting과 *정확히 같은 메커니즘*. Wind이지만 idea 동일.

**우리 PoC 적용**:
- 1h ahead σ -15.8% 효과의 학술적 근거
- ACF lag-1 0.63 → re-weighting 가치 정량화 표준

---

### 6. Heitsch & Römisch (2003) — *시나리오 축소*

**제목**: "Scenario reduction algorithms in stochastic programming"
**저널**: *Computational Optimization and Applications*, 24(2), 187-206

**왜**: 1000 → 100 시나리오 압축 표준 알고리즘. LNG planner 정교화 시 필요.

**우리 PoC 적용**:
- 현재 1000 trajectory → trajectory-aware planner에서 너무 많음
- Forward selection / backward reduction 적용

---

## Tier 3 — 옵션 (4편)

### 7. Wang, Shahidehpour, Li (2008)
**"Security-Constrained Unit Commitment with Volatile Wind Power Generation"**
*IEEE TPS*, 23(3), 1319-1327

→ SUC 응용 사례. 우리 LNG planner 정교화 reference.

### 8. Doucet & Johansen (2009)
**"A tutorial on particle filtering and smoothing"**
*Oxford Handbook of Nonlinear Filtering*

→ Bayesian filter 깊은 수학. 70페이지 — §1-3만 봐도 충분.

### 9. Wilder, Dilkina, Tambe (2019)
**"Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization"**
*AAAI 2019*

→ Donti 2017의 후속. Decision-focused learning 발전.

### 10. Mavromatidis et al. (2018)
**"A review of uncertainty characterisation approaches for the optimal design of distributed energy systems"**
*Renewable & Sustainable Energy Reviews*

→ 시나리오 기반 vs robust vs stochastic 비교 review.

---

## 추천 읽기 순서 + 시간

### 첫째날 (3~4시간) — 핵심 idea 잡기
1. **Pinson 2013** (1시간) — 큰 그림
2. **Bessa 2017** (45분) — Solar 사례
3. **Donti 2017 §1-3** (1시간) — decision-focused learning

### 둘째날 (2~3시간, 깊이)
4. **Donti 2017 §4** + 코드 — 실험
5. **Bouffard 2008** (1시간) — SUC 수학
6. **Pinson 2012** (45분) — re-weighting 기반

### 셋째날 (옵션)
7. Heitsch 2003 — 시나리오 reduction
8. Wang 2008 / Doucet 2009 — 깊이

---

## 접근 방법

| 출처 | 액세스 |
|---|---|
| arXiv | 무료 (Donti 2017) |
| projecteuclid.org | 오픈 (Pinson 2013) |
| IEEE Xplore | 학교 계정 필요 (Bouffard, Wang) |
| ScienceDirect | 학교 계정 (Bessa, Mavromatidis) |
| ResearchGate | 저자 업로드 종종 있음 |
| Google Scholar | 검색 → 무료 PDF 링크 종종 있음 |

---

## PoC 발표 시 활용

### 학술적 grounding 필요한 곳

1. **시나리오 frame 정당화** → cite Bouffard 2008, Pinson 2013
2. **Re-weighting 메커니즘** → cite Pinson 2012
3. **분산 사이트 합산** → cite Bessa 2017
4. **연결성 한계 인식** → cite Donti 2017 (future work)

### Future work 학술적 명시

```
"본 PoC는 forecast accuracy 기반 시나리오 생성 후
 휴리스틱 LNG 결정. 향후:
 - Decision-focused learning (Donti et al., 2017)으로
   forecast 모델을 LNG 비용 regret 직접 최소화로 학습
 - Stochastic Unit Commitment (Bouffard & Galiana, 2008)
   적용한 정교 LNG 결정 최적화"
```

이러면 *학술적으로 정직하고 강한* PoC.

---

## 우리 case에 *반드시* 봐야 할 것

너 *지금* 의문 ("연결성 부족")의 답은:

**Donti 2017 §1 introduction + §2 formulation** (한 30분)

핵심 idea 한 문장:
```
"Forecast 모델 학습 목적함수 = forecast loss"
   → "Forecast 모델 학습 목적함수 = downstream decision regret"
```

이거 한 줄이 우리 *연결성*의 답이고, PoC enhancement 방향.
