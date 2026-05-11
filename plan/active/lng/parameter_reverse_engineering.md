# LNG Asset Assumptions and Parameter Basis

> 목적: 이 문서는 더 이상 "실제 분당 LNG dispatch를 역산해 unit commitment를 복원하는 문서"가 아니다.  
> 현재 역할은 `model_final.md`에서 채택한 thermal planner가 어떤 LNG 자산 가정을 쓰는지, 그리고 그 수치가 어디서 왔는지를 정리하는 **parameter basis / feasibility memo**다.

---

## 1. 문서의 역할

현재 PoC에서 LNG 문서는 두 가지 역할만 수행한다.

1. planner가 가정하는 controllable LNG block의 물리적/운영상 범위를 설명
2. 그 가정이 완전히 임의가 아니라 4년치 실운전 데이터에서 나온 feasibility reference임을 보여줌

중요한 점은 다음과 같다.

- 이 문서는 실제 발전소 dispatch를 복원하는 목적이 아니다.
- 이 문서는 전국 계통 급전 로직을 설명하는 문서도 아니다.
- 이 문서는 **본 PoC에서 response-state planner가 의존하는 LNG envelope**를 정의하는 문서다.

---

## 2. 현재 PoC에서 LNG를 어떻게 해석하는가

### 2.1 자산의 의미

현재 PoC는 **분당 LNG CS2 등가 controllable block**을 balancing asset으로 가정한다.

이 말의 의미는 다음과 같다.

- 실제 CS2 dispatch가 우리 planner의 정답값이라는 뜻이 아니다.
- 전국 계통 차원의 KPX dispatch와 동일한 의사결정을 재현한다는 뜻도 아니다.
- 남동발전 PV portfolio의 short-horizon variability에 대응할 수 있는 **가용 LNG response block**을 대표 자원으로 추상화한다는 뜻이다.

### 2.2 왜 CS2인가

4년치 hourly 운전 이력 기준으로, CS2는 다음 이유에서 가장 자연스러운 reference asset이었다.

- 낮 시간 availability가 높다.
- partial output 운전이 자주 관찰된다.
- headroom이 크다.
- 빠른 ramp가 가능하다.

즉 CS2는 "새로 켜고 끄는 peaker"라기보다, **이미 운영상 호출 가능한 상태에서 추가 balancing을 받아낼 수 있는 block**으로 해석하기 좋다.

---

## 3. 현재 planner에서 채택한 LNG 가정

### 3.1 기본 가정

현재 `model_final.md` 기준 planner는 아래를 가정한다.

- controllable asset은 **CS2-equivalent LNG block 1개**
- must-run baseload는 planner가 건드리지 않는 외생 자원
- cold-start, startup cost, on/off commitment는 명시적으로 모델링하지 않음
- planner 출력은 full redispatch 명령이 아니라 **incremental balancing recommendation**

즉 이 PoC에서 LNG는 "언제 새로 켜느냐"보다,

- 현재 대응 상태를 유지할지
- 더 올릴지
- 천천히 해제할지

를 판단하는 response asset으로 쓰인다.

### 3.2 최종 숫자 가정

현재 채택 수치는 다음과 같다.

| parameter | final assumption | meaning |
|---|---:|---|
| `THERMAL_MIN` | `43 MW` | 가동 중 쉽게 내리기 어려운 하한 |
| `ONLINE_MAX` | `200 MW` | controllable upper envelope |
| `RAMP_RATE` | `30 MW/h` | 보수적 시간당 ramp 가정 |
| `reserve scale` | `1.282 × σ_p1 × cap_total` | forecast uncertainty 기반 reserve proxy |

이 수치들은 "정답"이 아니라, **planner simulation에서 사용할 operationally plausible envelope**다.

---

## 4. 이 수치는 어디서 왔는가

### 4.1 데이터 출처

- 분당 LNG 4년치 hourly operation data
- 기간: 2022~2025
- 목적: 실제 dispatch 추종이 아니라
  - daytime availability
  - operating band
  - headroom
  - ramp possibility
  를 확인

### 4.2 핵심 관찰

실운전 데이터에서 planner에 의미 있었던 관찰은 다음이다.

- CS2는 낮 시간에 비교적 자주 켜져 있었다.
- partial output 구간이 반복적으로 나타났다.
- 평균 daytime output이 pmax에 딱 붙은 binary peaker라기보다, **중간 출력 상태를 갖는 controllable block**에 더 가까웠다.
- headroom이 커서 현재 PV portfolio swing을 충분히 흡수할 수 있는 범위를 가졌다.

### 4.3 planner용 해석

이 관찰을 planner 관점으로 번역하면 다음과 같다.

- `THERMAL_MIN`은 "가동 중일 때 쉽게 밑으로 못 내리는 수준"
- `ONLINE_MAX`는 "이 block이 보수적으로 감당 가능한 상한"
- `RAMP_RATE`는 "short-horizon response-state change를 걸 때 넘지 않도록 잡은 안전한 속도"

즉, 이 문서의 수치는 plant physics를 완벽히 재현하는 것이 아니라 **planner가 너무 비현실적인 자산을 상정하지 않도록 하는 제약값**이다.

---

## 5. 현재 문서에서 하지 않는 것

다음은 의도적으로 이 문서의 범위 밖이다.

- 실제 CS2 dispatch 재현
- 전국 수요/SMP/예비력 기반 경제급전 모델링
- cold-start 시간과 기동비 최적화
- 복수 LNG 호기 간 dispatch split
- full unit commitment reconstruction

이 항목들을 섞기 시작하면 현재 PoC의 초점이 흐려진다. 현재는 **response-state platform PoC**이므로, 필요한 최소한의 LNG envelope만 유지하는 것이 맞다.

---

## 6. planner와의 직접 연결

현재 LNG parameter 문서는 `model_final.md`의 thermal planner와 다음처럼 연결된다.

### 현재 시점

- `realized_gap_t`로 현재 부족/초과를 본다.
- 이때 LNG block은 `THERMAL_MIN ~ ONLINE_MAX` 범위에서 current correction을 감당할 수 있다고 본다.

### 이후 시간

- `forward_gap`은 remaining-day reforecast curve를 따라 계산된다.
- planner는 이 정보를 보고:
  - response sustain
  - response increase
  - delayed release
  를 정한다.

즉 LNG parameter는 **response-state logic의 물리적 경계**를 제공한다.

---

## 7. 다음 단계

현재 문서 기준 다음 할 일은 새 파라미터를 더 복잡하게 추정하는 것이 아니라, 아래를 문서/시뮬레이션에 일관되게 맞추는 것이다.

1. `model_final.md`의 planner 설명과 숫자 일치
2. `proposal_outline.md`의 PoC framing과 LNG 해석 일치
3. dashboard에서 "실제 dispatch 재현"이 아니라 "response-state recommendation"으로 표현 일치

필요하면 후속으로 추가할 수 있는 것은:

- CS2 daytime availability summary figure
- headroom distribution figure
- hourly ramp distribution figure

하지만 지금은 문서 정합성이 우선이다.

---

## 8. 한 줄 요약

> 현재 LNG 문서의 목적은 실제 분당 LNG 운전을 역산하는 것이 아니라, CS2를 feasibility reference로 삼아 planner가 사용할 controllable LNG block의 operating envelope를 정의하는 것이다.
