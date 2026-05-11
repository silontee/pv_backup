# PV Mapping Problem — Official Framing

**확정일**: 2026-05-05
**Status**: 공식 문제 정의 (이후 모든 PV 실험에 적용)

---

## 1. 문제 정의

PV 발전량 예측 모델은 *기상 input → cf (capacity factor) 분포 (μ, σ)* 매핑 함수.

**입력**:
- 기상 관측값 (실측): `dsr_mean`, `ta`, `hm`, `ws`, `dc10Tca`
- 결정적 시간 feature: `zenith_center`, `hour_*`, `month_*`
- 사이트 정보: `site_id`, `group_id`, `site_capacity_kw`

**출력**:
- `cf` Gaussian 분포 (μ, σ) per (site, hour)
- Portfolio 합산 (독립 가정): μ_total = Σ μ_i, σ_total² = Σ σ_i²

---

## 2. ⚠️ Perfect-Foresight Upper Bound *명시*

본 PoC의 PV 모델 *모든 성능 수치*는 **perfect-foresight upper bound**임:

```
입력 weather = 실제 D 시점 ASOS + GK-2A 관측값 (실측)
            ≠ D-1 17:00 KPX 신고 시점에 실제로 가용한 forecast
```

### 함의
- 현재 측정된 NMAE (예: ResMLP+AdaLN 5.12%, Portfolio 4.13%)는 **PV mapping 자체의 *최고 가능 정확도***
- 실제 운영 D-1 forecast 환경에선 *NWP forecast 오차*가 추가로 더해져 *반드시 더 나쁨*
- 즉 본 수치는 **"PV mapping 함수의 *모델 한계*"** 측정 (lower bound on prediction error attributable to mapping itself)

### 왜 이 framing을 채택했나
1. **Strict D-1 forecast archive 미보유** — KMA API Hub forecast endpoint는 별도 신청 1-2일 대기, TIGGE는 해상도 50km로 한국 PoC에 부적합
2. **모델 비교 자체는 *정확하게 수행 가능*** — 모든 실험이 같은 입력 사용하면 *모델 간 상대 비교*는 의미 있음
3. **Future work**: forecast archive 확보 시 strict D-1 NMAE *재측정* (별도 evaluation track)

### Future work — strict D-1 evaluation
- **데이터**: KMA LDAPS archive (1.5km, hourly) 신청
- **MOS layer**: raw forecast → site/hour/month/lead 조건부 bias 보정
- **Pipeline**: raw forecast → MOS → PV model
- **결과**: strict D-1 NMAE 측정 → 본 upper-bound 대비 gap 정량화

---

## 3. 평가 metric

모든 PV 모델 동일 metric:
- **Site NMAE** (capacity-weighted): `Σ|err| × cap / Σ cap × 100%`
- **Portfolio NMAE**: 시간별 합산 후 NMAE
- **Cov80**: `[μ - 1.282σ, μ + 1.282σ]` 범위 안 실측 비율 (목표 80%)
- **Cov95**: 동일, 1.96σ
- **NLL** (선택): Gaussian likelihood
- **CRPS** (선택): probabilistic score

---

## 4. Train / Val / Test split

- **Train**: 2022-01 ~ 2023-12 (2년)
- **Val**: 2024-01 ~ 2024-12 (1년)
- **Test**: 2025-01 ~ 2025-12 (1년)
- 모든 split은 *target 시각 기준* (D)
- Test는 *out-of-sample* (학습 시 미사용)

---

## 5. 현재 best model (perfect-foresight upper bound)

| Model | Site NMAE | Portfolio NMAE | Cov80 |
|---|---|---|---|
| NGBoost baseline | 6.20% | 5.18% | 80.6% |
| Strong NGBoost (cross feat) | 6.07% | 5.04% | 81.7% |
| MLP+FiLM | 5.88% | 4.88% | 84.3% |
| FT-Transformer | 5.49% | 4.48% | 87.7% |
| **ResMLP+AdaLN** ⭐ | **5.12%** | **4.13%** | 82.6% |
| LSTM v2 C (window=12) | 6.25% | 5.16% | 81.7% |

→ **현 winner: ResMLP + AdaLN** (단일 모델). 이 위에 Bayesian reweight (Track B 개선) 가능.

---

## 6. PV → Thermal Planner 연결 (gap-driven dispatch)

### 정의
```
gap = actual - μ_pred  (signed)
  → gap < 0: PV under-delivers → 추가 LNG ramp 필요 (required_backup_mw = -gap × capacity)
  → gap > 0: PV over-delivers → LNG ramp down (excess_pv_mw = gap × capacity)
```

### Pipeline (운영 시점 단순 흐름)
```
D-1 17:00:
  PV model.predict() → (μ_pred, σ_pred) for D 24h
  → KPX 신고 (μ_pred 기준 capacity factor 신고)
  → 시나리오 생성 (Bayesian reweight prior)

D 매 시간 t:
  실측 cf_actual_t 들어옴
  gap_t = cf_actual_t - μ_pred_t
  required_backup_mw_t = max(0, -gap_t) × site_capacity
  → Thermal planner.plan(required_backup_mw_t, horizon)
  → Dispatch action (online ramp / peaker / cold-start)
```

---

## 7. 적용 범위

이 framing은 *이후 모든 PV 모델 실험에 자동 적용*:
- 신규 모델 학습 시 *perfect-foresight* setting 그대로 사용 (변경 없음)
- 결과 보고 시 *"perfect-foresight upper-bound"* 명시
- Pipeline 다이어그램에 *"D weather: actual observation (upper bound)"* 라벨

---

## 8. 변경 이력

- 2026-05-05: 본 문서 작성 (strict D-1 archive 부재로 upper-bound로 confirmed)
- (이후 strict D-1 transition 시 추가 기록)
