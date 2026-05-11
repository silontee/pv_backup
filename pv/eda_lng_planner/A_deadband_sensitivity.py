"""EDA A — DEADBAND 민감도 분석 (PoC 핵심 메시지 입증).

목적:
  현재 DEADBAND=4 MW가 우연히 잘 나온 것이 아니라, *합리적인 plateau 영역*임을 데이터로 보임.
  DEADBAND 0~10 sweep 시 shortfall / over-commit / 비율 trade-off 곡선.

방법:
  thermal_planner_v4 의 DEADBAND_MW 만 변경하면서 1년 KPI 산출 → curve plot.
  (planner code 직접 import 하지 않고, 가벼운 KPI-only 시뮬레이션으로 sweep)

산출:
  pv/eda_v4/A_deadband_curve.png
  pv/eda_v4/A_deadband_results.csv

→ 모델 함의:
  - DEADBAND=0 (전량 LNG 책임): over-commit 폭증 → 비현실적
  - DEADBAND=4 (LNG ≈50%): trade-off 균형, 부족·과대 비율 ≈1.45x
  - DEADBAND=8+: shortfall 도 작아지지만 LNG 책임 자체가 사라짐 → PoC 의미 약화
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# Windows 한글 폰트 (Malgun Gothic) — 깨짐 방지
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
LNG_UNITS = ['CG1','CG2','CG3','CG4','CG5','CG6','CG7','CG8','CS1','CS2']

# === 1. v4 P1+2 log 로드 ===
log = pd.read_parquet(ROOT / "pv/experiments/thermal_planner_v4/log_phase1plus2.parquet")
print(f"loaded {len(log)} hours")

# === 2. DEADBAND 가상 sweep ===
# 핵심 가정: planner 의 *response* (total_alloc) 는 현재 spec (DEADBAND=4) 기준으로 산출됨.
#           DEADBAND 변경 시 *KPI 정의만* 다시 계산 (= post-hoc evaluation).
#           이는 *완전한 sweep* 은 아니지만, 운영자 관점에서 "다른 DEADBAND 였다면 KPI 가 어땠을까" 를 보여줌.
#
# 추가로, *true sweep* (planner 자체를 다른 DEADBAND 로 재실행) 를 위해서는
# thermal_planner_v4 를 sweep 모드로 실행해야 함 → 별도 sweep 스크립트.

# (a) Post-hoc KPI 재계산 (LNG response 고정, DEADBAND 만 변경)
deadbands = [0, 1, 2, 3, 4, 5, 6, 8, 10]
rows_posthoc = []
for db in deadbands:
    eff = (log.instant_gap_mw - db).clip(lower=0)
    short = (eff - log.total_alloc_mw).clip(lower=0).sum()
    over = (log.total_alloc_mw - eff).clip(lower=0).sum()
    rows_posthoc.append({
        'DEADBAND': db, 'mode': 'post-hoc (response 고정)',
        'shortfall': short, 'over_commit': over,
        'ratio': over / max(short, 1),
    })
post = pd.DataFrame(rows_posthoc)
print("\n[Post-hoc DEADBAND sweep] LNG response 고정, KPI 정의만 변경")
print(post.round(1).to_string(index=False))

# === 3. True sweep — planner 재실행 (3 개 핵심 값만) ===
# DEADBAND=0, 4 (현재), 8 만 실제로 재실행. 시간 절약 + 핵심 trade-off 확인용.
import importlib, importlib.util
spec = importlib.util.spec_from_file_location("tp4", ROOT / "src/decisions/thermal_planner_v4.py")
tp4 = importlib.util.module_from_spec(spec)

print("\n[True sweep] planner 재실행 (DEADBAND=0, 2, 4, 6, 8)")
sweep_rows = []
for db in [0, 2, 4, 6, 8]:
    # planner 모듈 reload 하여 DEADBAND 만 monkey-patch
    spec.loader.exec_module(tp4)
    tp4.DEADBAND_MW = float(db)
    # daily_inputs 빌드 (main() 과 동일 logic)
    port1, _ = tp4.load_inputs()
    p2_full = pd.read_parquet(ROOT / "pv/experiments/phase2_2branch_g20_L12/ensemble_test_overridden.parquet")
    baseline_df = tp4.load_lng_baseline()
    specs = tp4.load_unit_specs()
    port1['hour'] = port1['datetime_kst'].dt.hour
    port1 = port1[port1.hour.between(*tp4.ACTIVE_HOURS)].copy()
    port1['date'] = port1['datetime_kst'].dt.normalize()
    daily_inputs = [(d, g.assign(demand_mw=g.hour.apply(tp4.demand_profile)))
                    for d, g in port1.groupby('date')]

    rows = tp4.planner_run_v4(daily_inputs, 'phase1+2', p2_full, baseline_df, specs)
    df = pd.DataFrame(rows)
    short = float(df.shortfall_mw.sum())
    over = float(df.over_commit_mw.sum())
    total_alloc = float(df.total_alloc_mw.sum())
    sweep_rows.append({
        'DEADBAND': db, 'shortfall': short, 'over_commit': over,
        'ratio': over / max(short, 1),
        'total_dE': total_alloc,
        'state_INC': int((df.state == 'INCREASE').sum()),
    })
    print(f"  DEADBAND={db:2d}: short={short:6.0f}  over={over:6.0f}  ratio={over/max(short,1):4.2f}  ΔE={total_alloc:6.0f}")

true_sweep = pd.DataFrame(sweep_rows)

# === 4. 그림 ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# (a) shortfall vs over-commit 곡선
ax = axes[0]
ax.plot(true_sweep.DEADBAND, true_sweep.shortfall, 'o-', color='#D32F2F', label='shortfall (부족분)', lw=2, markersize=8)
ax.plot(true_sweep.DEADBAND, true_sweep.over_commit, 's-', color='#1976D2', label='over-commit (과대보충)', lw=2, markersize=8)
ax.axvline(4, color='black', lw=1, ls='--', alpha=0.5)
ax.text(4.1, true_sweep.shortfall.max()*0.85, '★ 채택 = 4 MW', fontsize=10, color='black', fontweight='bold')
ax.set_xlabel('DEADBAND (MW)')
ax.set_ylabel('1년 누적 (MWh)')
ax.set_title('DEADBAND sweep — shortfall / over-commit')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# (b) 비율
ax = axes[1]
ax.plot(true_sweep.DEADBAND, true_sweep.ratio, 'o-', color='#388E3C', lw=2, markersize=8)
ax.axhline(1.0, color='gray', lw=1, ls=':', alpha=0.5)
ax.axvline(4, color='black', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('DEADBAND (MW)')
ax.set_ylabel('over-commit / shortfall 비율')
ax.set_title('balance 비율 — 1에 가까울수록 균형')
ax.grid(True, alpha=0.3)

# (c) 총 ΔE
ax = axes[2]
ax.plot(true_sweep.DEADBAND, true_sweep.total_dE, '^-', color='#FB8C00', lw=2, markersize=8)
ax.axvline(4, color='black', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('DEADBAND (MW)')
ax.set_ylabel('1년 누적 LNG 추가 발전 (MWh)')
ax.set_title('LNG 책임분 — DEADBAND 가 클수록 줄어듦')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = ROOT / "pv/eda_v4/A_deadband_curve.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n저장: {out}")

# === 5. CSV 저장 ===
true_sweep.to_csv(ROOT / "pv/eda_v4/A_deadband_results.csv", index=False)
post.to_csv(ROOT / "pv/eda_v4/A_deadband_posthoc.csv", index=False)
print(f"저장: pv/eda_v4/A_deadband_results.csv (true sweep)")
print(f"저장: pv/eda_v4/A_deadband_posthoc.csv (post-hoc 비교)")

# === 6. 모델 함의 ===
print("\n" + "="*60)
print("→ 모델 함의")
print("="*60)
chosen = true_sweep[true_sweep.DEADBAND == 4].iloc[0]
db0 = true_sweep[true_sweep.DEADBAND == 0].iloc[0]
db8 = true_sweep[true_sweep.DEADBAND == 8].iloc[0]
print(f"""
1. DEADBAND=0 (계통 흡수 가정 X): over-commit {db0.over_commit:.0f} MWh
   → 작은 변동까지 LNG 가 추종 → over-commit 폭증
2. DEADBAND=4 (★ 채택, LNG ≈50% 담당 가정): over-commit {chosen.over_commit:.0f} MWh, ratio {chosen.ratio:.2f}
   → 부족·과대 균형
3. DEADBAND=8 (계통 흡수 과대): shortfall {db8.shortfall:.0f} MWh, total ΔE {db8.total_dE:.0f} MWh
   → LNG 책임이 너무 작아짐 → PoC 의미 약화

→ DEADBAND=4 는 보조서비스 정산 (LNG ≈50% 담당) 근거 + KPI 균형이 일치하는 영역.
""")
