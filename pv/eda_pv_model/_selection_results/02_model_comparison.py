"""EDA 02 — Backbone 모델 비교 (왜 ResMLP+AdaLN 인가).

목적:
  60+ 실험 폴더 중 *동일 평가 protocol (test 2025)* 로 비교 가능한 모델들의 NMAE 정리.

→ 모델 함의:
  - LSTM / NGBoost / FT-Transformer / ResMLP+AdaLN 비교
  - ResMLP+AdaLN v2 ensemble 가 종합적으로 최우수
  - tabular 입력 + AdaLN conditioning + ensemble = 본 PoC 에 가장 적합
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda_v4"))
from _common import ROOT

EDA_DIR = ROOT / "pv/eda_pv_model"

# === 모델별 test_predictions.parquet 로드 + NMAE 산출 ===
def calc_nmae(parquet_path):
    if not parquet_path.exists(): return None
    p = pd.read_parquet(parquet_path)
    cols = p.columns.tolist()
    # 컬럼 이름 일관성 처리
    for c in ('mu_mean', 'pred_cf', 'mu_phase1', 'mu'):
        if c in cols:
            mu = p[c]; break
    else:
        return None
    if 'cf' not in cols: return None
    if 'site_capacity_kw' in cols:
        cap = p['site_capacity_kw']
    else:
        return None
    err_pct = (mu - p['cf']).abs() / 1.0 * 100   # cf 단위가 0~1 이므로 그대로 NMAE
    nmae = err_pct.mean()
    bias = (mu - p['cf']).mean() * 100
    return float(nmae), float(bias), len(p)

models = [
    ('LSTM baseline', 'lstm_baseline/test_predictions.parquet'),
    ('NGBoost baseline', 'ngboost_baseline/test_predictions.parquet'),
    ('FT-Transformer', 'ft_transformer/test_predictions.parquet'),
    ('ResMLP+AdaLN (single)', 'resmlp_adaln/test_predictions.parquet'),
    ('ResMLP+AdaLN v2 (single)', 'resmlp_adaln_v2_clean/test_predictions.parquet'),
    ('ResMLP+AdaLN v2 ensemble (★ 채택)', 'resmlp_adaln_v2_ensemble/ensemble_test.parquet'),
]
rows = []
for name, path in models:
    full = ROOT / 'pv/experiments' / path
    res = calc_nmae(full)
    if res is None:
        print(f"  skip: {name}  ({path})")
        continue
    nmae, bias, n = res
    rows.append({'model': name, 'NMAE': nmae, 'bias_%': bias, 'n': n})
df = pd.DataFrame(rows).sort_values('NMAE')
print("\n[Backbone 모델 비교 — site-level NMAE, test 2025]")
print(df.round(3).to_string(index=False))

# 5-seed ensemble 별도: per_seed_summary.csv
seed = pd.read_csv(ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/per_seed_summary.csv")
print(f"\n[ResMLP+AdaLN v2 ensemble 5-seed]")
print(seed[['seed','nmae','bias','cov80','cov95','nll']].round(3).to_string(index=False))
print(f"  ensemble NMAE: {seed.nmae.mean():.3f}%  (per-seed std {seed.nmae.std():.3f})")

# === 시각화 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 모델 비교 막대
ax = axes[0]
colors = ['#388E3C' if '★' in m else '#1976D2' for m in df.model]
ax.barh(df.model, df.NMAE, color=colors)
ax.invert_yaxis()
ax.set_xlabel('NMAE (%)')
ax.set_title('Backbone 모델 비교 — test 2025')
ax.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(df.NMAE):
    ax.text(v+0.05, i, f'{v:.2f}', va='center', fontsize=9)

# (b) 5-seed variance
ax = axes[1]
ax.bar(range(1, 6), seed.sort_values('nmae').nmae, color='#FB8C00')
ax.axhline(seed.nmae.mean(), color='#D32F2F', ls='--', label=f'평균 {seed.nmae.mean():.2f}')
ax.set_xticks(range(1, 6))
ax.set_xticklabels([f'seed {s}' for s in seed.sort_values('nmae').seed])
ax.set_ylabel('NMAE (%)')
ax.set_title(f'5-seed ensemble 변동 — std {seed.nmae.std():.3f}pp')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = EDA_DIR / "02_model_comparison.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n저장: {out}")

df.to_csv(EDA_DIR / "02_model_comparison.csv", index=False)
print(f"저장: {EDA_DIR / '02_model_comparison.csv'}")
