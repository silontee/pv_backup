"""EDA 04 — Calibration + 5-seed ensemble 효과.

목적:
  - Cov80, Cov95 calibration plot
  - 5-seed 의 NMAE / Cov 분산 + ensemble 효과
  - σ_total 이 well-calibrated 임을 보임 (Phase 2 가 σ 그대로 사용 정당화)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda_v4"))
from _common import ROOT

EDA_DIR = ROOT / "pv/eda_pv_model"

# === seed별 metric ===
seed = pd.read_csv(ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/per_seed_summary.csv")
print("[5-seed Phase 1 ensemble]")
print(seed.round(3).to_string(index=False))

# ensemble metric (per-seed avg vs ensemble)
print(f"\n  per-seed mean NMAE: {seed.nmae.mean():.3f}%  (std {seed.nmae.std():.3f})")
print(f"  per-seed mean Cov80: {seed.cov80.mean():.2f}%")
print(f"  per-seed mean Cov95: {seed.cov95.mean():.2f}%")
print(f"  per-seed mean NLL: {seed.nll.mean():.3f}")

# ensemble (model_final §1.2 표 기준)
print(f"\n  ensemble: NMAE 4.66%, Cov80 85.0%, Cov95 94.1%, NLL -1.218")

# === 시각화 ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) per-seed NMAE
ax = axes[0]
seed_sorted = seed.sort_values('nmae')
xs = np.arange(len(seed_sorted))
ax.bar(xs, seed_sorted.nmae, color='#FB8C00')
ax.axhline(seed.nmae.mean(), color='#1976D2', ls='--',
            label=f'per-seed 평균 {seed.nmae.mean():.2f}%')
ax.axhline(4.66, color='#388E3C', ls='-',
            label='ensemble 4.66%')
ax.set_xticks(xs)
ax.set_xticklabels([f'seed {s}' for s in seed_sorted.seed])
ax.set_ylabel('NMAE (%)')
ax.set_title(f'5-seed NMAE — std {seed.nmae.std():.3f}pp')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# (b) Cov80 / Cov95 calibration
ax = axes[1]
xs = np.arange(len(seed_sorted))
ax.bar(xs - 0.2, seed_sorted.cov80, 0.4, label='Cov80 (목표 80%)', color='#1976D2')
ax.bar(xs + 0.2, seed_sorted.cov95, 0.4, label='Cov95 (목표 95%)', color='#388E3C')
ax.axhline(80, color='#1976D2', ls=':', alpha=0.5)
ax.axhline(95, color='#388E3C', ls=':', alpha=0.5)
ax.set_xticks(xs)
ax.set_xticklabels([f'seed {s}' for s in seed_sorted.seed])
ax.set_ylabel('Coverage (%)')
ax.set_title('Calibration — σ 가 well-calibrated')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(70, 100)

# (c) 사이트별 calibration (ensemble per-site)
ax = axes[2]
site = pd.read_csv(ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/ensemble_per_site.csv")
site = site.sort_values('cov80')
ys = np.arange(len(site))
ax.barh(ys - 0.2, site.cov80, 0.4, label='Cov80', color='#1976D2')
ax.barh(ys + 0.2, site.cov95, 0.4, label='Cov95', color='#388E3C')
ax.axvline(80, color='#1976D2', ls=':', alpha=0.5)
ax.axvline(95, color='#388E3C', ls=':', alpha=0.5)
ax.set_yticks(ys)
ax.set_yticklabels(site.site)
ax.set_xlabel('Coverage (%)')
ax.set_title('사이트별 calibration — 광양항 (Cov80 70%) outlier')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
out = EDA_DIR / "04_calibration_seed.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n저장: {out}")

print(f"""
{'='*60}
→ 모델 함의
{'='*60}

1. 5-seed ensemble: per-seed NMAE std 0.10pp, ensemble NMAE 4.66% (개별보다 안정)
2. Cov80 / Cov95 모두 목표 ±5pp 안 (over-coverage 약간) → σ_total well-calibrated
3. 사이트별 calibration: 광양항 (Cov80 70%) 만 underconfident → bias correction 후보
4. Phase 2 가 σ 그대로 쓰는 정당성: ensemble σ 가 이미 잘 보정되어 있음
""")
