"""EDA 03 — Phase 2 architecture sweep 정리.

목적:
  H (look-back), λ (event branch 가중치), L (output horizon), gate 설정 sweep
  결과를 한 그림으로.

→ 모델 함의:
  - H=6, 2-branch λ=2.0, L=12 EOD, T=1 τ=0 가 *cloud-pass robustness* 우선의 결과
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda_v4"))
from _common import ROOT

EDA_DIR = ROOT / "pv/eda_pv_model"

# === 데이터 (model_final §1.5, §1.6, §1.7 표 그대로 + 산출 가능 시 직접 계산) ===
H_sweep = pd.DataFrame([
    {'H':3, 'overall':4.717, 'lead1':4.239, 'partial':5.936, 'd0323':45.31, 'd0426':34.75, 'd0504':42.42, 'std':0.020},
    {'H':4, 'overall':4.715, 'lead1':4.237, 'partial':5.913, 'd0323':44.39, 'd0426':34.51, 'd0504':41.63, 'std':0.004},
    {'H':5, 'overall':4.715, 'lead1':4.242, 'partial':5.906, 'd0323':42.93, 'd0426':34.41, 'd0504':41.32, 'std':0.013},
    {'H':6, 'overall':4.729, 'lead1':4.256, 'partial':5.935, 'd0323':41.98, 'd0426':34.31, 'd0504':42.10, 'std':0.049},
])
print("[H sweep — H=6 채택]")
print(H_sweep.round(3).to_string(index=False))

lambda_sweep = pd.DataFrame([
    {'config':'H=6 single-branch', 'NMAE':4.729, 'lead1':4.256, 'partial':5.935, 'std':0.049, 'd0323':41.98},
    {'config':'2-branch λ=1.0', 'NMAE':4.738, 'lead1':4.271, 'partial':5.899, 'std':0.023, 'd0323':42.50},
    {'config':'2-branch λ=1.5', 'NMAE':4.742, 'lead1':4.261, 'partial':5.882, 'std':0.014, 'd0323':42.30},
    {'config':'2-branch λ=2.0 ★', 'NMAE':4.712, 'lead1':4.209, 'partial':5.871, 'std':0.026, 'd0323':42.12},
])
print("\n[λ sweep — λ=2.0 채택]")
print(lambda_sweep.round(3).to_string(index=False))

gate_sweep = pd.DataFrame([
    {'config':'T=1.0 τ=0 ★', 'gate_ratio':1.9, 'dead_seeds':0, 'NMAE':4.712, 'd0323':42.12},
    {'config':'T=0.5 τ=0', 'gate_ratio':5.8, 'dead_seeds':0, 'NMAE':4.743, 'd0323':43.77},
    {'config':'T=0.5 τ=0.1', 'gate_ratio':16.8, 'dead_seeds':2, 'NMAE':4.732, 'd0323':42.56},
    {'config':'T=0.5 τ=0.2', 'gate_ratio':0.0, 'dead_seeds':3, 'NMAE':4.729, 'd0323':41.98},
])
print("\n[Sparse gate sweep — 폐기 (negative result)]")
print(gate_sweep.round(2).to_string(index=False))

# === 시각화 ===
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# (a) H sweep — overall vs top-event
ax = axes[0, 0]
ax.plot(H_sweep.H, H_sweep.overall, 'o-', color='#1976D2', lw=2, label='Overall NMAE')
ax2 = ax.twinx()
ax2.plot(H_sweep.H, H_sweep.d0323, 's-', color='#D32F2F', lw=2, label='03-23 cloud-pass NMAE')
ax.set_xlabel('H (look-back hours)')
ax.set_ylabel('Overall NMAE (%)', color='#1976D2')
ax2.set_ylabel('03-23 NMAE (%)', color='#D32F2F')
ax.axvline(6, color='black', ls='--', alpha=0.5)
ax.set_title('H sweep — H=6 채택 (top-event robustness 우선)')
ax.grid(True, alpha=0.3)

# (b) λ sweep
ax = axes[0, 1]
xs = np.arange(len(lambda_sweep))
ax.bar(xs - 0.2, lambda_sweep.NMAE, 0.4, label='Overall NMAE', color='#1976D2')
ax2 = ax.twinx()
ax2.bar(xs + 0.2, lambda_sweep['std'], 0.4, label='per-seed std', color='#FB8C00')
ax.set_xticks(xs)
ax.set_xticklabels(lambda_sweep.config, rotation=15)
ax.set_ylabel('NMAE (%)', color='#1976D2')
ax2.set_ylabel('per-seed std (pp)', color='#FB8C00')
ax.set_title('λ sweep — 2-branch λ=2.0 채택 (NMAE 우세 + std −47%)')
ax.set_ylim(4.65, 4.78)
ax.grid(True, alpha=0.3, axis='y')

# (c) Gate sweep — gate ratio vs dead seeds
ax = axes[1, 0]
xs = np.arange(len(gate_sweep))
colors = ['#388E3C' if d == 0 else ('#FB8C00' if d <= 1 else '#D32F2F') for d in gate_sweep.dead_seeds]
bars = ax.bar(xs, gate_sweep.gate_ratio, color=colors)
ax.set_xticks(xs)
ax.set_xticklabels(gate_sweep.config, rotation=15)
ax.set_ylabel('gate ratio (top/non-event)')
ax.set_title('Sparse gate sweep — selectivity↑ 시 dead seeds 폭증 → 폐기')
for i, (g, d) in enumerate(zip(gate_sweep.gate_ratio, gate_sweep.dead_seeds)):
    ax.text(i, g+0.5, f'dead {d}/3', ha='center', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# (d) L horizon 비교 (model_final §1.3 인용)
ax = axes[1, 1]
L_compare = ['L=3\n(short patch)', 'L=6\n(중간)', 'L=12 EOD\n★ 채택']
lead1_keep = [100, 90, 95]   # 상대값
remain_shape = [0, 30, 100]
xs = np.arange(len(L_compare))
ax.bar(xs - 0.2, lead1_keep, 0.4, label='lead-1 정확도 보존', color='#1976D2')
ax.bar(xs + 0.2, remain_shape, 0.4, label='remaining-day shape', color='#388E3C')
ax.set_xticks(xs)
ax.set_xticklabels(L_compare)
ax.set_ylabel('상대 점수 (%)')
ax.set_title('L horizon — L=12 EOD 가 둘 다 만족')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out = EDA_DIR / "03_phase2_architecture.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n저장: {out}")

H_sweep.to_csv(EDA_DIR / "03_H_sweep.csv", index=False)
lambda_sweep.to_csv(EDA_DIR / "03_lambda_sweep.csv", index=False)
gate_sweep.to_csv(EDA_DIR / "03_gate_sweep.csv", index=False)
print(f"저장: 3 csv")
