"""EDA 01 — 사이트 선정 근거.

목적:
  - 8 사이트 portfolio 의 cf 분포 / NMAE 분포 / capacity-weight 시각화
  - 왜 광양항·예천·창원 이 outlier 인지 + 그래도 portfolio 가 안정적인 이유
  - 사이트별 처치 차별화 정당화

→ 모델 함의:
  - cap 가중 portfolio 는 고흥만수상 (63 MW, 82%) 이 dominant → portfolio NMAE 3.75% (안정)
  - site-level outlier (광양항 NMAE 8.3% bias +7%, 예천 anomaly_zero) 는 *분리 보고*
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eda_v4"))
from _common import ROOT

EDA_DIR = ROOT / "pv/eda_pv_model"
EDA_DIR.mkdir(exist_ok=True)

# === 1. 사이트별 NMAE / bias / Cov80 / Cov95 (Phase 1 ensemble) ===
ens_per_site = pd.read_csv(ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/ensemble_per_site.csv")
ens_per_site = ens_per_site.sort_values('nmae').reset_index(drop=True)
print("[Phase 1 ensemble per-site]")
print(ens_per_site.round(2).to_string(index=False))

# === 2. capacity ===
p1 = pd.read_parquet(ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/ensemble_test.parquet")
caps = p1.groupby('site').site_capacity_kw.first().to_dict()
ens_per_site['cap_kW'] = ens_per_site.site.map(caps)
ens_per_site['cap_MW'] = ens_per_site.cap_kW / 1000

# === 3. 시각화 ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) NMAE vs capacity
ax = axes[0]
colors = ['#388E3C' if v < 6 else ('#FB8C00' if v < 8 else '#D32F2F') for v in ens_per_site.nmae]
ax.scatter(ens_per_site.cap_MW, ens_per_site.nmae, s=ens_per_site.cap_MW*5+50,
            c=colors, alpha=0.8, edgecolor='black')
for _, r in ens_per_site.iterrows():
    ax.annotate(r.site, (r.cap_MW, r.nmae), xytext=(5, 5), textcoords='offset points', fontsize=10)
ax.set_xlabel('capacity (MW)')
ax.set_ylabel('Phase 1 NMAE (%)')
ax.set_title('사이트별 NMAE vs capacity — 큰 cap (고흥만수상) 이 portfolio dominant')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

# (b) bias 분포
ax = axes[1]
ax.barh(ens_per_site.site, ens_per_site.bias,
         color=['#D32F2F' if abs(b)>3 else ('#FB8C00' if abs(b)>1 else '#388E3C') for b in ens_per_site.bias])
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel('bias (%)')
ax.set_title('사이트별 bias — 광양항 +7% (post-hoc 보정 후보)')
ax.grid(True, alpha=0.3, axis='x')
for i, b in enumerate(ens_per_site.bias):
    ax.text(b + 0.2 if b>0 else b - 0.2, i, f'{b:+.1f}', va='center',
            ha='left' if b>0 else 'right', fontsize=9)

# (c) capacity 비중 pie
ax = axes[2]
ens_per_site_sorted = ens_per_site.sort_values('cap_MW', ascending=False)
ax.pie(ens_per_site_sorted.cap_MW, labels=ens_per_site_sorted.site,
        autopct='%1.1f%%', startangle=90,
        colors=plt.cm.tab10.colors[:len(ens_per_site)])
ax.set_title('portfolio capacity 비중 — 고흥만수상 압도 (82%)')

plt.tight_layout()
out = EDA_DIR / "01_site_selection.png"
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"\n저장: {out}")

# === 4. portfolio NMAE 산출 ===
p1['mu_kw'] = p1.mu_mean * p1.site_capacity_kw
p1['act_kw'] = p1.cf * p1.site_capacity_kw
g = p1.groupby('datetime_kst').agg(mu=('mu_kw','sum'), act=('act_kw','sum'),
                                    cap=('site_capacity_kw','sum')).reset_index()
g['err_pct'] = (g.mu - g.act).abs() / g.cap * 100
port_nmae = g.err_pct.mean()
site_nmae_mean = ens_per_site.nmae.mean()
site_nmae_capw = (ens_per_site.nmae * ens_per_site.cap_MW).sum() / ens_per_site.cap_MW.sum()
print(f"""
[Portfolio 효과]
  - 단순 site mean NMAE  : {site_nmae_mean:.2f}%
  - cap-weighted site NMAE : {site_nmae_capw:.2f}%
  - portfolio NMAE          : {port_nmae:.2f}%

→ cap-weighted ≈ portfolio (고흥만수상 63 MW 가 dominant)
→ site-level outlier (광양항 8.3%, 예천 8.1%, 창원 8.5%) 가 portfolio 에 영향 작음
""")

ens_per_site.to_csv(EDA_DIR / "01_site_metrics.csv", index=False)
print(f"저장: {EDA_DIR / '01_site_metrics.csv'}")
