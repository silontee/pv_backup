"""EDA H — 호기별 운전 패턴 (4년 hourly 기반).

목적:
  - LNG plan §6.2 baseload/mid-merit/peaker 분류 정당화
  - "peaker도 6~7h 연속 운전" 가정 폐기 근거 (Q5)
  - 호기별 *운영 빈도* 와 *연속 운전 길이* 동시 시각화

→ 모델 함의:
  - peaker (CG2/CG4/CG5) 도 한 번 켜지면 5~8h 연속 운전 → 단발 spike 가정 X
  - mode 분류는 *운전 빈도* 차이일 뿐, *연속 운전 길이* 짧다는 의미 X
  - 따라서 호기 선택 룰에서 mode 우선순위는 근거 약함 → 헤드룸/ramp 기반이 옳음
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, EDA_DIR, LNG_UNITS, UNIT_COLORS

# === 1. 4년치 hourly LNG 데이터 ===
import glob
files = sorted(glob.glob(str(ROOT / 'data/thermal_hourly/thermal_hourly_*.csv')))
dfs = []
for f in files:
    try:
        df = pd.read_csv(f, encoding='utf-8', skipinitialspace=True,
                         header=None, skiprows=1, usecols=range(27))
    except: continue
    df.columns = ['plant','unit','date'] + [f'h{h}' for h in range(1, 25)]
    df['unit'] = df['unit'].astype(str).str.strip()
    df = df[df.unit.isin(LNG_UNITS)].copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    long = df.melt(id_vars=['unit','date'], value_vars=[f'h{h}' for h in range(1,25)],
                   var_name='hour_str', value_name='mw_kwh')
    long['hour'] = long.hour_str.str.replace('h','').astype(int) - 1
    long['mw'] = pd.to_numeric(long['mw_kwh'], errors='coerce') / 1000.0
    dfs.append(long[['unit','date','hour','mw']])
out = pd.concat(dfs, ignore_index=True)
out['datetime_kst'] = out.date + pd.to_timedelta(out.hour, unit='h')
out = out.sort_values(['unit','datetime_kst']).reset_index(drop=True)

# === 2. 호기별 운전 통계 ===
out['running'] = (out.mw > 5).astype(int)

# 연속 운전 길이 (run-length)
def run_lengths(s):
    """1 (running) sequences 의 길이 list."""
    runs = []
    cur = 0
    for v in s:
        if v == 1:
            cur += 1
        else:
            if cur > 0: runs.append(cur)
            cur = 0
    if cur > 0: runs.append(cur)
    return runs

stats = []
for u in LNG_UNITS:
    s = out[out.unit == u].sort_values('datetime_kst').running.values
    rl = run_lengths(s)
    if not rl: continue
    running_pct = (s.sum() / len(s)) * 100
    if running_pct >= 45:
        mode = 'baseload'
    elif running_pct >= 28:
        mode = 'mid-merit'
    else:
        mode = 'peaker'
    stats.append({
        'unit': u, 'mode': mode,
        'running_pct': running_pct,
        'mean_run_h': float(np.mean(rl)),
        'median_run_h': float(np.median(rl)),
        'max_run_h': int(np.max(rl)),
        'n_runs': len(rl),
    })
sdf = pd.DataFrame(stats).sort_values('running_pct', ascending=False)
print("\n[호기별 운전 패턴 (4년)]")
print(sdf.round(1).to_string(index=False))

# === 3. 시각화 ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) running_pct vs mean_run_h 산점
ax = axes[0]
mode_color = {'baseload': '#0D47A1', 'mid-merit': '#FB8C00', 'peaker': '#D32F2F'}
for _, r in sdf.iterrows():
    ax.scatter(r.running_pct, r.mean_run_h,
                s=200, color=mode_color[r['mode']], edgecolor='black', alpha=0.8)
    ax.annotate(r.unit, (r.running_pct, r.mean_run_h),
                 xytext=(5, 5), textcoords='offset points', fontsize=10)
ax.set_xlabel('운전율 (%)')
ax.set_ylabel('평균 연속 운전 길이 (h)')
ax.set_title('호기별 운전 패턴 — peaker 도 평균 5~8h 연속')
ax.axhline(7, color='gray', ls=':', alpha=0.5)
ax.text(5, 7.2, '7h baseline', fontsize=8, color='gray')
# legend
for m, c in mode_color.items():
    ax.scatter([], [], color=c, s=200, edgecolor='black', label=m)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# (b) 호기별 run-length 분포 violin (peaker 강조)
ax = axes[1]
units_sorted = sdf.unit.tolist()
data = []
for u in units_sorted:
    s = out[out.unit == u].sort_values('datetime_kst').running.values
    rl = run_lengths(s)
    data.append(rl if len(rl) > 0 else [0])
parts = ax.violinplot(data, showmeans=True, showmedians=False)
for i, (pc, u) in enumerate(zip(parts['bodies'], units_sorted)):
    pc.set_facecolor(UNIT_COLORS[u])
    pc.set_alpha(0.6)
ax.set_xticks(range(1, len(units_sorted)+1))
ax.set_xticklabels(units_sorted, rotation=15)
ax.set_ylabel('연속 운전 길이 (h)')
ax.set_title('호기별 연속 운전 분포 — peaker (CG2/CG4) 도 6h+ 흔함')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_png = EDA_DIR / "H_unit_pattern.png"
plt.savefig(out_png, dpi=130, bbox_inches='tight')
print(f"\n저장: {out_png}")

sdf.to_csv(EDA_DIR / "H_unit_pattern.csv", index=False)
print(f"저장: {EDA_DIR / 'H_unit_pattern.csv'}")

# === 모델 함의 ===
print(f"""
{'='*60}
→ 모델 함의
{'='*60}

1. peaker (CG2/CG4/CG5) 도 평균 연속 운전 5~8h — 단발 spike 가정 X
2. mode 차이는 *운전 빈도* (running_pct) 만, *연속 길이* 차이 X
3. 따라서 호기 선택 룰에서 mode 우선순위 보다 *헤드룸 + ramp 기반* 정렬이 옳음

→ LNG planner §6.2 의 "peaker = 단발 spike" 가정 폐기 (Q5) 정당화
""")
