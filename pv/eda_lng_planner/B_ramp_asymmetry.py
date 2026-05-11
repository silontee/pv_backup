"""EDA B — 호기별 ramp 비대칭 분석.

목적:
  잔존 over-commit 640 MWh의 메인 원인이 *호기 자체의 느린 하강 한도(ramp_dn)* 임을 데이터로 입증.
  4년 hourly LNG 발전량으로 ramp_up vs ramp_dn 분포 비교.

핵심 메시지:
  - baseload 호기 (CG6, CS2): ramp_up p90 ≈ 18~19 MW/h vs ramp_dn p90 ≈ 1.9~2.7 MW/h (7~10x)
  - 한 번 올라간 출력이 *천천히* 만 내려갈 수 있음 → 회복 후에도 잔류 → over-commit
  - 모델이 인위적으로 만든 게 아니라 *실제 운영 패턴* 그대로

→ 모델 함의:
  - over-commit 의 실측 근거 = 운영 데이터 자체의 비대칭
  - 향후 옵션 A (ramp_dn p99 사용) 시 over-commit 추가 감축 가능
  - 단 운영 직관 약화 (실제 plant도 천천히 내림)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, EDA_DIR, LNG_UNITS, UNIT_COLORS, GT_UNITS, ST_UNITS

# === 1. 4년치 hourly LNG 데이터 + ramp 통계 (build_lng_unit_specs.py 와 동일 방법) ===
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
out['mw_diff'] = out.groupby('unit')['mw'].diff()
print(f"loaded {len(out):,} rows  (4년 hourly)")

# === 2. 호기별 ramp 분포 — 운전 중만 (mw>5) ===
running = out[(out.mw > 5) & (out.mw_diff.notna())].copy()

# === 3. 시각화: ramp_up p90 vs ramp_dn p90 (호기별 막대) ===
stats = []
for u in LNG_UNITS:
    s = running[running.unit == u]
    pos = s[s.mw_diff > 0].mw_diff.values
    neg = -s[s.mw_diff < 0].mw_diff.values
    if len(pos) == 0 or len(neg) == 0:
        continue
    stats.append({
        'unit': u, 'type': 'GT' if u.startswith('CG') else 'ST',
        'ramp_up_p90': float(np.percentile(pos, 90)),
        'ramp_up_p95': float(np.percentile(pos, 95)),
        'ramp_up_p99': float(np.percentile(pos, 99)),
        'ramp_dn_p90': float(np.percentile(neg, 90)),
        'ramp_dn_p95': float(np.percentile(neg, 95)),
        'ramp_dn_p99': float(np.percentile(neg, 99)),
    })
sdf = pd.DataFrame(stats)
sdf['ratio_p90'] = sdf.ramp_up_p90 / sdf.ramp_dn_p90
sdf['ratio_p99'] = sdf.ramp_up_p99 / sdf.ramp_dn_p99
print("\n[호기별 ramp 통계 (4년 운전 중)]")
print(sdf.round(2).to_string(index=False))

# === 4. 그림: ramp_up vs ramp_dn 막대 + 비율 + 분포 ===
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) ramp_up vs ramp_dn 막대 (호기별)
ax = axes[0]
xs = np.arange(len(sdf))
w = 0.35
ax.bar(xs - w/2, sdf.ramp_up_p90, w, label='상승 한도 p90', color='#388E3C')
ax.bar(xs + w/2, sdf.ramp_dn_p90, w, label='하강 한도 p90', color='#D32F2F')
ax.set_xticks(xs)
ax.set_xticklabels(sdf.unit, rotation=0)
ax.set_ylabel('MW/h')
ax.set_title('호기별 시간당 변화 한도 (p90)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# (b) 비율 (up/dn)
ax = axes[1]
colors = [UNIT_COLORS[u] for u in sdf.unit]
ax.bar(xs, sdf.ratio_p90, color=colors)
ax.axhline(1, color='gray', ls=':', lw=1)
ax.set_xticks(xs)
ax.set_xticklabels(sdf.unit, rotation=0)
ax.set_ylabel('상승 / 하강 비율')
ax.set_title('비대칭 비율 (1 = 대칭)')
ax.grid(True, alpha=0.3, axis='y')
for i, r in enumerate(sdf.ratio_p90):
    ax.text(i, r+0.1, f'{r:.1f}x', ha='center', fontsize=10)

# (c) baseload (CG6, CS2) 의 ramp_dn 분포 (실제 데이터)
ax = axes[2]
for u, c in [('CG6', '#0D47A1'), ('CS2', '#1976D2'), ('CS1', '#42A5F5')]:
    s = running[running.unit == u]
    neg = -s[s.mw_diff < 0].mw_diff.values
    ax.hist(neg, bins=50, alpha=0.5, label=f'{u} (n={len(neg)})',
            color=c, density=True, range=(0, 30))
ax.axvline(2.7, color='black', ls='--', lw=1, alpha=0.5)
ax.text(2.8, ax.get_ylim()[1]*0.85, 'CS2 p90', fontsize=9)
ax.set_xlabel('|ΔMW/h| (하강 시)')
ax.set_ylabel('density')
ax.set_title('baseload 호기 하강 분포 — 대부분 0~5 MW/h')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_png = EDA_DIR / "B_ramp_asymmetry.png"
plt.savefig(out_png, dpi=130, bbox_inches='tight')
print(f"\n저장: {out_png}")

# === 5. CSV 저장 ===
sdf.to_csv(EDA_DIR / "B_ramp_stats.csv", index=False)
print(f"저장: {EDA_DIR / 'B_ramp_stats.csv'}")

# === 6. 모델 함의 ===
print("\n" + "="*60)
print("→ 모델 함의")
print("="*60)
worst = sdf.sort_values('ratio_p90', ascending=False).iloc[0]
print(f"""
1. 가장 비대칭 호기: {worst.unit} ({worst.type}) — 상승 {worst.ramp_up_p90:.1f} / 하강 {worst.ramp_dn_p90:.2f} = {worst.ratio_p90:.1f}x
2. baseload (CG6, CS2) 가 특히 느림 (운영상 *천천히 내림* 패턴)
3. 한 번 올라간 출력이 다음 시각에 actual_gap 줄어도 *느린 ramp_dn 한도* 만큼만 감소
4. 누적 잔류 ΔP = 잔존 over-commit 640 MWh 의 메인 원인

→ 모델이 만든 인위적 결과 X. 실제 운영 패턴 그대로 반영.
→ §15 옵션 A (ramp_dn p99 사용) 시 추가 감축 가능하지만 운영 직관 약화.
""")
