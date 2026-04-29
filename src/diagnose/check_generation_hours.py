"""07~18 KST GHI 수집 창이 실제 발전량을 얼마나 놓치는지 확인.

호기별·월별로:
  - 발전량 분포 by hour
  - 07~18 창 내부/외부 발전량 비율
  - 외부(05~06, 19~20)에서 발전이 유의미한 시간대 특정
"""
import os, sys, glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\pv_backup"

GHI_HOURS = set(range(7, 19))  # 07~18 inclusive

long_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "solar_hourly_long.csv"),
                     parse_dates=["datetime"])
long_df["hour"] = long_df["datetime"].dt.hour
long_df["month"] = long_df["datetime"].dt.month

# ---- 1. 호기별 전체 커버리지 ----
print("=" * 80)
print("호기별 발전량의 07~18 창 커버리지 (전 기간)")
print("=" * 80)
print(f"{'site_unit':<30} {'총 gen_kwh':>15} {'07-18 비율':>10} {'외부 비율':>10}")
print("-" * 80)

per_unit = []
for unit, g in long_df.groupby("site_unit"):
    total = g["gen_kwh"].sum()
    if total <= 0:
        continue
    inside = g[g["hour"].isin(GHI_HOURS)]["gen_kwh"].sum()
    outside_pct = (total - inside) / total * 100
    inside_pct = inside / total * 100
    per_unit.append((unit, total, inside_pct, outside_pct))
    print(f"{unit:<30} {total:>15,.0f} {inside_pct:>9.2f}% {outside_pct:>9.2f}%")

# ---- 2. 시간대별 전체 합산 프로파일 ----
print("\n" + "=" * 80)
print("시간대별 발전량 프로파일 (전 호기 합산, 전 기간)")
print("=" * 80)
by_hour = long_df.groupby("hour")["gen_kwh"].sum()
total_all = by_hour.sum()
print(f"{'hour':<6} {'gen_kwh':>15} {'전체대비':>10} {'누적':>10}")
cum = 0
for h in range(24):
    v = by_hour.get(h, 0)
    pct = v / total_all * 100
    cum += pct
    in_ghi = "✓" if h in GHI_HOURS else " "
    bar = "█" * int(pct / 2)
    print(f" {h:02d} {in_ghi}  {v:>15,.0f} {pct:>9.2f}% {cum:>9.2f}%  {bar}")

# ---- 3. 월별 외부(06·19·20시) 발전 비중 ----
print("\n" + "=" * 80)
print("월별 커버리지 — 07~18 외부에서 누수되는 발전량")
print("=" * 80)
print(f"{'월':<4} {'총량':>15} {'외부비율':>10} {'05시':>8} {'06시':>8} {'19시':>8} {'20시':>8}")
print("-" * 70)
by_month = long_df.groupby(["month", "hour"])["gen_kwh"].sum().unstack(fill_value=0)
for m in range(1, 13):
    if m not in by_month.index:
        continue
    row = by_month.loc[m]
    total_m = row.sum()
    inside_m = row[[h for h in GHI_HOURS if h in row.index]].sum()
    outside_pct = (total_m - inside_m) / total_m * 100 if total_m > 0 else 0
    def cell(h):
        v = row.get(h, 0)
        return f"{v/total_m*100:>7.2f}%" if total_m > 0 else "   -  "
    print(f"{m:02d}   {total_m:>15,.0f} {outside_pct:>9.2f}% "
          f"{cell(5)} {cell(6)} {cell(19)} {cell(20)}")

# ---- 4. 월별·호기별 요약 (19시만 초점) ----
print("\n" + "=" * 80)
print("19시 발전량 비중 (여름철 피크 관심 시간대)")
print("=" * 80)
summer = long_df[long_df["month"].isin([5, 6, 7, 8])]
by_unit_h = summer.groupby(["site_unit", "hour"])["gen_kwh"].sum().unstack(fill_value=0)
print(f"{'site_unit':<30} {'총(5~8월)':>15} {'19시비율':>10} {'20시비율':>10}")
for unit, row in by_unit_h.iterrows():
    total_u = row.sum()
    if total_u <= 0:
        continue
    h19 = row.get(19, 0) / total_u * 100
    h20 = row.get(20, 0) / total_u * 100
    print(f"{unit:<30} {total_u:>15,.0f} {h19:>9.2f}% {h20:>9.2f}%")
