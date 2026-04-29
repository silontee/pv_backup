"""GK-2A NaN 시각의 실제 날씨 확인 — 고흥만 케이스.

질문: GK-2A 고흥만수상 NaN 시각이 정말 눈/비/안개 오는 날인가?
방법: 동 시각의 ASOS 고흥(stnId=262) rn/ta/dc10Tca/icsr + 고흥만 발전량 대조.
"""
import os
import sys
import glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"D:\pv_backup"

GK2A_COL = "고흥만수상"
SOLAR_SITE = "고흥만 수상태양광"
ASOS_STN = 262  # 고흥

# ---- 1. GK-2A 고흥만 로드 ----
gk2a_files = sorted(glob.glob(os.path.join(ROOT, "data", "gk2a_ghi", "gk2a_ghi_20*.csv")))
gk2a_frames = []
for f in gk2a_files:
    df = pd.read_csv(f, usecols=["datetime_kst", GK2A_COL])
    gk2a_frames.append(df)
gk2a = pd.concat(gk2a_frames, ignore_index=True)
gk2a["datetime_kst"] = pd.to_datetime(gk2a["datetime_kst"])
gk2a = gk2a.rename(columns={GK2A_COL: "gk2a_ghi"})

# 낮 시간(07~18)만 — NaN 분석 대상
gk2a["hour"] = gk2a["datetime_kst"].dt.hour
gk2a_day = gk2a[(gk2a["hour"] >= 7) & (gk2a["hour"] <= 18)].copy()
print(f"GK-2A 고흥만 낮 슬롯: {len(gk2a_day):,}, NaN: {gk2a_day['gk2a_ghi'].isna().sum():,} "
      f"({gk2a_day['gk2a_ghi'].isna().mean()*100:.1f}%)")

# ---- 2. ASOS 고흥 로드 ----
asos_files = sorted(glob.glob(os.path.join(ROOT, "data", "asos_hourly", "asos_hourly_*.csv")))
asos_frames = []
for f in asos_files:
    df = pd.read_csv(f, usecols=["tm", "stnId", "ta", "rn", "hm", "icsr", "dc10Tca"])
    df = df[df["stnId"] == ASOS_STN].copy()
    asos_frames.append(df)
asos = pd.concat(asos_frames, ignore_index=True)
asos["tm"] = pd.to_datetime(asos["tm"])
asos = asos.rename(columns={"tm": "datetime_kst"})
# icsr MJ/m²/h → W/m² (시간평균)
asos["icsr_wm2"] = asos["icsr"] * (1e6 / 3600)
print(f"ASOS 고흥 로드: {len(asos):,} rows")

# ---- 3. 발전량 로드 ----
long_df = pd.read_csv(os.path.join(ROOT, "data", "processed", "solar_hourly_long.csv"),
                     parse_dates=["datetime"])
solar = long_df[long_df["site"] == SOLAR_SITE].copy()
# 사이트 내 호기별 합산 (고흥만수상태양광은 단일 호기일 가능성 높지만 안전하게)
solar = solar.groupby("datetime", as_index=False)["gen_kwh"].sum()
solar = solar.rename(columns={"datetime": "datetime_kst"})
print(f"발전량 로드: {len(solar):,} rows (호기 합산, 용량 63,481 kW)")

# ---- 4. Merge ----
merged = gk2a_day.merge(asos, on="datetime_kst", how="left") \
                 .merge(solar, on="datetime_kst", how="left")
merged["is_nan_gk2a"] = merged["gk2a_ghi"].isna()

# 용량 정규화 발전량 (%)
CAPACITY = 63481
merged["gen_pct"] = merged["gen_kwh"] / CAPACITY * 100

# ---- 5. NaN 시각 분류 ----
nan_df = merged[merged["is_nan_gk2a"]].copy()
ok_df  = merged[~merged["is_nan_gk2a"]].copy()

print("\n=== NaN 시각의 ASOS 날씨 분포 ===")
n_total = len(nan_df)
categories = {
    "강수(rn>0)":      nan_df["rn"].fillna(0) > 0,
    "영하(ta<0)":       nan_df["ta"] < 0,
    "영하+강수(눈)":    (nan_df["ta"] < 0) & (nan_df["rn"].fillna(0) > 0),
    "운량≥8(흐림)":     nan_df["dc10Tca"] >= 8,
    "운량 0~3(맑음)":   nan_df["dc10Tca"] <= 3,
    "ASOS icsr>100W/m²": nan_df["icsr_wm2"] > 100,  # "NaN인데 실제로는 해가 있었다"
    "ASOS icsr<10 W/m²": nan_df["icsr_wm2"] < 10,   # "NaN이 맞음, 햇빛 실측도 0"
}
for label, mask in categories.items():
    cnt = int(mask.sum())
    print(f"  {label:<22} {cnt:>6,} ({cnt/n_total*100:5.1f}%)")

print("\n=== NaN 시각 vs OK 시각 비교 ===")
for col, label in [("ta","기온(℃)"), ("rn","강수(mm)"), ("hm","습도(%)"),
                   ("dc10Tca","운량(0~10)"), ("icsr_wm2","ASOS icsr(W/m²)"),
                   ("gen_pct","발전량(용량%)")]:
    nan_mean = nan_df[col].mean()
    ok_mean  = ok_df[col].mean()
    print(f"  {label:<22}  NaN평균={nan_mean:7.2f}  OK평균={ok_mean:7.2f}")

# ---- 6. GK-2A vs ASOS icsr 비교 (OK 시각만) ----
valid = ok_df.dropna(subset=["gk2a_ghi", "icsr_wm2"])
valid = valid[(valid["gk2a_ghi"] > 5) | (valid["icsr_wm2"] > 5)]  # 둘 중 하나라도 0보다 큼
if len(valid) > 100:
    r = np.corrcoef(valid["gk2a_ghi"], valid["icsr_wm2"])[0, 1]
    rmse = np.sqrt(((valid["gk2a_ghi"] - valid["icsr_wm2"]) ** 2).mean())
    mbe = (valid["gk2a_ghi"] - valid["icsr_wm2"]).mean()
    print(f"\n=== GK-2A vs ASOS icsr (DQF=1 시각, n={len(valid):,}) ===")
    print(f"  R    = {r:.4f}")
    print(f"  R²   = {r**2:.4f}")
    print(f"  RMSE = {rmse:.2f} W/m²")
    print(f"  MBE  = {mbe:+.2f} W/m² (GK-2A - ASOS)")

# ---- 7. 시간대별 NaN 비율 ----
print("\n=== 시간대별 NaN 비율 (07~18 KST) ===")
by_hour = merged.groupby("hour")["is_nan_gk2a"].agg(["sum", "count", "mean"])
for h, row in by_hour.iterrows():
    bar = "█" * int(row["mean"] * 40)
    print(f"  {h:02d}시:  {int(row['sum']):>5,}/{int(row['count']):>5,}  "
          f"{row['mean']*100:5.1f}%  {bar}")

# ---- 8. NaN + 발전량 유의미 = "있었어야 할 데이터" ----
significant = nan_df[nan_df["gen_pct"] > 5]  # 용량의 5% 이상 발전 중인데 NaN
print(f"\n=== NaN인데 발전량 유의미한 경우 (gen_pct > 5%) ===")
print(f"  건수: {len(significant):,} / {len(nan_df):,} ({len(significant)/len(nan_df)*100:.1f}% of NaN)")
print(f"  ASOS 평균 icsr: {significant['icsr_wm2'].mean():.1f} W/m²")
print(f"  ASOS 평균 운량: {significant['dc10Tca'].mean():.1f}/10")
print(f"  ASOS 평균 강수: {significant['rn'].fillna(0).mean():.3f} mm")
print(f"  ASOS 평균 기온: {significant['ta'].mean():.1f} ℃")
print(f"  평균 발전량: {significant['gen_pct'].mean():.1f}% 용량")

# 기록용 저장
out = os.path.join(ROOT, "data", "processed", "goheung_ghi_nan_diagnosis.csv")
merged.to_csv(out, index=False, encoding="utf-8")
print(f"\n저장: {out}")
