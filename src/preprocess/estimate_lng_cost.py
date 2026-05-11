"""분당 LNG 변동비 추정 — 효율 측정 + 표준 단가로 산출.

입력:
  - data/fuel_procurement/  (사업소·월별 LNG 조달량, kg 추정)
  - data/thermal_hourly/    (분당화력 시간별 발전량, kWh)

출력:
  - data/processed/lng_cost.csv  (월별 효율, 변동비)

가정 (PoC):
  - LNG 발열량: 50 MJ/kg (천연가스 표준)
  - LNG 단가: 1000 원/kg (2022~2025 평균 추정, 변동 큼)
  - 1 kWh = 3.6 MJ
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# 표준 가정
LNG_HEAT_MJ_PER_KG = 50.0       # 천연가스 발열량 (MJ/kg)
KWH_PER_MJ = 1.0 / 3.6          # MJ → kWh
LNG_PRICE_KRW_PER_KG = 1000.0   # 평균 단가 (변동 큼, PoC 가정)


def load_lng_procurement() -> pd.DataFrame:
    """사업소별 월별 LNG 조달량 (분당화력만 추출)."""
    files = sorted((ROOT / "data" / "fuel_procurement").glob("fuel_procurement_*.csv"))
    dfs = []
    for f in files:
        d = pd.read_csv(f, encoding="cp949", sep="|")
        d.columns = [c.strip() for c in d.columns]
        d["사업소"] = d["사업소"].astype(str).str.strip()
        d = d[d["사업소"] == "분당화력"].copy()
        d["일자"] = d["일자"].astype(int)
        dfs.append(d[["사업소", "일자", "LNG"]])
    return pd.concat(dfs, ignore_index=True)


def load_lng_generation() -> pd.DataFrame:
    """분당화력 월별 총 발전량 (thermal_hourly 합계)."""
    files = sorted((ROOT / "data" / "thermal_hourly").glob("thermal_hourly_*.csv"))
    dfs = []
    for f in files:
        d = pd.read_csv(f, index_col=False)
        d.columns = [c.strip() for c in d.columns]
        d["호기"] = d["호기"].astype(str).str.strip()
        d["일자_str"] = d["일자"].astype(str).str.strip()
        d = d[d["일자_str"].str.match(r"^\d{4}-\d{2}-\d{2}")].copy()
        d["일자"] = pd.to_datetime(d["일자_str"])
        # 평균(KW) × 24h × n_days = 월 총 발전량 (kWh)
        # 단순화: 24개 시간 컬럼 합산
        hour_cols = [f"{h}시 발전량(MWh)" for h in range(1, 25)]
        d["day_total_kwh"] = d[hour_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        d["yyyymm"] = d["일자"].dt.strftime("%Y%m").astype(int)
        # 호기 outlier 제거 (이전 분석과 동일 로직)
        # 단순화: 일별 총량이 호기 capacity × 24h 초과면 outlier
        # CG 호기 cap 130 MW = 130000 kWh × 24 = 3,120,000 kWh/일
        # CS 호기 cap 200 MW = 4,800,000 kWh/일
        cap_per_unit = {"CG1": 130, "CG2": 130, "CG3": 130, "CG4": 130,
                        "CG5": 130, "CG6": 130, "CG7": 130, "CG8": 130,
                        "CS1": 200, "CS2": 200}
        d["unit_cap_mw"] = d["호기"].map(cap_per_unit)
        d["max_day_kwh"] = d["unit_cap_mw"] * 1000 * 24  # MW × 1000 × 24h
        d.loc[d["day_total_kwh"] > d["max_day_kwh"], "day_total_kwh"] = np.nan

        dfs.append(d[["호기", "yyyymm", "day_total_kwh"]])

    df = pd.concat(dfs, ignore_index=True)
    monthly = df.groupby("yyyymm", as_index=False)["day_total_kwh"].sum().rename(columns={"day_total_kwh": "gen_kwh"})
    return monthly


def main():
    print("[1/3] 분당화력 LNG 조달량 로딩...")
    lng = load_lng_procurement()
    print(f"  rows: {len(lng)}, 기간: {lng['일자'].min()} ~ {lng['일자'].max()}")
    print(f"  LNG 컬럼 단위 추측: kg 또는 톤? — 확인 필요")
    print(f"  월평균 LNG: {lng['LNG'].mean():,.0f}")
    print(f"  LNG 누적: {lng['LNG'].sum():,.0f}")

    print("\n[2/3] 분당화력 월별 발전량 로딩...")
    gen = load_lng_generation()
    print(f"  rows: {len(gen)}, 기간: {gen['yyyymm'].min()} ~ {gen['yyyymm'].max()}")
    print(f"  월평균 발전량: {gen['gen_kwh'].mean()/1e6:,.1f} GWh")

    print("\n[3/3] 효율 + 변동비 계산...")
    df = lng.merge(gen, left_on="일자", right_on="yyyymm", how="inner")
    df = df.rename(columns={"LNG": "lng_consumed"})

    # 단위 검증: LNG_consumed × heat × η ≈ gen_kwh?
    # heat = 50 MJ/kg, η = 0.5 가정
    # gen_kwh = LNG(kg) × 50 × 0.5 / 3.6
    df["expected_gen_kwh_at_50pct"] = df["lng_consumed"] * LNG_HEAT_MJ_PER_KG * 0.5 * KWH_PER_MJ
    df["ratio"] = df["gen_kwh"] / df["expected_gen_kwh_at_50pct"]

    print(f"\n발전량 / 50% 효율 가정 발전량 비율 (1.0이면 50% 효율 일치):")
    print(df[["일자", "lng_consumed", "gen_kwh", "expected_gen_kwh_at_50pct", "ratio"]].head(10).to_string(index=False))
    print(f"\n  평균 비율: {df['ratio'].mean():.3f}")
    print(f"  → 비율 ~1 이면 LNG 단위가 kg, 효율 ~50%")
    print(f"  → 비율 ~0.001 이면 LNG 단위가 톤")

    # 효율 역계산
    # gen_kwh = LNG(unit) × heat × η / 3.6
    # η = gen_kwh × 3.6 / (LNG × heat)
    df["efficiency_if_kg"] = df["gen_kwh"] * 3.6 / (df["lng_consumed"] * LNG_HEAT_MJ_PER_KG)
    df["efficiency_if_ton"] = df["gen_kwh"] * 3.6 / (df["lng_consumed"] * 1000 * LNG_HEAT_MJ_PER_KG)

    print(f"\n  LNG가 kg 단위 가정 시 평균 효율: {df['efficiency_if_kg'].mean()*100:.1f}%  (정상 ~50%)")
    print(f"  LNG가 톤 단위 가정 시 평균 효율: {df['efficiency_if_ton'].mean()*100:.1f}%  (정상 ~50%)")

    # 변동비 계산 (가정: 1000원/kg 평균)
    # 변동비 (원/kWh) = LNG단가 / (heat × η / 3.6)
    eff_kg = df["efficiency_if_kg"].mean()
    if 0.3 < eff_kg < 0.7:
        unit_label = "kg"
        eff = eff_kg
    else:
        unit_label = "ton"
        eff = df["efficiency_if_ton"].mean()
    variable_cost_krw_per_kwh = LNG_PRICE_KRW_PER_KG / (LNG_HEAT_MJ_PER_KG * eff / 3.6)

    print(f"\n=== LNG 변동비 추정 ===")
    print(f"  LNG 단위: {unit_label}")
    print(f"  발전 효율: {eff*100:.1f}%")
    print(f"  LNG 단가: {LNG_PRICE_KRW_PER_KG} 원/kg (PoC 가정)")
    print(f"  변동비: {variable_cost_krw_per_kwh:.1f} 원/kWh")
    print(f"  → CLAUDE.md 명시: ~160 원/kWh")

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "lng_cost_monthly.csv"
    df[["일자", "lng_consumed", "gen_kwh", "ratio",
        "efficiency_if_kg", "efficiency_if_ton"]].to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
