"""시나리오 → LNG 출력 plan + 비용.

입력:
  - data/processed/scenarios_quantile.parquet      (시간별 5 quantile)
  - data/processed/scenarios_trajectories.parquet  (시간별 1000 trajectory)
  - data/processed/thermal_params.csv              (호기별 운전 제약)

출력:
  - pv/experiments/lng_planner/lng_plan_quantile.parquet  (시나리오별 LNG plan)
  - pv/experiments/lng_planner/lng_unit_dispatch.parquet  (호기별 출력 dispatch)

PoC 단순화:
  - 수요 D_t = "예측 시 PV 기대치 (q50)" 가정 (실제론 KPX 수요 외부값)
  - LNG_needed = max(0, D_t - PV_scenario)
    → "내가 신고했던 PV 양이 안 나오면 그 부족분을 LNG로 채움"
  - 호기 분담:
      Base (always on, ~3 호기): CG6 + CG8 + CS2 (총 ~286 MW Pmax 기준 50% 출력)
      Peakers: CG2, CG4 (필요 시 추가 기동)
  - 비용 = 변동비 × LNG kWh + 기동비 × cold_start 횟수
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]


# ========== 운전 파라미터 (`thermal_params.csv` 기반) ==========

LNG_PARAMS = {
    # 비용
    "variable_cost_krw_per_kwh": 165,
    "cold_start_cost_krw": 50_000_000,
    "warm_start_cost_krw": 10_000_000,

    # 호기 분담 (data_strategy + thermal_params 측정 기반)
    "base_units": ["CG6", "CG8", "CS2"],   # 가동률 44~52%
    "mid_merit_units": ["CG1", "CG3", "CG5", "CG7", "CS1"],
    "peaker_units": ["CG2", "CG4"],         # 가동률 20~25%

    # 호기별 Pmax / Pmin (실측 p95 / p05)
    "unit_pmax_mw": {
        "CG1": 87, "CG2": 89, "CG3": 89, "CG4": 89,
        "CG5": 89, "CG6": 87, "CG7": 90, "CG8": 90,
        "CS1": 142, "CS2": 106,
    },
    "unit_pmin_mw": {
        "CG1": 45, "CG2": 34, "CG3": 43, "CG4": 31,
        "CG5": 47, "CG6": 50, "CG7": 46, "CG8": 48,
        "CS1": 43, "CS2": 43,
    },
}


def compute_unit_dispatch(lng_total_mw: float, units_already_on: list = None) -> tuple:
    """LNG 총 필요량 → 호기별 출력 분배.

    PoC 단순 규칙:
    1. Base 3 호기 (CG6, CG8, CS2)는 항상 가동 (총 Pmin 합 ≈ 148 MW)
    2. 부족분 → mid-merit → peakers 순으로 추가 기동
    3. 호기는 Pmin 이상으로만 가동 (그 미만이면 끔)

    Returns:
        (unit_outputs, cold_starts_needed)
        unit_outputs: dict {unit: output_mw}
    """
    if units_already_on is None:
        units_already_on = LNG_PARAMS["base_units"].copy()

    pmax = LNG_PARAMS["unit_pmax_mw"]
    pmin = LNG_PARAMS["unit_pmin_mw"]
    base = LNG_PARAMS["base_units"]
    mid = LNG_PARAMS["mid_merit_units"]
    peakers = LNG_PARAMS["peaker_units"]

    # 1. Base 호기 우선 가동 (균등 분배)
    base_capacity = sum(pmax[u] for u in base)
    base_min = sum(pmin[u] for u in base)

    output = {u: 0 for u in pmax}
    cold_starts = 0
    remaining = lng_total_mw

    if remaining <= 0:
        # PV 충분 → Base 최소 출력만 유지
        for u in base:
            output[u] = pmin[u]
        return output, 0

    # Base 분담 (Pmax까지)
    base_actual = min(remaining, base_capacity)
    # 균등 비율 분배
    if base_capacity > 0:
        ratio = base_actual / base_capacity
        for u in base:
            output[u] = max(pmin[u], pmax[u] * ratio)
        remaining -= sum(output[u] for u in base)
    else:
        remaining = 0

    # 2. 부족 시 mid-merit
    if remaining > 0:
        for u in mid:
            if remaining <= 0:
                break
            extra = min(remaining, pmax[u])
            if extra >= pmin[u]:
                output[u] = extra
                remaining -= extra
                if u not in units_already_on:
                    cold_starts += 1

    # 3. 그래도 부족 → peakers
    if remaining > 0:
        for u in peakers:
            if remaining <= 0:
                break
            extra = min(remaining, pmax[u])
            if extra >= pmin[u]:
                output[u] = extra
                remaining -= extra
                if u not in units_already_on:
                    cold_starts += 1

    return output, cold_starts


def plan_quantile_scenarios(quantile_df: pd.DataFrame) -> pd.DataFrame:
    """5 quantile 시나리오별 LNG plan.

    PoC 단순화:
      D_t = q50 (median, "내가 D-1에 신고한 기대치")
      각 시나리오 PV가 D_t보다 부족하면 → LNG 추가 필요
      과잉이면 → LNG 줄이기 가능
    """
    df = quantile_df.copy()

    # D_t = q50 (median을 기대 PV로 가정)
    df["d_baseline_kwh"] = df["q50_expected"]

    # 각 quantile에 대한 LNG 필요량
    quantile_cols = ["q05_worst", "q25_pessimistic", "q50_expected", "q75_optimistic", "q95_best"]
    for col in quantile_cols:
        scenario = col.split("_")[1]
        df[f"lng_{scenario}_kwh"] = (df["d_baseline_kwh"] - df[col]).clip(lower=0)
        df[f"pv_diff_{scenario}_kwh"] = df["d_baseline_kwh"] - df[col]

    # 호기 dispatch (q50 기준 — 평균 결정)
    dispatch_rows = []
    for _, row in df.iterrows():
        lng_mw = row["lng_expected_kwh"] / 1000   # kWh/h → MW
        output, cold = compute_unit_dispatch(lng_mw)
        rec = {"datetime_kst": row["datetime_kst"]}
        for u, o in output.items():
            rec[f"unit_{u}_mw"] = o
        rec["lng_total_mw"] = lng_mw
        rec["cold_starts"] = cold
        dispatch_rows.append(rec)

    dispatch = pd.DataFrame(dispatch_rows)
    return df, dispatch


def compute_costs(lng_plan: pd.DataFrame) -> dict:
    """LNG plan 누적 비용."""
    var_cost = LNG_PARAMS["variable_cost_krw_per_kwh"]
    cold_cost = LNG_PARAMS["cold_start_cost_krw"]

    # 변동비 (Q50 기준 — 운영자가 신고한 시나리오)
    total_lng_kwh = lng_plan["lng_expected_kwh"].sum()
    variable_cost = total_lng_kwh * var_cost

    return {
        "total_lng_kwh": total_lng_kwh,
        "total_lng_mwh": total_lng_kwh / 1000,
        "variable_cost_krw": variable_cost,
        "variable_cost_billion_krw": variable_cost / 1e9,
    }


def main():
    print("[1/3] 시나리오 로딩...")
    quantile_path = ROOT / "data" / "processed" / "scenarios_quantile.parquet"
    quantile_df = pd.read_parquet(quantile_path)
    print(f"  rows: {len(quantile_df):,}")

    print("\n[2/3] LNG plan 도출 (quantile별)...")
    plan, dispatch = plan_quantile_scenarios(quantile_df)
    print(f"  plan rows: {len(plan):,}, dispatch rows: {len(dispatch):,}")

    # 시나리오별 평균 LNG
    print("\n=== 시나리오별 평균 LNG 필요량 (시간 평균, MW) ===")
    for scen in ["worst", "pessimistic", "expected", "optimistic", "best"]:
        col = f"lng_{scen}_kwh"
        avg_mw = plan[col].mean() / 1000
        max_mw = plan[col].max() / 1000
        print(f"  {scen:<15} avg: {avg_mw:6.1f} MW   max: {max_mw:7.1f} MW")

    print("\n[3/3] 비용 계산 (q50 baseline)...")
    costs = compute_costs(plan)
    print(f"  총 LNG 가동량: {costs['total_lng_mwh']:.1f} MWh ({costs['total_lng_mwh']/1000:.1f} GWh)")
    print(f"  변동비:        {costs['variable_cost_billion_krw']:.2f} 십억원 ({costs['variable_cost_krw']:,.0f} 원)")

    # Save
    out_dir = ROOT / "pv" / "experiments" / "lng_planner"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan.to_parquet(out_dir / "lng_plan_quantile.parquet", index=False)
    dispatch.to_parquet(out_dir / "lng_unit_dispatch.parquet", index=False)
    print(f"\n저장:")
    print(f"  {out_dir / 'lng_plan_quantile.parquet'}")
    print(f"  {out_dir / 'lng_unit_dispatch.parquet'}")

    # 호기별 dispatch 평균
    print()
    print("=== 호기별 평균 출력 (q50 baseline, MW) ===")
    unit_cols = [c for c in dispatch.columns if c.startswith("unit_")]
    for uc in unit_cols:
        unit = uc.replace("unit_", "").replace("_mw", "")
        running = dispatch[uc] > 0
        avg_when_on = dispatch.loc[running, uc].mean() if running.any() else 0
        run_pct = running.mean() * 100
        print(f"  {unit:<5}: 가동률 {run_pct:5.1f}%, 가동 시 평균 {avg_when_on:5.1f} MW")


if __name__ == "__main__":
    main()
