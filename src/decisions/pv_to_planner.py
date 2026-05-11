"""PV prediction → Thermal backup planner integration.

Pipeline:
  1. PV model: μ_pred, σ_pred per (site, hour) — perfect-foresight upper bound
  2. Portfolio aggregation: μ_total = Σ μ_i × cap_i (kWh)
  3. Realtime: actual_t 도착 → gap_t = actual_t - μ_pred_t
  4. Required backup: max(0, -gap_t) → MW conversion
  5. Thermal planner: dispatch plan

Output:
  per-hour dispatch plan + 일별 fuel cost summary

Framing (plan/main/framing.md):
  - PV input weather = actual observation (perfect-foresight upper bound)
  - Real D-1 NMAE은 *추가로 NWP forecast error*까지 포함하므로 *더 나쁨*
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decisions.thermal_planner import OnlineFirstPlanner

LNG_COST_PER_KWH = 169   # 원/kWh (estimate_lng_cost.py 측정값)


def compute_gap_series(predictions_path: Path) -> pd.DataFrame:
    """PV prediction 파일에서 시간별 gap (actual - predicted) 계산.

    Args:
        predictions_path: site-level test_predictions.parquet
            컬럼: datetime_kst, site, site_capacity_kw, cf, pred_cf, pred_std_cf
    """
    pred = pd.read_parquet(predictions_path)
    pred["pred_kwh"] = pred["pred_cf"] * pred["site_capacity_kw"]
    pred["actual_kwh"] = pred["cf"] * pred["site_capacity_kw"]
    pred["sigma_kwh"] = pred["pred_std_cf"] * pred["site_capacity_kw"]
    pred["var_kwh2"] = pred["sigma_kwh"] ** 2

    # Portfolio level: 시간별 합산 (독립 가정)
    port = pred.groupby("datetime_kst", as_index=False).agg(
        pred_kwh=("pred_kwh", "sum"),
        actual_kwh=("actual_kwh", "sum"),
        var_kwh2=("var_kwh2", "sum"),
        cap_kw=("site_capacity_kw", "sum"),
    )
    port["sigma_kwh"] = np.sqrt(port["var_kwh2"])
    port["gap_kwh"] = port["actual_kwh"] - port["pred_kwh"]   # signed: 음수면 under-deliver
    port["abs_gap_kwh"] = port["gap_kwh"].abs()
    port["required_backup_mw"] = (-port["gap_kwh"]).clip(lower=0) / 1000   # MW
    port["excess_pv_mw"] = port["gap_kwh"].clip(lower=0) / 1000
    return port


def run_dispatch_per_hour(port: pd.DataFrame,
                          horizon_h: float = 1.0,
                          planner: OnlineFirstPlanner = None) -> pd.DataFrame:
    """매 시간 dispatch plan 생성 (시간별 결정).

    Args:
        port: portfolio-level df with required_backup_mw 컬럼
        horizon_h: dispatch 결정 horizon (1h = 즉시 다음 시간만)
        planner: OnlineFirstPlanner 인스턴스. None이면 기본 생성

    Returns:
        df with: datetime_kst, required_backup_mw, dispatched_mw, unmet_mw,
                 n_actions, lng_cost_won, action_summary
    """
    if planner is None:
        planner = OnlineFirstPlanner()

    units = planner.units_from_profile()   # default: peaker offline, others online

    rows = []
    for _, r in port.iterrows():
        req = float(r["required_backup_mw"])
        if req <= 0:
            rows.append({
                "datetime_kst": r["datetime_kst"],
                "required_backup_mw": 0.0,
                "dispatched_mw": 0.0,
                "unmet_mw": 0.0,
                "n_actions": 0,
                "lng_cost_won": 0.0,
                "action_summary": "",
            })
            continue
        plan = planner.plan(required_backup_mw=req,
                             shortfall_horizon_h=horizon_h,
                             units=units)
        # LNG 비용 추정: dispatched_mw × 1h × LNG_COST_PER_KWH × 1000 (MW→kW)
        lng_cost = plan.total_added_mw * horizon_h * 1000 * LNG_COST_PER_KWH

        action_summary = "; ".join(
            [f"{a.unit}({a.action})+{a.delta_mw:.1f}MW" for a in plan.actions]
        )
        rows.append({
            "datetime_kst": r["datetime_kst"],
            "required_backup_mw": req,
            "dispatched_mw": plan.total_added_mw,
            "unmet_mw": plan.unmet_mw,
            "n_actions": len(plan.actions),
            "lng_cost_won": lng_cost,
            "action_summary": action_summary,
        })
    return pd.DataFrame(rows)


def summarize(dispatch_df: pd.DataFrame, port: pd.DataFrame, label: str = "Run"):
    """결과 요약."""
    n = len(dispatch_df)
    n_shortfall = (dispatch_df["required_backup_mw"] > 0).sum()
    total_unmet = dispatch_df["unmet_mw"].sum()
    total_dispatched_mwh = dispatch_df["dispatched_mw"].sum()
    total_cost = dispatch_df["lng_cost_won"].sum()

    # PV pred quality
    mae_kwh = port["abs_gap_kwh"].mean()
    nmae_pct = port["abs_gap_kwh"].sum() / port["cap_kw"].sum() * 100

    # Gap distribution
    under_freq = (port["gap_kwh"] < 0).mean() * 100
    over_freq = (port["gap_kwh"] > 0).mean() * 100

    print(f"\n=== {label} 요약 ===")
    print(f"  rows: {n:,}, PV under-deliver 시간: {n_shortfall:,} ({n_shortfall/n*100:.1f}%)")
    print(f"  PV mean gap (signed): {port['gap_kwh'].mean()/1000:+.2f} MWh")
    print(f"  PV NMAE: {nmae_pct:.2f}%")
    print(f"  Under-deliver 빈도: {under_freq:.1f}%, Over-deliver: {over_freq:.1f}%")
    print(f"  Dispatched 총량: {total_dispatched_mwh:.1f} MW·hour")
    print(f"  Unmet 총량: {total_unmet:.1f} MW·hour")
    print(f"  LNG cost (예상 dispatch): {total_cost/1e8:.2f} 억원")

    # Dispatch breakdown by action type
    summary_actions = (dispatch_df.assign(has_action=lambda d: d["n_actions"] > 0)
                                  .groupby("has_action")
                                  .size())
    print(f"  Action 발생: {summary_actions.get(True, 0):,} 시간 / 무액션: {summary_actions.get(False, 0):,}")


def main():
    print("=" * 70)
    print("PV → Thermal Planner Integration")
    print("=" * 70)

    # Best D-1 model (perfect-foresight upper bound) — ResMLP+AdaLN
    pred_path = ROOT / "pv/experiments/resmlp_adaln/test_predictions.parquet"
    print(f"\n[Load] PV predictions: {pred_path}")
    if not pred_path.exists():
        print(f"  ⚠️ 파일 없음. 다른 모델 시도 가능.")
        return
    print(f"  rows: {len(pd.read_parquet(pred_path)):,}")

    print("\n[Compute] gap (signed) per hour...")
    port = compute_gap_series(pred_path)
    print(f"  portfolio rows: {len(port):,}")
    print(f"  date range: {port['datetime_kst'].min()} ~ {port['datetime_kst'].max()}")

    print("\n[Dispatch] hour-by-hour planner (horizon=1h)...")
    planner = OnlineFirstPlanner()
    dispatch = run_dispatch_per_hour(port, horizon_h=1.0, planner=planner)

    out_dir = ROOT / "pv/experiments/pv_to_planner"
    out_dir.mkdir(parents=True, exist_ok=True)
    port.to_parquet(out_dir / "portfolio_gaps.parquet", index=False)
    dispatch.to_parquet(out_dir / "hourly_dispatch.parquet", index=False)
    print(f"  저장: {out_dir / 'hourly_dispatch.parquet'}")

    summarize(dispatch, port, label="ResMLP+AdaLN (perfect-foresight) → Online-first dispatch")

    # 가장 큰 부족 시간 5개
    print("\n[Top 5 largest shortfall hours]")
    top5 = dispatch.nlargest(5, "required_backup_mw")
    for _, r in top5.iterrows():
        print(f"  {r['datetime_kst']}: req {r['required_backup_mw']:.1f} MW, "
              f"dispatched {r['dispatched_mw']:.1f} MW, unmet {r['unmet_mw']:.1f} MW")
        if r["action_summary"]:
            print(f"    actions: {r['action_summary']}")


if __name__ == "__main__":
    main()
