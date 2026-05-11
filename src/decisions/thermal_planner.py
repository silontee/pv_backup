"""Online-first thermal backup planner.

전략 (priority order):
  1. Online baseload/mid-merit 조정  — 이미 가동 중인 호기에서 ramp up
  2. Peaker 활성화                  — 부족 시 빠른 대응 가능 호기 추가
  3. Cold-start                    — 장기 부족 예측 시 신규 호기 startup

Inputs (per dispatch event):
  required_backup_mw   : PV shortfall으로 추가 필요한 MW (양수면 부족)
  shortfall_horizon_h  : 부족 지속 예상 시간 (h)
  current_state        : 호기별 현재 상태 (online?, output, time_running)

Output:
  dispatch plan: 호기별 출력 변경 (online ramp / startup / shutdown)

Data:
  thermal_unit_profile.csv:
    호기, pmax_p95, pmax_obs, pmin_p05_run, mode (mid-merit/peaker), ...
    headroom_avg, downroom_avg, ramp_up_p95, ramp_up_p99,
    starts, cold_starts, warm_starts, avg_run_hrs, avg_off_hrs, backup_score
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


# ===== Unit state representation =====

@dataclass
class UnitState:
    name: str
    mode: str                         # "baseload" | "mid-merit" | "peaker"
    pmax: float                       # observed
    pmin: float                       # observed
    headroom: float                   # avg headroom in MW
    ramp_up_per_h: float              # MW/h (p95)
    is_online: bool                   # 현재 가동 중인가
    current_output_mw: float          # 현재 출력
    cold_start_hrs: float = 6.0       # cold start 시간 (peaker는 빠름)
    warm_start_hrs: float = 2.0
    backup_score: float = 0.5         # backup 적합도 (0~1, profile에서)


@dataclass
class DispatchAction:
    unit: str
    action: str          # "ramp_up" | "ramp_down" | "startup" | "shutdown"
    delta_mw: float      # 출력 변경량 (양수면 증가)
    duration_h: float = 1.0
    reason: str = ""


@dataclass
class DispatchPlan:
    actions: List[DispatchAction] = field(default_factory=list)
    unmet_mw: float = 0.0             # 계획으로 cover 못한 부족분
    total_added_mw: float = 0.0
    notes: List[str] = field(default_factory=list)


# ===== Planner =====

class OnlineFirstPlanner:
    """Online-first thermal backup planner.

    호기 우선순위:
      1. Online + 큰 headroom + 작은 ramp_up: ramp 빠르고 추가 비용 없음
      2. Peaker (offline): warm start 가능하면 빠른 대응
      3. Mid-merit/baseload (offline): cold start, 장기 부족 시만
    """

    def __init__(self, profile_path: Optional[Path] = None):
        if profile_path is None:
            profile_path = ROOT / "data/processed/thermal_unit_profile.csv"
        self.profile = pd.read_csv(profile_path)
        self.profile.columns = [c.strip() for c in self.profile.columns]

    def units_from_profile(self, online_units: Optional[List[str]] = None,
                            current_outputs: Optional[dict] = None) -> List[UnitState]:
        """profile + 외부 상태 → UnitState 리스트.

        online_units: 현재 가동 중인 호기 이름 리스트. None이면 mode≠peaker 가정.
        current_outputs: dict[name → MW] 현재 출력
        """
        units = []
        for _, r in self.profile.iterrows():
            name = r["호기"]
            mode = r["mode"]
            pmax = float(r["pmax_obs"])
            pmin = float(r["pmin_p05_run"])
            headroom = float(r["headroom_avg"])
            ramp = float(r["ramp_up_p95"])
            backup_score = float(r["backup_score"])

            if online_units is not None:
                online = name in online_units
            else:
                online = mode != "peaker"   # peaker는 default offline

            output_now = (current_outputs or {}).get(name, pmin if online else 0.0)

            units.append(UnitState(
                name=name, mode=mode, pmax=pmax, pmin=pmin,
                headroom=headroom, ramp_up_per_h=ramp,
                is_online=online, current_output_mw=output_now,
                backup_score=backup_score,
            ))
        return units

    def plan(self, required_backup_mw: float,
              shortfall_horizon_h: float,
              units: List[UnitState]) -> DispatchPlan:
        """Online-first dispatch.

        Steps:
          1. Online units에서 ramp up (headroom 만큼) → priority 1
          2. 부족 시 peaker startup (warm start 가능, 1-2h)
          3. 장기 부족 (≥ 6h) + 더 부족 시 baseload/mid-merit startup
        """
        plan = DispatchPlan()
        remaining = required_backup_mw

        # ===== Step 1: Online units ramp up =====
        # Sort: 큰 ramp_up_per_h × 큰 headroom 순
        online = [u for u in units if u.is_online]
        online.sort(key=lambda u: -(u.headroom * u.ramp_up_per_h))
        for u in online:
            if remaining <= 0:
                break
            available_ramp = min(u.headroom, u.ramp_up_per_h * shortfall_horizon_h)
            available_ramp = min(available_ramp, u.pmax - u.current_output_mw)
            if available_ramp <= 0:
                continue
            delta = min(available_ramp, remaining)
            plan.actions.append(DispatchAction(
                unit=u.name, action="ramp_up", delta_mw=delta,
                duration_h=shortfall_horizon_h,
                reason=f"online ramp (headroom {u.headroom:.1f}, ramp_up {u.ramp_up_per_h:.1f})",
            ))
            remaining -= delta
            plan.total_added_mw += delta

        # ===== Step 2: Peaker startup (warm start, 1-2h) =====
        if remaining > 0:
            peakers = [u for u in units if not u.is_online and u.mode == "peaker"]
            peakers.sort(key=lambda u: -u.backup_score)
            for u in peakers:
                if remaining <= 0:
                    break
                ramp_up_window = max(0, shortfall_horizon_h - u.warm_start_hrs)
                available = min(u.pmax - u.pmin, u.ramp_up_per_h * ramp_up_window)
                if available <= 0:
                    continue
                delta = min(available, remaining)
                plan.actions.append(DispatchAction(
                    unit=u.name, action="startup", delta_mw=delta,
                    duration_h=shortfall_horizon_h,
                    reason=f"peaker warm-start (backup_score {u.backup_score:.2f})",
                ))
                remaining -= delta
                plan.total_added_mw += delta

        # ===== Step 3: Cold start (장기 부족 only) =====
        if remaining > 0 and shortfall_horizon_h >= 6:
            cold = [u for u in units if not u.is_online and u.mode != "peaker"]
            cold.sort(key=lambda u: -u.backup_score)
            for u in cold:
                if remaining <= 0:
                    break
                ramp_up_window = max(0, shortfall_horizon_h - u.cold_start_hrs)
                if ramp_up_window <= 0:
                    continue
                available = min(u.pmax - u.pmin, u.ramp_up_per_h * ramp_up_window)
                delta = min(available, remaining)
                plan.actions.append(DispatchAction(
                    unit=u.name, action="startup", delta_mw=delta,
                    duration_h=shortfall_horizon_h,
                    reason=f"cold-start (long shortfall, backup_score {u.backup_score:.2f})",
                ))
                remaining -= delta
                plan.total_added_mw += delta

        plan.unmet_mw = remaining
        if remaining > 0:
            plan.notes.append(f"⚠️ unmet shortfall: {remaining:.1f} MW")
        return plan


# ===== Quick demo / sanity check =====

def _demo():
    planner = OnlineFirstPlanner()
    units = planner.units_from_profile()
    print("Loaded units:")
    for u in units:
        print(f"  {u.name:>4}: {u.mode:>10} pmax={u.pmax:.0f} headroom={u.headroom:.1f} "
              f"ramp={u.ramp_up_per_h:.1f} score={u.backup_score:.2f} online={u.is_online}")

    print("\n=== Demo: required 50 MW, horizon 3h ===")
    plan = planner.plan(required_backup_mw=50, shortfall_horizon_h=3, units=units)
    for a in plan.actions:
        print(f"  {a.unit}: {a.action} +{a.delta_mw:.1f} MW for {a.duration_h:.1f}h | {a.reason}")
    print(f"  Total added: {plan.total_added_mw:.1f} MW, unmet: {plan.unmet_mw:.1f} MW")
    if plan.notes:
        for n in plan.notes:
            print(f"  {n}")

    print("\n=== Demo: required 200 MW, horizon 8h ===")
    plan = planner.plan(required_backup_mw=200, shortfall_horizon_h=8, units=units)
    for a in plan.actions:
        print(f"  {a.unit}: {a.action} +{a.delta_mw:.1f} MW for {a.duration_h:.1f}h | {a.reason}")
    print(f"  Total added: {plan.total_added_mw:.1f} MW, unmet: {plan.unmet_mw:.1f} MW")


if __name__ == "__main__":
    _demo()
