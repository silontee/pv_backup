"""Online-first LNG 백업 플래너.

분당 LNG 10호기 데이터로 PV 변동성 백업 의사결정을 매 시간(1h step) 시뮬레이션.

정책 핵심:
  1. 매 step backup_need(t) — PV 부족분 (MW) — 가 들어옴.
  2. 현재 가동 중 baseload + mid-merit 호기의 (headroom_avg ∩ ramp_up_p95)로 우선 충당.
  3. 잔여분만 가동 중 peaker로.
  4. 그래도 부족하면, 향후 H시간 forecast 모두 부족할 때만 새 peaker startup.
  5. min-up / min-down 제약 (LNG CCGT spec) 강제.

cost-optimal vs risk-aware 두 정책 비교:
  - cost-optimal: H=4, safety_margin=0, accept under-backup
  - risk-aware:   H=2, safety_margin=10 MW, no under-backup tolerance

산출:
  - timeline: per-step dispatch + 이벤트
  - metrics:  fuel_cost / under_backup_freq / peaker_usage
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]

# ===== 비용·운전 상수 (CLAUDE.md / lng_planner.py 표준) =====
VAR_COST_KRW_PER_KWH = 165
COLD_START_COST = 50_000_000   # off >= 24h
WARM_START_COST = 10_000_000   # off 2-24h
MIN_UP_HOURS = 4               # LNG CCGT spec
MIN_DOWN_HOURS = 2
COLD_OFF_HOURS = 24            # 그 이상 정지 = cold


# ===== 호기 상태 =====
@dataclass
class UnitState:
    name: str
    mode: str
    pmax: float
    pmin: float
    headroom_avg: float
    ramp_up: float
    ramp_dn: float
    running: bool = False
    output_mw: float = 0.0
    hours_in_state: int = 999       # 상태 변경 후 경과 시간

    def step_capacity(self) -> float:
        """이 step에서 위로 추가 가능한 MW (running 기준)."""
        if not self.running:
            return 0.0
        ceiling = max(0.0, self.pmax - self.output_mw)
        return min(self.headroom_avg, self.ramp_up, ceiling)

    def can_start(self) -> bool:
        return (not self.running) and (self.hours_in_state >= MIN_DOWN_HOURS)


@dataclass
class PolicyConfig:
    name: str
    persist_threshold_h: int               # forecast 부족 H시간 이상 지속 시 startup
    safety_margin_mw: float                # 매 step 추가 출력 (risk-aware buffer)
    accept_under_backup: bool              # 부족 발생 허용?
    initial_running: tuple[str, ...] = (   # 시뮬 시작 시 가동 중 호기
        "CG6", "CG8", "CS2",               # baseload
        "CS1",                             # 1순위 mid-merit (최고 backup_score)
    )


# ===== 프로파일 로더 =====
def load_profile(path: Path) -> dict[str, UnitState]:
    df = pd.read_csv(path)
    out: dict[str, UnitState] = {}
    for _, r in df.iterrows():
        out[r["호기"]] = UnitState(
            name=r["호기"],
            mode=r["mode"],
            pmax=float(r["pmax_p95"]),
            pmin=float(r["pmin_p05_run"]),
            headroom_avg=float(r["headroom_avg"]),
            ramp_up=float(r["ramp_up_p95"]),
            ramp_dn=float(r["ramp_dn_p95"]),
        )
    return out


def initialize(units: dict[str, UnitState], running: tuple[str, ...]) -> None:
    for name, u in units.items():
        u.running = name in running
        u.output_mw = u.pmin if u.running else 0.0
        u.hours_in_state = 999  # 시작 시 제약 없음


# ===== 정책 1 step =====
def step_decide(
    units: dict[str, UnitState],
    backup_need_mw: float,
    forecast_mw: list[float],
    cfg: PolicyConfig,
) -> dict:
    """한 step 의사결정 + 상태 업데이트.

    Returns: per-step record (dispatch + 이벤트).
    """
    # 매 step 시작 시: 가동 중 호기는 pmin으로 리셋 후 ramp 결정.
    # (실제 운영에선 직전 출력에서 ramp; PoC는 step 단위 결정 단순화)
    for u in units.values():
        if u.running:
            u.output_mw = u.pmin

    target = backup_need_mw + (cfg.safety_margin_mw if backup_need_mw > 0 else 0.0)
    remaining = target
    starts: list[tuple[str, int, str]] = []  # (unit, cost_krw, type)

    # 1) 가동 중 baseload + mid-merit 우선 — capacity 큰 순
    base_mid = sorted(
        [u for u in units.values() if u.running and u.mode in ("baseload", "mid-merit")],
        key=lambda x: -x.step_capacity(),
    )
    for u in base_mid:
        if remaining <= 0:
            break
        give = min(u.step_capacity(), remaining)
        u.output_mw += give
        remaining -= give

    # 2) 가동 중 peaker
    online_peakers = [u for u in units.values() if u.running and u.mode == "peaker"]
    for u in online_peakers:
        if remaining <= 0:
            break
        give = min(u.step_capacity(), remaining)
        u.output_mw += give
        remaining -= give

    # 3) 잔여 → forecast 게이트로 peaker startup
    if remaining > 0:
        H = cfg.persist_threshold_h
        # forecast[0] = 다음 step의 backup_need (이미 현재는 처리 중)
        # 향후 H step 모두 양수면 persistent
        lookahead = forecast_mw[:H]
        persistent = (len(lookahead) >= H) and all(x > 0 for x in lookahead)

        if persistent:
            candidates = sorted(
                [u for u in units.values() if u.mode == "peaker" and u.can_start()],
                key=lambda x: x.hours_in_state,  # warm 우선 (cost↓)
            )
            for u in candidates:
                if remaining <= 0:
                    break
                cost = COLD_START_COST if u.hours_in_state >= COLD_OFF_HOURS else WARM_START_COST
                stype = "cold" if u.hours_in_state >= COLD_OFF_HOURS else "warm"
                u.running = True
                u.output_mw = u.pmin
                give = min(u.headroom_avg, u.ramp_up, u.pmax - u.pmin)
                u.output_mw += give
                remaining -= give
                u.hours_in_state = 0
                starts.append((u.name, cost, stype))
                if cfg.accept_under_backup:
                    break  # cost-optimal: 한 번에 한 호기만

    # 부족분: target 기준 잔여(=margin 포함)와 실 need 기준 두 가지 분리
    margin_short = max(0.0, remaining)             # target에 못 미친 양
    dispatched = target - remaining                 # 실제 공급량
    under_backup = max(0.0, backup_need_mw - dispatched)  # 실 need 부족 — 평가 기준

    # 4) 상태 업데이트 + 기록
    rec: dict = {
        "backup_need_mw": backup_need_mw,
        "target_mw": target,
        "under_backup_mw": under_backup,
        "margin_short_mw": margin_short,
        "lng_total_mw": sum(u.output_mw for u in units.values() if u.running),
        "starts": starts,
        "n_running": sum(int(u.running) for u in units.values()),
        "any_peaker_running": int(any(u.running and u.mode == "peaker" for u in units.values())),
    }
    for u in units.values():
        rec[f"out_{u.name}"] = u.output_mw if u.running else 0.0
        rec[f"on_{u.name}"] = int(u.running)
        u.hours_in_state += 1

    return rec


# ===== 시뮬레이션 =====
def run_planner(
    backup_need: pd.Series,
    profile_path: Path,
    cfg: PolicyConfig,
    forecast_window: int = 12,
) -> tuple[pd.DataFrame, dict]:
    """backup_need 시계열 (MW, hourly) → 시뮬레이션 결과.

    forecast_mw = backup_need 본인의 lookahead (perfect-foresight PoC 가정).
    """
    units = load_profile(profile_path)
    initialize(units, cfg.initial_running)

    arr = backup_need.fillna(0).clip(lower=0).values.astype(float)
    records = []
    for t in range(len(arr)):
        forecast = arr[t + 1 : t + 1 + forecast_window].tolist()
        rec = step_decide(units, arr[t], forecast, cfg)
        rec["t"] = t
        records.append(rec)

    tl = pd.DataFrame(records)
    tl.index = backup_need.index

    # ----- metrics -----
    fuel_cost = tl["lng_total_mw"].sum() * 1000 * VAR_COST_KRW_PER_KWH  # MW*1h*1000 = kWh
    startup_cost = 0
    n_starts = 0
    n_peaker_starts = 0
    for sl in tl["starts"]:
        for name, cost, _stype in sl:
            startup_cost += cost
            n_starts += 1
            if units[name].mode == "peaker":
                n_peaker_starts += 1

    # parquet 호환을 위해 starts를 JSON 문자열로 변환
    import json
    tl["starts"] = tl["starts"].apply(json.dumps)

    n_under = int((tl["under_backup_mw"] > 1e-6).sum())
    total_under_mwh = float(tl["under_backup_mw"].sum())
    peaker_hours = int(tl["any_peaker_running"].sum())

    metrics = {
        "policy": cfg.name,
        "n_steps": len(tl),
        "fuel_cost_million_krw": round(fuel_cost / 1e6, 1),
        "startup_cost_million_krw": round(startup_cost / 1e6, 1),
        "total_cost_million_krw": round((fuel_cost + startup_cost) / 1e6, 1),
        "n_starts": n_starts,
        "n_peaker_starts": n_peaker_starts,
        "peaker_hours": peaker_hours,
        "peaker_pct_of_steps": round(100 * peaker_hours / len(tl), 1),
        "under_backup_count": n_under,
        "under_backup_freq_pct": round(100 * n_under / len(tl), 2),
        "under_backup_total_mwh": round(total_under_mwh, 1),
    }
    return tl, metrics


# ===== 정책 프리셋 =====
COST_OPTIMAL = PolicyConfig(
    name="cost_optimal",
    persist_threshold_h=4,
    safety_margin_mw=0.0,
    accept_under_backup=True,
)

RISK_AWARE = PolicyConfig(
    name="risk_aware",
    persist_threshold_h=2,
    safety_margin_mw=10.0,
    accept_under_backup=False,
)


# ===== Demo: scenarios_quantile.parquet → backup_need =====
def derive_backup_need_from_quantile(
    quantile_path: Path,
    use_q05_worst_case: bool = False,
) -> pd.Series:
    """
    backup_need(t) = max(0, q50_expected - actual_kwh) / 1000  → MW

    use_q05_worst_case=True → max(0, q50 - q05) — D-1 시점 'worst-case 부족 가능성'
    """
    df = pd.read_parquet(quantile_path).sort_values("datetime_kst").reset_index(drop=True)
    if use_q05_worst_case:
        deficit_kwh = (df["q50_expected"] - df["q05_worst"]).clip(lower=0)
    else:
        deficit_kwh = (df["q50_expected"] - df["actual_kwh"]).clip(lower=0)
    s = pd.Series((deficit_kwh / 1000.0).values, index=pd.to_datetime(df["datetime_kst"]))
    s.name = "backup_need_mw"
    return s


def main():
    profile_path = ROOT / "data" / "processed" / "thermal_unit_profile.csv"
    quantile_path = ROOT / "data" / "processed" / "scenarios_quantile.parquet"
    out_dir = ROOT / "pv" / "experiments" / "online_backup_planner"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3-tier 케이스 — 정책 차이가 (1) 비용 (2) startup (3) under-backup 모두에서 보이도록
    cases = {
        "realized":   dict(use_q05_worst_case=False,
                           initial=("CG6", "CG8", "CS2", "CS1"),    # base+CS1 (충분)
                           need_scale=1.0),
        "stress_q05": dict(use_q05_worst_case=True,
                           initial=("CG6", "CG8", "CS2"),            # baseload만
                           need_scale=1.0),
        "severe":     dict(use_q05_worst_case=True,
                           initial=("CG6", "CG8"),                   # 2 baseload만
                           need_scale=1.5),                          # need ×1.5
    }

    all_metrics = []
    for case_name, opts in cases.items():
        print(f"\n========== CASE: {case_name} ==========")
        backup_need = derive_backup_need_from_quantile(
            quantile_path, use_q05_worst_case=opts["use_q05_worst_case"]
        ) * opts.get("need_scale", 1.0)
        print(f"backup_need: mean {backup_need.mean():.1f} MW, "
              f"max {backup_need.max():.1f} MW, "
              f"positive% {100*(backup_need>0).mean():.1f}%, "
              f"p95 {np.percentile(backup_need, 95):.1f} MW")

        for cfg_base in (COST_OPTIMAL, RISK_AWARE):
            cfg = PolicyConfig(
                name=cfg_base.name,
                persist_threshold_h=cfg_base.persist_threshold_h,
                safety_margin_mw=cfg_base.safety_margin_mw,
                accept_under_backup=cfg_base.accept_under_backup,
                initial_running=tuple(opts["initial"]),
            )
            tl, met = run_planner(backup_need, profile_path, cfg)
            met = {"case": case_name, **met}
            all_metrics.append(met)
            tl.to_parquet(out_dir / f"timeline_{case_name}_{cfg.name}.parquet")
            print(f"  ✔ {cfg.name}: "
                  f"fuel {met['fuel_cost_million_krw']:.0f}M  "
                  f"start {met['n_starts']:>3} (peaker {met['n_peaker_starts']:>3})  "
                  f"peaker_hrs {met['peaker_hours']:>4}  "
                  f"under {met['under_backup_count']:>3} ({met['under_backup_freq_pct']:.1f}%, "
                  f"{met['under_backup_total_mwh']:.1f} MWh)")

    summary = pd.DataFrame(all_metrics).set_index(["case", "policy"])
    print("\n========== 전체 비교 ==========")
    print(summary.T.to_string())
    summary.to_csv(out_dir / "policy_comparison.csv", encoding="utf-8-sig")
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
