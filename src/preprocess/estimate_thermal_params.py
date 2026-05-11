"""분당 LNG 운전 파라미터 역추정.

KOEN이 화력 운전 파라미터 미제공 (problem.md §0).
→ data/thermal_hourly/ 4년 운영 history에서 다음 추정:

  Pmin, Pmax           : 5%/95% percentile of running hours
  Ramp rate            : max |Δoutput| per hour
  Min up time          : 연속 가동 시간 분포 (median, p25)
  Min down time        : 연속 정지 시간 분포 (제외: 장기 정비)
  Cold start frequency : 24h+ 정지 후 재가동 빈도

출력:
  data/processed/thermal_params.csv  (호기별 + plant 합산)
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
SRC_DIR = ROOT / "data" / "thermal_hourly"
OUT_DIR = ROOT / "data" / "processed"


# ========== Data Loading ==========

# 물리적 호기당 capacity 상한 (분당 LNG, 일반 spec 기준)
# 가스터빈: 110 MW spec, 약간 여유 두고 130 MW
# 증기터빈: ~190 MW spec, 200 MW
PHYSICAL_CAP_MW = {
    "CG1": 130, "CG2": 130, "CG3": 130, "CG4": 130,
    "CG5": 130, "CG6": 130, "CG7": 130, "CG8": 130,
    "CS1": 200, "CS2": 200,
}


def load_thermal(clean_outliers: bool = True) -> pd.DataFrame:
    """월별 wide CSV → 호기별 시간별 long format.

    clean_outliers=True 면 호기 capacity × 1.1 초과 시 NaN으로 처리.
    """
    files = sorted(SRC_DIR.glob("thermal_hourly_*.csv"))
    dfs = []
    for f in files:
        d = pd.read_csv(f, index_col=False)
        d.columns = [c.strip() for c in d.columns]
        d["호기"] = d["호기"].astype(str).str.strip()
        d["일자_str"] = d["일자"].astype(str).str.strip()
        d = d[d["일자_str"].str.match(r"^\d{4}-\d{2}-\d{2}")].copy()
        d["일자"] = pd.to_datetime(d["일자_str"])

        hour_cols = [f"{h}시 발전량(MWh)" for h in range(1, 25)]
        long = d.melt(
            id_vars=["호기", "일자"],
            value_vars=hour_cols,
            var_name="hl",
            value_name="gen_kwh",   # 헤더가 MWh이지만 실제 kWh
        )
        long["hour"] = long["hl"].str.extract(r"(\d+)시").astype(int)
        long["datetime_kst"] = long["일자"] + pd.to_timedelta(long["hour"], unit="h")
        long["gen_kwh"] = pd.to_numeric(long["gen_kwh"], errors="coerce")
        long["output_mw"] = long["gen_kwh"] / 1000.0
        dfs.append(long[["호기", "datetime_kst", "output_mw"]])

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["호기", "datetime_kst"]).reset_index(drop=True)

    # 음수 → 0 (간헐 보고 오류)
    df.loc[df.output_mw < 0, "output_mw"] = 0

    if clean_outliers:
        # 호기 capacity 초과 → NaN (물리적 불가 spike, 데이터 오류 추정)
        before = len(df)
        outliers = []
        for unit, cap in PHYSICAL_CAP_MW.items():
            mask = (df.호기 == unit) & (df.output_mw > cap)
            n_out = mask.sum()
            if n_out > 0:
                outliers.append((unit, n_out, df.loc[mask, "output_mw"].max()))
                df.loc[mask, "output_mw"] = np.nan
        if outliers:
            print(f"  [클리닝] outlier 처리 (호기 capacity 초과):")
            for unit, n, max_v in outliers:
                print(f"    {unit}: {n}개 행, max 관측 {max_v:.1f} MW (cap {PHYSICAL_CAP_MW[unit]} MW)")
        # NaN을 forward fill or 0? — outlier는 단발성 spike니까 NaN 유지하고 stats에서 제외
    return df


# ========== Parameter Estimation ==========

def estimate_per_unit(df_unit: pd.DataFrame, threshold_mw: float = 1.0) -> dict:
    """단일 호기 운전 파라미터 추정.

    threshold_mw: 이 미만은 '정지'로 간주 (sub-MW noise floor)
    NaN은 outlier 클리닝된 시간 → 분석에서 제외 (정지로 간주 X)
    """
    df = df_unit.sort_values("datetime_kst").reset_index(drop=True)
    # NaN 행은 분석 제외 (outlier 클리닝됨)
    valid = df.output_mw.notna()
    output = df.loc[valid, "output_mw"].values

    is_running = output >= threshold_mw

    # Pmin / Pmax (가동 중 시간만)
    running_output = output[is_running]
    if len(running_output) == 0:
        return None
    pmin = float(np.percentile(running_output, 5))
    pmax = float(np.percentile(running_output, 95))
    pmax_observed = float(running_output.max())

    # Ramp rate: 연속 가동 시간 사이의 |Δoutput| max
    diff = np.abs(np.diff(output))
    # 양쪽 시간 모두 가동 중인 transition만
    both_running = is_running[:-1] & is_running[1:]
    ramp_max = float(diff[both_running].max()) if both_running.any() else 0.0
    ramp_p95 = float(np.percentile(diff[both_running], 95)) if both_running.any() else 0.0

    # Run lengths (연속 가동) / Off lengths (연속 정지)
    run_lengths = []
    off_lengths = []
    cur = is_running[0]
    cnt = 1
    for v in is_running[1:]:
        if v == cur:
            cnt += 1
        else:
            (run_lengths if cur else off_lengths).append(cnt)
            cur = v
            cnt = 1
    (run_lengths if cur else off_lengths).append(cnt)

    run_lengths = np.array(run_lengths)
    off_lengths = np.array(off_lengths)

    # Min up/down time: 짧은 transition만 보면 됨 (장기 정비 outlier 배제)
    min_up_p25 = float(np.percentile(run_lengths, 25)) if len(run_lengths) else 0
    min_down_p25 = float(np.percentile(off_lengths, 25)) if len(off_lengths) else 0
    min_up_median = float(np.median(run_lengths)) if len(run_lengths) else 0
    min_down_median = float(np.median(off_lengths)) if len(off_lengths) else 0

    # Cold start: 24h+ 정지 후 재가동 빈도
    cold_starts = int((off_lengths >= 24).sum())
    warm_starts = int(((off_lengths >= 2) & (off_lengths < 24)).sum())

    # Capacity factor (전체 평균 / Pmax)
    avg_output = float(output.mean())
    cf_overall = avg_output / pmax if pmax > 0 else 0

    return {
        "pmax_p95": round(pmax, 2),
        "pmax_observed": round(pmax_observed, 2),
        "pmin_p05": round(pmin, 2),
        "ramp_max_mw_per_h": round(ramp_max, 2),
        "ramp_p95_mw_per_h": round(ramp_p95, 2),
        "min_up_hrs_p25": round(min_up_p25, 1),
        "min_up_hrs_median": round(min_up_median, 1),
        "min_down_hrs_p25": round(min_down_p25, 1),
        "min_down_hrs_median": round(min_down_median, 1),
        "n_cold_starts (24h+ off)": cold_starts,
        "n_warm_starts (2-24h off)": warm_starts,
        "n_run_segments": len(run_lengths),
        "avg_output_mw": round(avg_output, 2),
        "cf_overall": round(cf_overall, 3),
        "running_hours": int(is_running.sum()),
        "total_hours": int(len(output)),
        "running_pct": round(is_running.mean() * 100, 1),
    }


def estimate_plant_total(df: pd.DataFrame, threshold_mw: float = 5.0) -> dict:
    """분당 plant 전체 (모든 호기 합산) 운전 파라미터."""
    plant = (
        df.groupby("datetime_kst", as_index=False)["output_mw"].sum()
          .sort_values("datetime_kst")
          .reset_index(drop=True)
    )
    return estimate_per_unit(plant.rename(columns={"output_mw": "output_mw"}), threshold_mw=threshold_mw)


# ========== Main ==========

def main():
    print("[1/3] thermal_hourly 4년 데이터 로딩...")
    df = load_thermal()
    print(f"  rows: {len(df):,}, 호기: {df.호기.nunique()}, 기간: {df.datetime_kst.min()} ~ {df.datetime_kst.max()}")

    print("\n[2/3] 호기별 파라미터 추정...")
    rows = []
    for unit in sorted(df.호기.unique()):
        sub = df[df.호기 == unit]
        params = estimate_per_unit(sub, threshold_mw=1.0)
        if params is None:
            print(f"  {unit}: 가동 기록 없음 (skip)")
            continue
        params["호기"] = unit
        rows.append(params)
        print(f"  {unit}: Pmax={params['pmax_p95']:.0f}MW, Pmin={params['pmin_p05']:.0f}MW, ramp_p95={params['ramp_p95_mw_per_h']:.0f}MW/h, run%={params['running_pct']:.0f}%")

    print("\n[3/3] Plant 전체 합산 파라미터...")
    plant_params = estimate_plant_total(df, threshold_mw=5.0)
    plant_params["호기"] = "분당_PLANT_TOTAL"
    rows.append(plant_params)
    print(f"  PLANT: Pmax={plant_params['pmax_p95']:.0f}MW, Pmin={plant_params['pmin_p05']:.0f}MW, ramp_p95={plant_params['ramp_p95_mw_per_h']:.0f}MW/h, run%={plant_params['running_pct']:.0f}%")

    # Save
    OUT_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    cols = ["호기"] + [c for c in out.columns if c != "호기"]
    out = out[cols]
    out_path = OUT_DIR / "thermal_params.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_path}")
    print(f"\n{out.to_string(index=False)}")

    # 비교 — CLAUDE.md 명시 추정값과
    print("\n=== CLAUDE.md 추정값 vs 측정값 ===")
    print("LNG (분당) Pmin: 추정 221MW (48%) — 우리 측정 PLANT Pmin = {:.0f}MW".format(plant_params["pmin_p05"]))
    print("LNG (분당) Pmax: 920MW — 우리 측정 PLANT Pmax = {:.0f}MW".format(plant_params["pmax_p95"]))
    print("LNG ramp rate: ~30 MW/min (= 1800MW/h) — 우리 측정 PLANT ramp_p95 = {:.0f}MW/h".format(plant_params["ramp_p95_mw_per_h"]))


if __name__ == "__main__":
    main()
