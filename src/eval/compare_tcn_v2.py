"""Compare Step1 baseline / Step3 TCN-base / Step4 TCN-trans ensembles.

Outputs:
  - Overall: site NMAE, port NMAE, Cov80, Cov95, NLL
  - Per-site (광양항세방 / 예천 / 고흥만수상)
  - Cloud bin 3-7 (partial cloud) NMAE
  - Top cloud-pass events (2025-03-23, 2025-04-26, 2025-05-04 정오~15시) portfolio error
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "Baseline (5)": ROOT / "pv/experiments/resmlp_adaln_v2_ensemble/ensemble_test.parquet",
    "TCN-base (3)": ROOT / "pv/experiments/resmlp_tcn_v2_base/ensemble_test.parquet",
    "TCN-trans(3)": ROOT / "pv/experiments/resmlp_tcn_v2_trans/ensemble_test.parquet",
}


def overall(df):
    cap = df["site_capacity_kw"].values
    y = df["cf"].values
    mu = df["mu_mean"].values
    sig = df["sigma_total"].values
    err = np.abs(y - mu)
    nmae = (err * cap).sum() / cap.sum() * 100
    bias = ((mu - y) * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu - 1.282 * sig) & (y <= mu + 1.282 * sig)).mean() * 100
    cov95 = ((y >= mu - 1.96 * sig) & (y <= mu + 1.96 * sig)).mean() * 100
    nll = (0.5 * np.log(2 * np.pi * sig ** 2) + (y - mu) ** 2 / (2 * sig ** 2)).mean()
    pdf = pd.DataFrame({"datetime_kst": df["datetime_kst"].values,
                        "pred_kwh": mu * cap, "actual_kwh": y * cap, "cap": cap,
                        "var_kwh2": (sig * cap) ** 2})
    port = pdf.groupby("datetime_kst", as_index=False).agg(
        p=("pred_kwh", "sum"), a=("actual_kwh", "sum"),
        v=("var_kwh2", "sum"), c=("cap", "sum"))
    pnmae = (port.p - port.a).abs().sum() / port.c.sum() * 100
    return {"nmae": nmae, "port": pnmae, "bias": bias,
            "cov80": cov80, "cov95": cov95, "nll": float(nll)}


def per_site(df, sites):
    out = {}
    for s in sites:
        d = df[df.site == s]
        cap = d["site_capacity_kw"].values
        y = d["cf"].values; mu = d["mu_mean"].values; sig = d["sigma_total"].values
        nmae = (np.abs(y - mu) * cap).sum() / cap.sum() * 100
        cov80 = ((y >= mu - 1.282 * sig) & (y <= mu + 1.282 * sig)).mean() * 100
        out[s] = {"nmae": nmae, "cov80": cov80}
    return out


def cloud_bin_partial(df, training_set):
    """Cloud bin 3-7 NMAE.
    dc10Tca (0~10 scale) bin 3-7 = partial cloud. Join via (datetime_kst, site).
    """
    key_cols = ["datetime_kst", "site"]
    cloud_df = training_set[key_cols + ["dc10Tca"]].copy()
    cloud_df["datetime_kst"] = pd.to_datetime(cloud_df["datetime_kst"])
    df2 = df.copy()
    df2["datetime_kst"] = pd.to_datetime(df2["datetime_kst"])
    m = df2.merge(cloud_df, on=key_cols, how="left")
    # bin = floor(dc10Tca)
    m["cbin"] = np.floor(m["dc10Tca"].fillna(-1)).astype(int)
    sub = m[(m.cbin >= 3) & (m.cbin <= 7)]
    cap = sub["site_capacity_kw"].values
    y = sub["cf"].values; mu = sub["mu_mean"].values
    nmae = (np.abs(y - mu) * cap).sum() / cap.sum() * 100
    return {"nmae": nmae, "n": len(sub),
            "by_bin": sub.assign(err=np.abs(y - mu) * cap)
                        .groupby("cbin").apply(lambda g: pd.Series({
                            "nmae": (g.err.values).sum() /
                                    g["site_capacity_kw"].sum() * 100,
                            "n": len(g)}), include_groups=False)}


def event_errors(df, dates_hours):
    """Per-event portfolio NMAE."""
    df2 = df.copy()
    df2["datetime_kst"] = pd.to_datetime(df2["datetime_kst"])
    out = {}
    for date_str, hours in dates_hours.items():
        date = pd.Timestamp(date_str)
        hr_start, hr_end = hours
        m = df2[(df2.datetime_kst.dt.date == date.date()) &
                (df2.datetime_kst.dt.hour >= hr_start) &
                (df2.datetime_kst.dt.hour <= hr_end)]
        if len(m) == 0:
            out[date_str] = None; continue
        cap = m["site_capacity_kw"].values
        y = m["cf"].values; mu = m["mu_mean"].values
        # portfolio: sum kWh per hour, then NMAE
        m2 = m.assign(pred_kwh=mu * cap, actual_kwh=y * cap)
        port = m2.groupby("datetime_kst", as_index=False).agg(
            p=("pred_kwh", "sum"), a=("actual_kwh", "sum"), c=("site_capacity_kw", "sum"))
        pnmae = (port.p - port.a).abs().sum() / port.c.sum() * 100
        out[date_str] = {"port_nmae": pnmae, "hours": len(port),
                          "site_nmae": (np.abs(y - mu) * cap).sum() / cap.sum() * 100}
    return out


def main():
    print("=" * 80)
    print("Compare: Step1 Baseline / Step3 TCN-base / Step4 TCN-trans")
    print("=" * 80)

    runs = {k: pd.read_parquet(v) for k, v in RUNS.items()}
    training_set = pd.read_parquet(ROOT / "data/processed/training_set.parquet")

    # ===== 1. Overall =====
    print("\n[1] Overall (test set, ensemble)")
    print(f"  {'Run':<14} {'NMAE':>7} {'Port':>7} {'bias':>7} {'Cov80':>7} {'Cov95':>7} {'NLL':>8}")
    rows_overall = {}
    for name, df in runs.items():
        m = overall(df)
        rows_overall[name] = m
        print(f"  {name:<14} {m['nmae']:>6.3f}% {m['port']:>6.3f}% "
              f"{m['bias']:>+6.3f}% {m['cov80']:>6.1f}% {m['cov95']:>6.1f}% {m['nll']:>8.4f}")

    base = rows_overall["Baseline (5)"]
    print("\n  Δ vs Baseline:")
    for name in ["TCN-base (3)", "TCN-trans(3)"]:
        m = rows_overall[name]
        print(f"  {name:<14} ΔNMAE {m['nmae']-base['nmae']:>+6.3f} | ΔPort {m['port']-base['port']:>+6.3f} | "
              f"ΔCov80 {m['cov80']-base['cov80']:>+6.2f} | ΔCov95 {m['cov95']-base['cov95']:>+6.2f}")

    # ===== 2. Per-site (target sites) =====
    print("\n[2] Per-site (광양항세방 / 예천 / 고흥만수상)")
    targets = ["광양항세방", "예천", "고흥만수상"]
    site_metrics = {name: per_site(df, targets) for name, df in runs.items()}
    print(f"  {'Site':<12} {'Metric':<6}", end="")
    for name in runs: print(f" {name:>14}", end="")
    print()
    for s in targets:
        for metric in ["nmae", "cov80"]:
            print(f"  {s:<12} {metric:<6}", end="")
            for name in runs:
                v = site_metrics[name][s][metric]
                print(f" {v:>13.2f}", end="")
            print()

    # ===== 3. Cloud bin 3-7 (partial cloud) =====
    print("\n[3] Cloud bin 3-7 (partial cloud) NMAE")
    print(f"  {'Run':<14} {'NMAE':>8} {'n':>8}")
    cloud_results = {}
    for name, df in runs.items():
        c = cloud_bin_partial(df, training_set)
        cloud_results[name] = c
        print(f"  {name:<14} {c['nmae']:>7.3f}% {c['n']:>8}")
    print("\n  By individual bin:")
    print(f"  {'bin':>3}", end="")
    for name in runs: print(f" {name:>13}", end="")
    print(f" {'n':>6}")
    base_by_bin = cloud_results["Baseline (5)"]["by_bin"]
    for cbin in sorted(base_by_bin.index):
        print(f"  {cbin:>3}", end="")
        for name in runs:
            try:
                v = cloud_results[name]["by_bin"].loc[cbin, "nmae"]
                print(f" {v:>12.3f}%", end="")
            except KeyError:
                print(f" {'N/A':>13}", end="")
        n = base_by_bin.loc[cbin, "n"]
        print(f" {int(n):>6}")

    # ===== 4. Top cloud-pass events =====
    print("\n[4] Top cloud-pass events (정오~15시 = 12~15h)")
    events = {"2025-03-23": (12, 15), "2025-04-26": (12, 15), "2025-05-04": (12, 15)}
    print(f"  {'Date':<12}", end="")
    for name in runs: print(f" {name:>14}", end="")
    print()
    print("  Port NMAE:")
    ev_results = {name: event_errors(df, events) for name, df in runs.items()}
    for d in events:
        print(f"  {d:<12}", end="")
        for name in runs:
            r = ev_results[name].get(d)
            if r is None: print(f" {'no data':>14}", end="")
            else: print(f" {r['port_nmae']:>13.2f}%", end="")
        print()

    # ===== 5. Decision summary =====
    print("\n" + "=" * 80)
    print("Decision summary")
    print("=" * 80)
    print("규칙: TCN-trans가 전체 NMAE + (partial cloud / top event / portfolio) 중 2+ 개에서 좋아지면 채택.")
    base_m = rows_overall["Baseline (5)"]
    trans_m = rows_overall["TCN-trans(3)"]
    base_cloud = cloud_results["Baseline (5)"]["nmae"]
    trans_cloud = cloud_results["TCN-trans(3)"]["nmae"]
    print(f"\n  전체 NMAE       : Baseline {base_m['nmae']:.3f}% vs Trans {trans_m['nmae']:.3f}% "
          f"→ {'Trans win' if trans_m['nmae'] < base_m['nmae'] else 'Baseline win'}")
    print(f"  Portfolio NMAE  : Baseline {base_m['port']:.3f}% vs Trans {trans_m['port']:.3f}% "
          f"→ {'Trans win' if trans_m['port'] < base_m['port'] else 'Baseline win'}")
    print(f"  Partial cloud   : Baseline {base_cloud:.3f}% vs Trans {trans_cloud:.3f}% "
          f"→ {'Trans win' if trans_cloud < base_cloud else 'Baseline win'}")
    base_evs = sum(ev_results["Baseline (5)"][d]["port_nmae"] for d in events
                    if ev_results["Baseline (5)"][d] is not None)
    trans_evs = sum(ev_results["TCN-trans(3)"][d]["port_nmae"] for d in events
                     if ev_results["TCN-trans(3)"][d] is not None)
    print(f"  Top events sum  : Baseline {base_evs:.2f}% vs Trans {trans_evs:.2f}% "
          f"→ {'Trans win' if trans_evs < base_evs else 'Baseline win'}")


if __name__ == "__main__":
    main()
