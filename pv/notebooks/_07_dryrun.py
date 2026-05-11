"""07 노트북 dry-run — OOM 없이 돌아가는지 확인 + 결과 미리보기."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGET_DATE = pd.Timestamp("2025-08-20").date()
NOISE = 3000.0

print("[1] reading trajectories (filtered)...")
# 전체 4M 행 로드 → 메모리 부담. pyarrow filter로 1일치만.
traj = pd.read_parquet(
    ROOT / "data/processed/scenarios_trajectories.parquet",
    filters=[("date", "=", TARGET_DATE)],
)
port = pd.read_parquet(ROOT / "data/processed/portfolio_predictions.parquet")
port = port[port["datetime_kst"].dt.date == TARGET_DATE].sort_values("datetime_kst")
print(f"  traj rows: {len(traj):,}, port rows: {len(port)}")

hours = sorted(traj["hour"].unique().tolist())
S = traj.pivot(index="trajectory_idx", columns="hour", values="sample_kwh").sort_index()
print(f"  S shape: {S.shape}")

actual = port.set_index(port["datetime_kst"].dt.hour)["actual_kwh"].reindex(hours)
mu = port.set_index(port["datetime_kst"].dt.hour)["mu_kwh"].reindex(hours)
sigma = port.set_index(port["datetime_kst"].dt.hour)["sigma_kwh"].reindex(hours)

print("\n[2] day summary:")
for h in hours:
    z = (actual[h] - mu[h]) / sigma[h]
    print(f"  {h:>2}시: mu={mu[h]/1000:>6.1f}, σ={sigma[h]/1000:>5.1f}, actual={actual[h]/1000:>6.1f}  z={z:+.2f}")

# Sequential reweight
print(f"\n[3] sequential reweight (noise={NOISE/1000} MWh)...")
n = len(S)
w = np.ones(n) / n
weights_history = {}
ess_history = {}
S_arr = S.values  # ndarray for speed
hour_to_col = {h: i for i, h in enumerate(hours)}
actual_arr = actual.values

for ti, t in enumerate(hours):
    X_t = actual_arr[ti]
    traj_t = S_arr[:, ti]
    log_lik = -0.5 * ((traj_t - X_t) / NOISE) ** 2
    log_lik -= log_lik.max()
    w_new = w * np.exp(log_lik)
    s = w_new.sum()
    if s == 0:
        print(f"  hour {t}: ESS=0 collapse → reset uniform")
        w_new = np.ones(n) / n
    else:
        w_new /= s
    w = w_new
    weights_history[t] = w.copy()
    ess_history[t] = 1.0 / (w**2).sum()
    print(f"  after hour {t:>2}: ESS = {ess_history[t]:>7.1f}")


def wstats(values, weights):
    m = (values * weights).sum()
    v = ((values - m) ** 2 * weights).sum()
    return m, np.sqrt(v)


def wquantile(values, weights, q):
    idx = np.argsort(values)
    return np.interp(q, np.cumsum(weights[idx]), values[idx])


# Evolution table
print("\n[4] evolution table (now=11시 snapshot)...")
uniform = np.ones(n) / n
w_post11 = weights_history[11]
print(f"{'future':>6} {'mean_A':>8} {'mean_B':>8} {'sig_A':>6} {'sig_B':>6} {'actual':>8} {'errA':>6} {'errB':>6}")
for fi, f in enumerate(hours):
    if f <= 11:
        continue
    v = S_arr[:, fi]
    mA, sA = wstats(v, uniform)
    mB, sB = wstats(v, w_post11)
    aA = actual_arr[fi]
    print(f"{f:>6} {mA/1000:>8.2f} {mB/1000:>8.2f} {sA/1000:>6.2f} {sB/1000:>6.2f} {aA/1000:>8.2f} {abs(mA-aA)/1000:>6.2f} {abs(mB-aA)/1000:>6.2f}")

print("\n[5] horizon summary:")
rows = []
for ti, now in enumerate(hours):
    w_post = weights_history[now]
    for fi, future in enumerate(hours):
        if future <= now:
            continue
        v = S_arr[:, fi]
        mA, sA = wstats(v, uniform)
        mB, sB = wstats(v, w_post)
        rows.append({
            "now": now, "future": future, "horizon": future - now,
            "mean_A": mA, "sigma_A": sA,
            "mean_B": mB, "sigma_B": sB,
            "actual": actual_arr[fi],
            "errA": abs(mA - actual_arr[fi]),
            "errB": abs(mB - actual_arr[fi]),
        })
ev = pd.DataFrame(rows)
by_hz = ev.groupby("horizon").agg(
    n=("horizon", "size"),
    mean_errA=("errA", lambda s: s.mean() / 1000),
    mean_errB=("errB", lambda s: s.mean() / 1000),
    sigA=("sigma_A", lambda s: s.mean() / 1000),
    sigB=("sigma_B", lambda s: s.mean() / 1000),
)
by_hz["err_red_%"] = (1 - by_hz["mean_errB"] / by_hz["mean_errA"]) * 100
by_hz["sig_red_%"] = (1 - by_hz["sigB"] / by_hz["sigA"]) * 100
print(by_hz.round(2).to_string())

print("\n[6] saving PNGs...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

OUT = ROOT / "pv/experiments/realtime_simulation"
OUT.mkdir(parents=True, exist_ok=True)


def wquantile(v, w, q):
    idx = np.argsort(v)
    return np.interp(q, np.cumsum(w[idx]), v[idx])


prior_means = np.array([wstats(S_arr[:, i], uniform)[0] for i in range(len(hours))]) / 1000
prior_q05 = np.array([wquantile(S_arr[:, i], uniform, 0.05) for i in range(len(hours))]) / 1000
prior_q95 = np.array([wquantile(S_arr[:, i], uniform, 0.95) for i in range(len(hours))]) / 1000

# Snapshot grid
snapshots = [9, 11, 13, 15]
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)
axes = axes.flatten()
for ax, now in zip(axes, snapshots):
    ti = hours.index(now)
    w_post = weights_history[now]
    pm = np.array([wstats(S_arr[:, i], w_post)[0] for i in range(len(hours))]) / 1000
    p05 = np.array([wquantile(S_arr[:, i], w_post, 0.05) for i in range(len(hours))]) / 1000
    p95 = np.array([wquantile(S_arr[:, i], w_post, 0.95) for i in range(len(hours))]) / 1000

    ax.fill_between(hours, prior_q05, prior_q95, alpha=0.15, color="gray", label="prior 90% (A)")
    ax.plot(hours, prior_means, color="gray", lw=1.5, ls="--", label="prior mean")
    ax.fill_between(hours, p05, p95, alpha=0.30, color="steelblue", label="posterior 90% (B)")
    ax.plot(hours, pm, color="steelblue", lw=2.5, label="posterior mean")
    ax.plot(hours, actual_arr / 1000, "ro-", markersize=7, lw=1.5, label="actual")
    ax.axvspan(hours[0] - 0.4, now + 0.4, alpha=0.08, color="green")
    ax.axvline(now + 0.5, color="green", lw=1.2, ls=":")
    ax.set_title(f"now = {now}h (obs {ti+1}회) · ESS = {ess_history[now]:.0f}")
    ax.set_xlabel("hour (KST)")
    ax.set_ylabel("PV (MWh)")
    ax.legend(loc="upper right", fontsize=8)
plt.suptitle(f"2025-03-23 outlier — Reweight snapshots (noise={NOISE/1000} MWh)", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "outlier_reweight_snapshots.png", bbox_inches="tight", dpi=110)
plt.close()
print(f"  {OUT / 'outlier_reweight_snapshots.png'}")

# Evolution plot
h1 = ev[ev["horizon"] == 1].sort_values("future")
h3 = ev[ev["horizon"] == 3].sort_values("future")
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
ax = axes[0]
ax.plot(h1["future"], h1["mean_A"] / 1000, "o--", color="gray", label="A: prior mean")
ax.plot(h1["future"], h1["mean_B"] / 1000, "s-", color="steelblue", label="B: posterior mean")
ax.plot(h1["future"], h1["actual"] / 1000, "r^-", label="actual", markersize=8)
ax.set_ylabel("PV (MWh)")
ax.set_title("1h-ahead: mean A vs B vs actual")
ax.legend()
ax = axes[1]
ax.plot(h1["future"], h1["sigma_A"] / 1000, "o--", color="gray", label="σ_A (h+1)")
ax.plot(h1["future"], h1["sigma_B"] / 1000, "s-", color="steelblue", label="σ_B (h+1)")
ax.plot(h3["future"], h3["sigma_B"] / 1000, "^-", color="orange", alpha=0.7, label="σ_B (h+3)")
ax.set_ylabel("σ (MWh)")
ax.set_title("Posterior σ — 누적 obs로 좁아짐")
ax.legend()
ax = axes[2]
ax.plot(h1["future"], h1["errA"] / 1000, "o--", color="gray", label="|errA| h+1")
ax.plot(h1["future"], h1["errB"] / 1000, "s-", color="steelblue", label="|errB| h+1")
ax.plot(h3["future"], h3["errB"] / 1000, "^-", color="orange", alpha=0.7, label="|errB| h+3")
ax.set_xlabel("future hour (KST)")
ax.set_ylabel("|예측 - 실측| (MWh)")
ax.set_title("예측-실측 괴리: A vs B")
ax.legend()
plt.suptitle("2025-03-23 — Reweight evolution", fontsize=12)
plt.tight_layout()
plt.savefig(OUT / "outlier_reweight_evolution.png", bbox_inches="tight", dpi=110)
plt.close()
print(f"  {OUT / 'outlier_reweight_evolution.png'}")

# ESS plot
fig, ax = plt.subplots(figsize=(10, 4))
ess_arr = [ess_history[t] for t in hours]
ax.plot(hours, ess_arr, "go-", lw=2, markersize=8)
ax.axhline(y=len(S), color="gray", ls="--", alpha=0.5, label=f"prior ESS = {len(S)}")
ax.set_yscale("log")
ax.set_xlabel("hour")
ax.set_ylabel("ESS (log)")
ax.set_title("2025-03-23 — ESS 추이")
for t, e in zip(hours, ess_arr):
    ax.annotate(f"{e:.0f}", (t, e), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "outlier_reweight_ess.png", bbox_inches="tight", dpi=110)
plt.close()
print(f"  {OUT / 'outlier_reweight_ess.png'}")

print("\n[OK] 연산 + plot 완료 (no OOM)")
