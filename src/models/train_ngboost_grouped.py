"""NGBoost: 단일 vs 3그룹 vs 사이트별 비교.

목적:
  사이트별 cloud-PV 관계 차이 → 그룹별/사이트별 학습이 더 정확한가?

전략:
  1. baseline (1 모델): 현재 strong NGBoost 결과 사용
  2. 3 group: 수상 / 해안 / 내륙
  3. per-site: 8 모델

Group 정의:
  수상  (1): 고흥만수상
  해안  (3): 광양항세방, 삼천포, 영흥
  내륙  (4): 경상대, 창원, 구미, 예천

비교 metric:
  - 사이트별 NMAE
  - Cloud bin별 NMAE
  - Portfolio NMAE (capacity-weighted sum)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.tree import DecisionTreeRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# Group 정의
GROUPS = {
    "수상": ["고흥만수상"],
    "해안": ["광양항세방", "삼천포", "영흥"],
    "내륙": ["경상대", "창원", "구미", "예천"],
}

BASE_FEATURES = [
    "dsr_mean", "zenith_center",
    "ta", "hm", "ws", "dc10Tca",
    "hour", "month",
]
CROSS_FEATURES = ["cloud_x_dsr_inv", "hm_x_dsr_inv", "rn"]
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")


def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    inv_dsr = 1.0 / (df["dsr_mean"] + 100.0)
    df["cloud_x_dsr_inv"] = df["dc10Tca"] * inv_dsr
    df["hm_x_dsr_inv"] = df["hm"] * inv_dsr
    return df


def train_one(df_train, df_val, features, label):
    print(f"\n[Train: {label}]  n_train={len(df_train):,}  n_val={len(df_val):,}")
    if len(df_train) < 1000 or len(df_val) < 100:
        print(f"  ! 데이터 부족, skip")
        return None
    base = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20)
    model = NGBRegressor(
        Dist=Normal, Base=base,
        n_estimators=400, learning_rate=0.05,
        minibatch_frac=0.5, col_sample=0.9,
        verbose=False, random_state=42,
    )
    Xtr = df_train[features].values.astype(float)
    ytr = df_train[TARGET].values
    Xv = df_val[features].values.astype(float)
    yv = df_val[TARGET].values
    model.fit(Xtr, ytr, X_val=Xv, Y_val=yv, early_stopping_rounds=30)
    return model


def predict_with(model, df_test, features):
    if model is None:
        return None, None
    X = df_test[features].values.astype(float)
    dist = model.pred_dist(X)
    mu = np.clip(np.asarray(dist.loc), 0, None)
    sig = np.asarray(dist.scale)
    return mu, sig


def evaluate_predictions(df, mu, sig, label, by_cloud=True):
    cap = df["site_capacity_kw"].values
    y = df[TARGET].values
    err = np.abs(y - mu)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu - 1.28 * sig) & (y <= mu + 1.28 * sig)).mean() * 100
    print(f"\n=== {label} ===")
    print(f"  rows: {len(df):,}, NMAE: {nmae:.2f}%, cov80: {cov80:.1f}%")

    # 사이트별
    df = df.copy()
    df["mu_pred"] = mu
    df["sig_pred"] = sig
    df["err"] = err
    print(f"  {'site':<14} {'cap MW':>7} {'NMAE %':>8} {'crash':>6}")
    for s in sorted(df["site"].unique()):
        sub = df[df["site"] == s]
        c = sub["site_capacity_kw"].iloc[0] / 1000
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        # site-level crash: actual_kwh < pred_kwh * 0.5
        a_kwh = sub[TARGET].values * sub["site_capacity_kw"].values
        p_kwh = sub["mu_pred"].values * sub["site_capacity_kw"].values
        crash = ((a_kwh < p_kwh * 0.5) & (p_kwh > 100)).sum()
        print(f"  {s:<14} {c:>7.1f} {n:>7.2f}% {crash:>6}")
    return nmae, cov80


def aggregate_portfolio(df, mu, sig):
    df = df.copy()
    df["pred_kwh"] = mu * df["site_capacity_kw"]
    df["actual_kwh"] = df[TARGET] * df["site_capacity_kw"]
    df["var_kwh2"] = (sig * df["site_capacity_kw"]) ** 2
    port = df.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        var_sum=("var_kwh2", "sum"), cap=("site_capacity_kw", "sum"),
    )
    port["sigma"] = np.sqrt(port["var_sum"])
    port["err"] = (port["pred"] - port["actual"]).abs()
    port["is_crash"] = (port["actual"] < port["pred"] * 0.5).astype(int)
    nmae = port["err"].sum() / port["cap"].sum() * 100
    crash = port["is_crash"].sum()
    cov80 = ((port["actual"] >= port["pred"] - 1.28 * port["sigma"])
             & (port["actual"] <= port["pred"] + 1.28 * port["sigma"])).mean() * 100
    return nmae, crash, cov80


# ========== Strategies ==========

def strategy_group(train, val, test):
    """3 group 학습."""
    print("\n" + "=" * 70)
    print("Strategy: 3 GROUP (수상 / 해안 / 내륙)")
    print("=" * 70)

    # Site one-hot 추가 (within-group differentiation)
    all_sites = sorted(train["site"].unique())
    for s in all_sites:
        for df_ in [train, val, test]:
            df_[f"site_{s}"] = (df_["site"] == s).astype(int)
    site_cols = [f"site_{s}" for s in all_sites]
    features = BASE_FEATURES + CROSS_FEATURES + site_cols

    mu_all = np.zeros(len(test))
    sig_all = np.zeros(len(test))
    test_rev = test.reset_index(drop=True)

    for gname, gsites in GROUPS.items():
        tr = train[train["site"].isin(gsites)]
        v = val[val["site"].isin(gsites)]
        te_idx = test_rev["site"].isin(gsites)
        te = test_rev[te_idx]
        model = train_one(tr, v, features, f"Group={gname} ({len(gsites)} sites)")
        mu, sig = predict_with(model, te, features)
        if mu is not None:
            mu_all[te_idx.values] = mu
            sig_all[te_idx.values] = sig

    return test_rev, mu_all, sig_all


def strategy_per_site(train, val, test):
    """사이트별 학습 (8 모델)."""
    print("\n" + "=" * 70)
    print("Strategy: PER-SITE (8 models)")
    print("=" * 70)
    # site one-hot 제외 (단일 사이트 모델이라 의미 없음)
    features = BASE_FEATURES + CROSS_FEATURES

    mu_all = np.zeros(len(test))
    sig_all = np.zeros(len(test))
    test_rev = test.reset_index(drop=True)

    for s in sorted(train["site"].unique()):
        tr = train[train["site"] == s]
        v = val[val["site"] == s]
        te_idx = test_rev["site"] == s
        te = test_rev[te_idx]
        model = train_one(tr, v, features, f"Site={s}")
        mu, sig = predict_with(model, te, features)
        if mu is not None:
            mu_all[te_idx.values] = mu
            sig_all[te_idx.values] = sig

    return test_rev, mu_all, sig_all


def main():
    print("=" * 70)
    print("NGBoost: 1 model vs 3-group vs per-site 비교")
    print("=" * 70)

    print("\n[1/3] 데이터 로딩...")
    df = load_data()
    train = df[df.datetime_kst < TRAIN_END].copy()
    val = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)].copy()
    test = df[df.datetime_kst >= VAL_END].copy()
    print(f"  train {len(train):,} / val {len(val):,} / test {len(test):,}")
    print(f"  사이트 수: train={train['site'].nunique()}, test={test['site'].nunique()}")

    # Group 학습
    test_g, mu_g, sig_g = strategy_group(train.copy(), val.copy(), test.copy())
    nmae_g, cov_g = evaluate_predictions(test_g, mu_g, sig_g, "Test (group strategy)")
    pnmae_g, pcrash_g, pcov_g = aggregate_portfolio(test_g, mu_g, sig_g)
    print(f"  Portfolio NMAE: {pnmae_g:.2f}% / crash {pcrash_g} / cov80 {pcov_g:.1f}%")

    # Per-site 학습
    test_p, mu_p, sig_p = strategy_per_site(train.copy(), val.copy(), test.copy())
    nmae_p, cov_p = evaluate_predictions(test_p, mu_p, sig_p, "Test (per-site strategy)")
    pnmae_p, pcrash_p, pcov_p = aggregate_portfolio(test_p, mu_p, sig_p)
    print(f"  Portfolio NMAE: {pnmae_p:.2f}% / crash {pcrash_p} / cov80 {pcov_p:.1f}%")

    # 최종 비교
    print("\n" + "=" * 70)
    print("FINAL — 비교 정리")
    print("=" * 70)
    print(f"{'strategy':<25} {'site NMAE':>10} {'port NMAE':>11} {'port crash':>11} {'cov80':>7}")
    print("-" * 70)
    print(f"{'1 model (strong baseline)':<25} {6.07:>9.2f}% {5.04:>10.2f}% {145:>11} {81.7:>6.1f}%   <-- 기준")
    print(f"{'3 group':<25} {nmae_g:>9.2f}% {pnmae_g:>10.2f}% {pcrash_g:>11} {pcov_g:>6.1f}%")
    print(f"{'per-site (8 models)':<25} {nmae_p:>9.2f}% {pnmae_p:>10.2f}% {pcrash_p:>11} {pcov_p:>6.1f}%")
    print("-" * 70)

    # Save
    out_dir = ROOT / "pv/experiments/ngboost_grouped"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "strategy": ["1_model_baseline", "3_group", "per_site"],
        "site_nmae": [6.07, nmae_g, nmae_p],
        "port_nmae": [5.04, pnmae_g, pnmae_p],
        "port_crash": [145, pcrash_g, pcrash_p],
        "cov80": [81.7, pcov_g, pcov_p],
    }).to_csv(out_dir / "comparison.csv", index=False)
    print(f"\n저장: {out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
