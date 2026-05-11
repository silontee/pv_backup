"""NGBoost Regime model — DSR 기반 conditional split (Mixture-of-Experts).

Regime 정의 (data-driven, 2D heatmap 결과):
  Normal:  dsr_mean >= 350  (맑음, n=2294 in 2025)
  Crash:   dsr_mean <  350  (흐림/비, recall 77%, crash rate 5.6%)

학습:
  M_normal = NGBoost(normal data, Normal dist) → 평일 (μ, σ)
  M_crash  = NGBoost(crash data, Normal dist)  → 흐림 (μ, σ)
  Gate     = GradientBoosting(features → P(regime=normal))

추론 (Mixture of Normals → Normal 근사 by mean/var matching):
  μ_mix  = w * μ_n + (1-w) * μ_c
  σ²_mix = w*(σ_n² + μ_n²) + (1-w)*(σ_c² + μ_c²) - μ_mix²
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.tree import DecisionTreeRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# ========== Config ==========
FEATURES = [
    "dsr_mean", "zenith_center",
    "ta", "hm", "ws", "dc10Tca",
    "hour", "month",
]
TARGET = "cf"
DSR_THRESHOLD = 350.0   # data-driven regime boundary

TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")


# ========== Data ==========

def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df = pd.get_dummies(df, columns=["site"], prefix="site")
    return df


def split_temporal(df):
    train = df[df.datetime_kst < TRAIN_END]
    val = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test = df[df.datetime_kst >= VAL_END]
    return train, val, test


def feature_cols(df):
    site_cols = sorted(c for c in df.columns if c.startswith("site_") and c != "site_capacity_kw")
    return FEATURES + site_cols


# ========== Regime label ==========

def is_crash_regime(df):
    return (df["dsr_mean"] < DSR_THRESHOLD).values


# ========== Models ==========

def train_ngboost(X_tr, y_tr, X_val, y_val, label):
    print(f"\n  [{label}] NGBoost 학습 (n_train={len(X_tr):,}, n_val={len(X_val):,})...")
    base = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20)
    model = NGBRegressor(
        Dist=Normal, Base=base,
        n_estimators=500, learning_rate=0.05,
        minibatch_frac=0.5, col_sample=0.9,
        verbose=True, verbose_eval=50,
        random_state=42,
    )
    model.fit(X_tr, y_tr, X_val=X_val, Y_val=y_val, early_stopping_rounds=30)
    return model


def train_gate(X_tr, y_regime_tr, X_val, y_regime_val):
    print(f"\n  [Gate] GradientBoostingClassifier 학습 "
          f"(crash 비율 train={y_regime_tr.mean()*100:.1f}%, val={y_regime_val.mean()*100:.1f}%)...")
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.9, random_state=42,
    )
    clf.fit(X_tr, y_regime_tr)
    p_val = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_regime_val, p_val)
    print(f"  Gate AUC (val): {auc:.4f}")
    return clf


def mixture_inference(X, model_normal, model_crash, gate):
    """Soft mixture of two Normals → Normal approx (mean/var matching)."""
    p_crash = gate.predict_proba(X)[:, 1]
    p_normal = 1.0 - p_crash

    dist_n = model_normal.pred_dist(X)
    dist_c = model_crash.pred_dist(X)
    mu_n = np.asarray(dist_n.loc)
    sig_n = np.asarray(dist_n.scale)
    mu_c = np.asarray(dist_c.loc)
    sig_c = np.asarray(dist_c.scale)

    mu_mix = p_normal * mu_n + p_crash * mu_c
    e_x2 = p_normal * (sig_n ** 2 + mu_n ** 2) + p_crash * (sig_c ** 2 + mu_c ** 2)
    var_mix = e_x2 - mu_mix ** 2
    var_mix = np.clip(var_mix, 1e-8, None)
    sig_mix = np.sqrt(var_mix)

    return mu_mix, sig_mix, p_crash, mu_n, sig_n, mu_c, sig_c


# ========== Evaluation ==========

def evaluate(y, mu, sigma, capacity, label):
    mu_clipped = np.clip(mu, 0, None)
    abs_err = np.abs(y - mu_clipped)
    nmae_cap = np.sum(abs_err * capacity) / np.sum(capacity)
    cov80 = ((y >= mu - 1.28 * sigma) & (y <= mu + 1.28 * sigma)).mean()
    cov95 = ((y >= mu - 1.96 * sigma) & (y <= mu + 1.96 * sigma)).mean()
    print(f"\n=== {label} ===")
    print(f"  MAE (cf):       {abs_err.mean():.4f}")
    print(f"  NMAE (capacity): {nmae_cap*100:.2f}%")
    print(f"  80% coverage:   {cov80*100:.1f}%")
    print(f"  95% coverage:   {cov95*100:.1f}%")
    return {"nmae_cap": nmae_cap, "cov80": cov80, "cov95": cov95, "mae": abs_err.mean()}


def evaluate_by_dsr_bin(test_df, mu, sigma, label):
    """DSR 구간별 NMAE 비교."""
    df = test_df.copy()
    df["mu_pred"] = np.clip(mu, 0, None)
    df["sigma_pred"] = sigma
    bins = [0, 150, 250, 350, 500, 2000]
    df["dsr_bin"] = pd.cut(df["dsr_mean"], bins=bins, include_lowest=True)
    print(f"\n=== {label} — DSR 구간별 NMAE (capacity 기준) ===")
    print(f"{'dsr_bin':<14} {'n':>6} {'NMAE %':>8} {'cov80%':>8}")
    for b, sub in df.groupby("dsr_bin", observed=True):
        if len(sub) == 0:
            continue
        cap = sub["site_capacity_kw"].values
        y = sub[TARGET].values
        mu_b = sub["mu_pred"].values
        sig_b = sub["sigma_pred"].values
        nmae = np.sum(np.abs(y - mu_b) * cap) / np.sum(cap) * 100
        cov80 = ((y >= mu_b - 1.28 * sig_b) & (y <= mu_b + 1.28 * sig_b)).mean() * 100
        print(f"{str(b):<14} {len(sub):>6,} {nmae:>7.2f}% {cov80:>7.1f}%")


# ========== Main ==========

def main():
    print("=" * 70)
    print("NGBoost Regime model (Mixture-of-Experts)")
    print(f"Regime split: dsr_mean < {DSR_THRESHOLD} → crash")
    print("=" * 70)

    print("\n[1/6] 데이터 로딩...")
    df = load_data()
    print(f"  daytime 행: {len(df):,}")

    train, val, test = split_temporal(df)
    features = feature_cols(df)
    print(f"\n[2/6] split: train {len(train):,}, val {len(val):,}, test {len(test):,}")
    print(f"  features ({len(features)}): {features}")

    # Regime split
    is_crash_tr = is_crash_regime(train)
    is_crash_val = is_crash_regime(val)
    print(f"\n[3/6] Regime 분포 (train): "
          f"crash={is_crash_tr.sum():,} ({is_crash_tr.mean()*100:.1f}%), "
          f"normal={(~is_crash_tr).sum():,}")

    Xtr = train[features].values.astype(float)
    ytr = train[TARGET].values
    Xv = val[features].values.astype(float)
    yv = val[TARGET].values

    # Train two NGBoost
    print("\n[4/6] Two NGBoost 학습 (regime별)...")
    M_normal = train_ngboost(
        Xtr[~is_crash_tr], ytr[~is_crash_tr],
        Xv[~is_crash_val], yv[~is_crash_val],
        label="Normal",
    )
    M_crash = train_ngboost(
        Xtr[is_crash_tr], ytr[is_crash_tr],
        Xv[is_crash_val], yv[is_crash_val],
        label="Crash",
    )

    # Gate
    print("\n[5/6] Gate classifier 학습...")
    gate = train_gate(Xtr, is_crash_tr, Xv, is_crash_val)

    # Test eval
    print("\n[6/6] Test 평가 (mixture inference)...")
    Xte = test[features].values.astype(float)
    yte = test[TARGET].values
    cap_te = test["site_capacity_kw"].values

    mu_mix, sig_mix, p_crash_te, mu_n, sig_n, mu_c, sig_c = mixture_inference(Xte, M_normal, M_crash, gate)
    res_mix = evaluate(yte, mu_mix, sig_mix, cap_te, "TEST 전체 (Mixture)")
    evaluate_by_dsr_bin(test, mu_mix, sig_mix, "Mixture")

    # Compare with hard regime (just pick one model based on DSR)
    print("\n--- 비교: Hard regime (dsr<350 → crash model 그대로) ---")
    is_crash_te = is_crash_regime(test)
    mu_hard = np.where(is_crash_te, mu_c, mu_n)
    sig_hard = np.where(is_crash_te, sig_c, sig_n)
    evaluate(yte, mu_hard, sig_hard, cap_te, "TEST 전체 (Hard regime)")

    # Save
    out_dir = ROOT / "pv/experiments/ngboost_regime"
    out_dir.mkdir(parents=True, exist_ok=True)
    site_cols = sorted(c for c in test.columns if c.startswith("site_") and c != "site_capacity_kw")
    test_out = test[["datetime_kst", "cf", "site_capacity_kw"]].copy()
    test_out["site"] = test[site_cols].idxmax(axis=1).str.replace("site_", "")
    test_out["pred_cf"] = np.clip(mu_mix, 0, None)
    test_out["pred_std_cf"] = sig_mix
    test_out["pred_cf_normal"] = np.clip(mu_n, 0, None)
    test_out["pred_cf_crash"] = np.clip(mu_c, 0, None)
    test_out["p_crash"] = p_crash_te
    test_out.to_parquet(out_dir / "test_predictions.parquet", index=False)
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}")

    import joblib
    joblib.dump(M_normal, out_dir / "model_normal.joblib")
    joblib.dump(M_crash, out_dir / "model_crash.joblib")
    joblib.dump(gate, out_dir / "gate.joblib")
    joblib.dump({
        "features": features,
        "site_one_hot_cols": site_cols,
        "dsr_threshold": DSR_THRESHOLD,
    }, out_dir / "feature_schema.joblib")
    print(f"저장: {out_dir / 'model_normal.joblib'}, model_crash, gate, feature_schema")


if __name__ == "__main__":
    main()
