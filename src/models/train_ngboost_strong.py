"""NGBoost 강화 버전 — Cloud/Humidity-aware single model.

vs train_ngboost.py 차이:
  1. Cross features: cloud_x_dsr_inv, hm_x_dsr_inv (interaction)
  2. rn (강수) 추가 — 기존 feature에 빠진 신호
  3. Sample weight: cloud≥5 시간 2배 가중
  4. Tree depth 5 → 6
  5. n_estimators 500 → 800

목표:
  Cloud feature importance ↑ (4% → 10%+)
  Cloud 9-10 regime NMAE 개선
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# ========== Config ==========
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
    # rn: NaN = 비 안옴 = 0
    df["rn"] = df["rn"].fillna(0)

    # Cross features
    inv_dsr = 1.0 / (df["dsr_mean"] + 100.0)   # +100 to avoid /near-zero
    df["cloud_x_dsr_inv"] = df["dc10Tca"] * inv_dsr
    df["hm_x_dsr_inv"] = df["hm"] * inv_dsr

    df = pd.get_dummies(df, columns=["site"], prefix="site")
    return df


def split_temporal(df):
    train = df[df.datetime_kst < TRAIN_END]
    val = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test = df[df.datetime_kst >= VAL_END]
    return train, val, test


def feature_cols(df):
    site_cols = sorted(c for c in df.columns if c.startswith("site_") and c != "site_capacity_kw")
    return BASE_FEATURES + CROSS_FEATURES + site_cols


def main():
    print("=" * 70)
    print("NGBoost 강화 (Cloud/Humidity-aware single model)")
    print("=" * 70)

    print("\n[1/5] 데이터 로딩...")
    df = load_data()
    print(f"  rows: {len(df):,}")

    train, val, test = split_temporal(df)
    features = feature_cols(df)
    print(f"\n[2/5] split: train {len(train):,}, val {len(val):,}, test {len(test):,}")
    print(f"  features ({len(features)}): {features}")

    # Sample weight: 비활성 (이전 시도에서 역효과 확인됨)
    train_sw = np.ones(len(train))
    val_sw = np.ones(len(val))
    print(f"\n[3/5] Sample weight: 균등 (1.0) — cross feature만 효과 보기")

    Xtr = train[features].values.astype(float)
    ytr = train[TARGET].values
    Xv = val[features].values.astype(float)
    yv = val[TARGET].values

    print("\n[4/5] NGBoost 학습 (강화 hyperparam)...")
    # baseline 동일 hyperparam — cross feature 효과만 격리
    base = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20)
    model = NGBRegressor(
        Dist=Normal, Base=base,
        n_estimators=500, learning_rate=0.05,
        minibatch_frac=0.5, col_sample=0.9,
        verbose=True, verbose_eval=50,
        random_state=42,
    )
    model.fit(Xtr, ytr, X_val=Xv, Y_val=yv,
              sample_weight=train_sw, val_sample_weight=val_sw,
              early_stopping_rounds=40)

    print("\n[5/5] Test 평가...")
    Xte = test[features].values.astype(float)
    yte = test[TARGET].values
    cap_te = test["site_capacity_kw"].values

    pred_dist = model.pred_dist(Xte)
    mu = np.clip(np.asarray(pred_dist.loc), 0, None)
    sig = np.asarray(pred_dist.scale)

    abs_err = np.abs(yte - mu)
    nmae = (abs_err * cap_te).sum() / cap_te.sum()
    cov80 = ((yte >= mu - 1.28 * sig) & (yte <= mu + 1.28 * sig)).mean()
    cov95 = ((yte >= mu - 1.96 * sig) & (yte <= mu + 1.96 * sig)).mean()
    print(f"\n=== TEST 전체 (강화) ===")
    print(f"  MAE (cf):       {abs_err.mean():.4f}")
    print(f"  NMAE (capacity): {nmae*100:.2f}%   (baseline 6.20%)")
    print(f"  80% coverage:   {cov80*100:.1f}%   (baseline 80.6%)")
    print(f"  95% coverage:   {cov95*100:.1f}%   (baseline 92.9%)")

    # Feature importance
    print("\n=== Feature importance (mu, top 15) ===")
    fi = model.feature_importances_[0]
    order = np.argsort(fi)[::-1]
    for i in order[:15]:
        marker = " <-- new" if features[i] in CROSS_FEATURES else ""
        print(f"  {features[i]:<22} {fi[i]:.4f}{marker}")

    # Cloud bin별 portfolio NMAE
    print("\n=== Cloud bin별 Portfolio NMAE 비교 ===")
    test_out = test[["datetime_kst", "cf", "site_capacity_kw", "dc10Tca"]].copy()
    test_out["pred_cf"] = mu
    test_out["pred_std_cf"] = sig
    test_out["date"] = test_out["datetime_kst"].dt.date
    test_out["hour"] = test_out["datetime_kst"].dt.hour
    test_out["pred_kwh"] = test_out["pred_cf"] * test_out["site_capacity_kw"]
    test_out["actual_kwh"] = test_out["cf"] * test_out["site_capacity_kw"]
    cloud_h = test_out.groupby(["date", "hour"], as_index=False).agg(
        cloud=("dc10Tca", "mean"),
        pred=("pred_kwh", "sum"),
        actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"),
    )
    cloud_h["err"] = (cloud_h["pred"] - cloud_h["actual"]).abs()
    cloud_h["is_crash"] = (cloud_h["actual"] < cloud_h["pred"] * 0.5).astype(int)

    bins = [-1, 3, 5, 7, 9, 11]
    labels = ["0-3", "3-5", "5-7", "7-9", "9-10"]
    cloud_h["cloud_bin"] = pd.cut(cloud_h["cloud"], bins=bins, labels=labels)
    print(f"{'cloud':<10} {'n':>6} {'NMAE %':>10} {'crash':>7}")
    print("-" * 38)
    for lbl in labels:
        sub = cloud_h[cloud_h["cloud_bin"] == lbl]
        if len(sub) == 0:
            continue
        nmae_b = sub["err"].sum() / sub["cap"].sum() * 100
        print(f"{lbl:<10} {len(sub):>6} {nmae_b:>9.2f}% {sub['is_crash'].sum():>7}")
    overall = cloud_h["err"].sum() / cloud_h["cap"].sum() * 100
    print("-" * 38)
    print(f"{'전체':<10} {len(cloud_h):>6} {overall:>9.2f}% {cloud_h['is_crash'].sum():>7}")
    print()
    print("Baseline 비교 (이전 학습 결과):")
    print("  0-3:   4.56% / 23 crash")
    print("  3-5:   6.40% / 11")
    print("  5-7:   6.18% / 8")
    print("  7-9:   5.87% / 24")
    print("  9-10:  3.74% / 64")
    print("  전체:  5.18% / 130")

    # Save
    out_dir = ROOT / "pv/experiments/ngboost_strong"
    out_dir.mkdir(parents=True, exist_ok=True)
    site_cols = sorted(c for c in test.columns if c.startswith("site_") and c != "site_capacity_kw")
    test_out_save = test[["datetime_kst", "cf", "site_capacity_kw"]].copy()
    test_out_save["site"] = test[site_cols].idxmax(axis=1).str.replace("site_", "")
    test_out_save["pred_cf"] = mu
    test_out_save["pred_std_cf"] = sig
    test_out_save.to_parquet(out_dir / "test_predictions.parquet", index=False)
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}")

    import joblib
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump({
        "features": features,
        "site_one_hot_cols": site_cols,
        "cross_features": CROSS_FEATURES,
    }, out_dir / "feature_schema.joblib")
    print(f"저장: {out_dir / 'model.joblib'}, feature_schema.joblib")


if __name__ == "__main__":
    main()
