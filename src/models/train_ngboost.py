"""NGBoost 학습 — Track A (CPU baseline, parametric distribution).

목표:
  - PV cf 예측 + 불확실성(Normal: mean, std)
  - LNG 시나리오 입력으로 사용 가능
  - LightGBM/TFT 비교 anchor

설정:
  - Train: 2022-2023 (2년)
  - Val:   2024 (1년, early stopping)
  - Test:  2025 (1년, 최종 평가)
  - Daytime only (dsr_mean.notna()) — 야간은 0으로 후처리
  - Distribution: Normal (post-clip ≥0)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# ========== Config ==========
FEATURES = [
    "dsr_mean",        # GHI ★
    "zenith_center",   # 천문
    "ta", "hm", "ws",  # ASOS 기상
    "dc10Tca",         # 운량
    "hour", "month",   # 시간 (raw)
]
TARGET = "cf"

TRAIN_END = pd.Timestamp("2024-01-01")  # < this is train
VAL_END = pd.Timestamp("2025-01-01")    # < this is val
# >= VAL_END is test


# ========== Data ==========

def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "training_set.parquet")
    # Daytime only
    df = df[df.dsr_mean.notna() & df.cf.notna()].copy()
    # ASOS 결측 imputation (median per site)
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    # Site one-hot (NGBoost는 categorical 직접 처리 X)
    df = pd.get_dummies(df, columns=["site"], prefix="site")
    return df


def split(df: pd.DataFrame):
    train = df[df.datetime_kst < TRAIN_END]
    val = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test = df[df.datetime_kst >= VAL_END]
    return train, val, test


def feature_cols(df: pd.DataFrame) -> list[str]:
    site_cols = sorted(c for c in df.columns if c.startswith("site_") and c != "site_capacity_kw")
    return FEATURES + site_cols


# ========== Model ==========

def train_model(train: pd.DataFrame, val: pd.DataFrame, features: list[str]):
    Xtr = train[features].values.astype(float)
    ytr = train[TARGET].values
    Xv = val[features].values.astype(float)
    yv = val[TARGET].values

    base = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20)
    model = NGBRegressor(
        Dist=Normal,
        Base=base,
        n_estimators=500,
        learning_rate=0.05,
        minibatch_frac=0.5,
        col_sample=0.9,
        verbose=True,
        verbose_eval=20,
        random_state=42,
    )
    model.fit(Xtr, ytr, X_val=Xv, Y_val=yv, early_stopping_rounds=30)
    return model


# ========== Evaluation ==========

def evaluate(model, X, y, capacity_arr, label="set"):
    pred_dist = model.pred_dist(X)
    mu = np.asarray(pred_dist.loc)
    sigma = np.asarray(pred_dist.scale)
    mu_clipped = np.clip(mu, 0, None)

    mae = mean_absolute_error(y, mu_clipped)
    rmse = np.sqrt(mean_squared_error(y, mu_clipped))

    # NMAE: MAE normalized by mean capacity factor (since target is cf)
    # 그리고 절대값 MAE도 (kWh) 계산하기 위해 capacity 곱
    abs_err = np.abs(y - mu_clipped)
    abs_err_kwh = abs_err * capacity_arr
    nmae_cap = np.sum(abs_err_kwh) / np.sum(capacity_arr)  # NMAE w.r.t. capacity

    # Coverage of 80% interval (z=1.28)
    lower = mu - 1.28 * sigma
    upper = mu + 1.28 * sigma
    cov80 = ((y >= lower) & (y <= upper)).mean()

    # Coverage of 95%
    lower95 = mu - 1.96 * sigma
    upper95 = mu + 1.96 * sigma
    cov95 = ((y >= lower95) & (y <= upper95)).mean()

    print(f"\n=== {label} 성능 ===")
    print(f"  MAE (cf):            {mae:.4f}")
    print(f"  RMSE (cf):           {rmse:.4f}")
    print(f"  NMAE (capacity 기준): {nmae_cap*100:.2f}%   ← 목표 ≤ 6%")
    print(f"  80% 구간 coverage:    {cov80*100:.1f}%   (목표 80%)")
    print(f"  95% 구간 coverage:    {cov95*100:.1f}%   (목표 95%)")

    return {
        "mae": mae, "rmse": rmse, "nmae_cap": nmae_cap,
        "cov80": cov80, "cov95": cov95,
        "mu": mu_clipped, "sigma": sigma,
    }


def evaluate_per_site(test: pd.DataFrame, features: list[str], model):
    site_cols = sorted(c for c in test.columns if c.startswith("site_") and c != "site_capacity_kw")
    site_names = [c.replace("site_", "") for c in site_cols]
    test = test.copy()
    test["site"] = test[site_cols].idxmax(axis=1).str.replace("site_", "")

    print("\n=== 사이트별 NMAE (capacity 기준) ===")
    print(f"{'site':<12} {'rows':>7} {'NMAE %':>8} {'cov80 %':>9}")
    for s in sorted(site_names):
        sub = test[test.site == s]
        if len(sub) == 0:
            continue
        X = sub[features].values.astype(float)
        y = sub[TARGET].values
        cap = sub["site_capacity_kw"].values
        r = evaluate(model, X, y, cap, label="(suppressed)") if False else None
        pred_dist = model.pred_dist(X)
        mu = np.clip(np.asarray(pred_dist.loc), 0, None)
        sigma = np.asarray(pred_dist.scale)
        nmae_kwh = np.sum(np.abs(y - mu) * cap) / np.sum(cap)
        cov80 = ((y >= mu - 1.28 * sigma) & (y <= mu + 1.28 * sigma)).mean()
        print(f"{s:<12} {len(sub):>7,} {nmae_kwh*100:>7.2f}% {cov80*100:>8.1f}%")


def evaluate_portfolio(test: pd.DataFrame, features: list[str], model):
    """포트폴리오 합산 (사이트별 예측 → kWh로 환산 → 시간별 sum)."""
    site_cols = sorted(c for c in test.columns if c.startswith("site_") and c != "site_capacity_kw")
    test = test.copy()
    test["site"] = test[site_cols].idxmax(axis=1).str.replace("site_", "")

    X = test[features].values.astype(float)
    pred_dist = model.pred_dist(X)
    test["pred_cf"] = np.clip(np.asarray(pred_dist.loc), 0, None)
    test["pred_std_cf"] = np.asarray(pred_dist.scale)
    test["pred_kwh"] = test["pred_cf"] * test["site_capacity_kw"]
    test["actual_kwh"] = test["cf"] * test["site_capacity_kw"]

    portfolio = test.groupby("datetime_kst", as_index=False).agg(
        actual_kwh=("actual_kwh", "sum"),
        pred_kwh=("pred_kwh", "sum"),
        total_capacity_kw=("site_capacity_kw", "sum"),
    )
    abs_err = np.abs(portfolio.actual_kwh - portfolio.pred_kwh)
    nmae_port = abs_err.sum() / portfolio.total_capacity_kw.sum()

    print(f"\n=== 포트폴리오 성능 (시간별 합산) ===")
    print(f"  시간 수: {len(portfolio):,}")
    print(f"  총 actual: {portfolio.actual_kwh.sum()/1e6:.2f} GWh")
    print(f"  총 pred:   {portfolio.pred_kwh.sum()/1e6:.2f} GWh")
    print(f"  포트폴리오 NMAE: {nmae_port*100:.2f}%   ← 본 PoC 목표 ≤ 6%")


# ========== Main ==========

def main():
    print("[1/5] 데이터 로딩...")
    df = load_data()
    print(f"  daytime 행: {len(df):,}")

    print("\n[2/5] split...")
    train, val, test = split(df)
    print(f"  Train: {len(train):,} ({train.datetime_kst.min()} ~ {train.datetime_kst.max()})")
    print(f"  Val:   {len(val):,} ({val.datetime_kst.min()} ~ {val.datetime_kst.max()})")
    print(f"  Test:  {len(test):,} ({test.datetime_kst.min()} ~ {test.datetime_kst.max()})")

    features = feature_cols(df)
    print(f"\n[3/5] features ({len(features)}): {features}")

    print(f"\n[4/5] NGBoost 학습 시작...")
    model = train_model(train, val, features)

    print(f"\n[5/5] Test 평가...")
    Xte = test[features].values.astype(float)
    yte = test[TARGET].values
    cap_te = test["site_capacity_kw"].values
    evaluate(model, Xte, yte, cap_te, label="TEST 전체")
    evaluate_per_site(test, features, model)
    evaluate_portfolio(test, features, model)

    # Save predictions for later analysis
    out_dir = ROOT / "pv" / "experiments" / "ngboost_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dist = model.pred_dist(Xte)
    site_cols = sorted(c for c in test.columns if c.startswith("site_") and c != "site_capacity_kw")
    test_out = test[["datetime_kst", "cf", "site_capacity_kw"]].copy()
    test_out["site"] = test[site_cols].idxmax(axis=1).str.replace("site_", "")
    test_out["pred_cf"] = np.clip(np.asarray(pred_dist.loc), 0, None)
    test_out["pred_std_cf"] = np.asarray(pred_dist.scale)
    test_out.to_parquet(out_dir / "test_predictions.parquet", index=False)
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}")

    # 모델 + feature schema 저장 (forecast 추론용)
    import joblib
    model_path = out_dir / "model.joblib"
    schema_path = out_dir / "feature_schema.joblib"
    joblib.dump(model, model_path)
    schema = {
        "features": features,
        "site_one_hot_cols": site_cols,
        "feature_order": features,
    }
    joblib.dump(schema, schema_path)
    print(f"저장: {model_path}")
    print(f"저장: {schema_path}")


if __name__ == "__main__":
    main()
