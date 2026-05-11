"""TFT 학습 — Track B (GPU, quantile head).

목표:
  - Multi-step direct forecast (24h, AR 누적 없음)
  - 9 quantile 분포 출력 (시나리오 sampling 가능)
  - Attention 기반 시간 의존성 학습
  - NGBoost baseline과 비교

설정:
  - Train: 2022-2023 (2년)
  - Val:   2024 (1년, early stopping)
  - Test:  2025 (1년, 최종 평가)
  - Encoder: 7일(168h) 과거
  - Prediction: 24h (1일 forecast = D-1 use case)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# Config (OOM 방지 — 작은 모델, gradient accumulation으로 effective batch 유지)
ENCODER_LENGTH = 24       # 1일 (이전 72에서 축소)
PREDICTION_LENGTH = 24    # 1일
BATCH_SIZE = 16           # 이전 128에서 1/8
ACCUMULATE_GRAD = 8       # effective batch = 128
MAX_EPOCHS = 10
LR = 0.003
HIDDEN_SIZE = 8           # 이전 16에서 1/2
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / "training_set.parquet")
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)

    # NaN 처리: 야간 cf=0, dsr=0
    df["cf"] = df["cf"].fillna(0)
    df["dsr_mean"] = df["dsr_mean"].fillna(0)
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df = df.dropna(subset=["ta", "hm", "ws", "dc10Tca"])  # 사이트 전체 NaN인 경우 drop

    # time_idx: 시간 단위 정수 (전 사이트 공통 grid)
    min_dt = df.datetime_kst.min()
    df["time_idx"] = ((df.datetime_kst - min_dt).dt.total_seconds() / 3600).astype(int)

    # daytime mask (평가용 보존)
    df["is_daytime"] = (df["zenith_center"] < 90).astype(int)
    return df


def main():
    print("=" * 60)
    print("TFT 학습 (Track B)")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    print("\n[1/5] Loading data...")
    df = load_data()
    print(f"  rows={len(df):,}, sites={df.site.nunique()}")

    train_cutoff = int((pd.Timestamp("2024-01-01") - df.datetime_kst.min()).total_seconds() / 3600)
    val_cutoff = int((pd.Timestamp("2025-01-01") - df.datetime_kst.min()).total_seconds() / 3600)
    test_cutoff = int((pd.Timestamp("2026-01-01") - df.datetime_kst.min()).total_seconds() / 3600)
    print(f"  cutoffs (time_idx): train<{train_cutoff}, val<{val_cutoff}, test<{test_cutoff}")

    print("\n[2/5] Building TimeSeriesDataSet...")
    training = TimeSeriesDataSet(
        df[df.time_idx < train_cutoff],
        time_idx="time_idx",
        target="cf",
        group_ids=["site"],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=["site"],
        time_varying_known_reals=[
            "dsr_mean", "zenith_center",
            "ta", "hm", "ws", "dc10Tca",
            "hour", "month",
        ],
        time_varying_unknown_reals=["cf"],
        target_normalizer=GroupNormalizer(groups=["site"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training, df[df.time_idx < val_cutoff],
        stop_randomization=True,
    )
    test = TimeSeriesDataSet.from_dataset(
        training, df[df.time_idx < test_cutoff],
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = test.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

    print(f"  train batches: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

    print("\n[3/5] Building TFT...")
    pl.seed_everything(42)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LR,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(quantiles=QUANTILES),
        log_interval=20,
        reduce_on_plateau_patience=3,
    )
    print(f"  params: {sum(p.numel() for p in tft.parameters())/1e3:.1f}k")

    print("\n[4/5] Training...")
    log_dir = ROOT / "pv" / "experiments" / "tft_baseline"
    log_dir.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=True)
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        callbacks=[early_stop],
        logger=False,
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        accumulate_grad_batches=ACCUMULATE_GRAD,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"  Best val_loss: {early_stop.best_score:.4f}" if early_stop.best_score else "")

    print("\n[5/5] Test 평가...")
    # 예측 + index
    raw_predictions = tft.predict(
        test_loader,
        mode="raw",
        return_index=True,
        return_x=False,
        trainer_kwargs=dict(accelerator="gpu", precision="bf16-mixed"),
    )
    # raw.output["prediction"] : [n_batch_total, prediction_length, n_quantiles]
    # raw.index : DataFrame with time_idx, site (decoder start)

    pred = raw_predictions.output["prediction"].cpu().numpy()
    idx = raw_predictions.index
    print(f"  predictions shape: {pred.shape}, index rows: {len(idx)}")

    # 각 예측 row를 펼쳐서 (site, time_idx, quantile) 매트릭스
    rows = []
    for i, row in idx.iterrows():
        site, t0 = row["site"], int(row["time_idx"])
        for h in range(PREDICTION_LENGTH):
            r = {"site": site, "time_idx": t0 + h}
            for j, q in enumerate(QUANTILES):
                r[f"q{int(q*100):02d}"] = float(pred[i, h, j])
            rows.append(r)
    pred_df = pd.DataFrame(rows)

    # actual cf와 join
    eval_df = df[["site", "time_idx", "datetime_kst", "cf", "site_capacity_kw", "is_daytime", "zenith_center"]].copy()
    merged = pred_df.merge(eval_df, on=["site", "time_idx"], how="inner")
    # 2025 test만
    merged = merged[merged.datetime_kst >= pd.Timestamp("2025-01-01")]
    # daytime만 평가 (NGBoost와 fair 비교)
    merged_day = merged[merged.is_daytime == 1].copy()
    print(f"  test rows (daytime): {len(merged_day):,}")

    # NMAE
    merged_day["pred_med"] = merged_day["q50"].clip(lower=0)
    merged_day["abs_err"] = np.abs(merged_day["cf"] - merged_day["pred_med"])
    merged_day["abs_err_kwh"] = merged_day["abs_err"] * merged_day["site_capacity_kw"]
    nmae_cap = merged_day["abs_err_kwh"].sum() / merged_day["site_capacity_kw"].sum()

    # Coverage
    in_80 = ((merged_day["cf"] >= merged_day["q10"]) & (merged_day["cf"] <= merged_day["q90"])).mean()
    in_95 = ((merged_day["cf"] >= merged_day["q05"]) & (merged_day["cf"] <= merged_day["q95"])).mean()

    print(f"\n=== TEST 전체 (daytime) ===")
    print(f"  rows: {len(merged_day):,}")
    print(f"  MAE (cf):     {merged_day['abs_err'].mean():.4f}")
    print(f"  NMAE (capacity): {nmae_cap*100:.2f}%   ← NGBoost 6.20% 비교")
    print(f"  80% coverage: {in_80*100:.1f}%")
    print(f"  95% coverage: {in_95*100:.1f}%")

    # 사이트별
    print(f"\n=== 사이트별 NMAE ===")
    print(f"{'site':<12} {'rows':>7} {'NMAE %':>8} {'cov80 %':>9}")
    for s in sorted(merged_day.site.unique()):
        sub = merged_day[merged_day.site == s]
        nmae_s = (sub["abs_err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum()
        cov80_s = ((sub["cf"] >= sub["q10"]) & (sub["cf"] <= sub["q90"])).mean()
        print(f"{s:<12} {len(sub):>7,} {nmae_s*100:>7.2f}% {cov80_s*100:>8.1f}%")

    # 포트폴리오
    portfolio = merged_day.groupby("datetime_kst").agg(
        actual_kwh=("cf", lambda s: (s * merged_day.loc[s.index, "site_capacity_kw"]).sum()),
        pred_kwh=("pred_med", lambda s: (s * merged_day.loc[s.index, "site_capacity_kw"]).sum()),
        cap=("site_capacity_kw", "sum"),
    )
    nmae_port = (portfolio["actual_kwh"] - portfolio["pred_kwh"]).abs().sum() / portfolio["cap"].sum()
    print(f"\n=== 포트폴리오 ===")
    print(f"  rows: {len(portfolio):,}, NMAE: {nmae_port*100:.2f}%   ← NGBoost 5.18% 비교")

    # Save predictions
    out_path = log_dir / "test_predictions.parquet"
    merged_day.to_parquet(out_path, index=False)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
