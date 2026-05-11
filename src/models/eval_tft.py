"""TFT checkpoint 로드 → test set 평가만 (재학습 없이).

체크포인트 위치: checkpoints/*.ckpt
test 데이터: 2025년
출력: pv/experiments/tft_baseline/test_predictions.parquet + metrics
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


def load_data():
    df = pd.read_parquet(ROOT / "data" / "processed" / "training_set.parquet")
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)
    df["cf"] = df["cf"].fillna(0)
    df["dsr_mean"] = df["dsr_mean"].fillna(0)
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df = df.dropna(subset=["ta", "hm", "ws", "dc10Tca"])
    min_dt = df.datetime_kst.min()
    df["time_idx"] = ((df.datetime_kst - min_dt).dt.total_seconds() / 3600).astype(int)
    df["is_daytime"] = (df["zenith_center"] < 90).astype(int)
    return df


def main():
    print("=== TFT Evaluation (체크포인트 로드 → test 평가) ===", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    print("\n[1/4] 데이터 로딩...", flush=True)
    df = load_data()
    train_cutoff = int((pd.Timestamp("2024-01-01") - df.datetime_kst.min()).total_seconds() / 3600)
    val_cutoff = int((pd.Timestamp("2025-01-01") - df.datetime_kst.min()).total_seconds() / 3600)
    test_cutoff = int((pd.Timestamp("2026-01-01") - df.datetime_kst.min()).total_seconds() / 3600)

    # 학습 시 사용한 동일 config로 dataset 재생성
    print("[2/4] TimeSeriesDataSet 재생성 (학습 동일 config)...", flush=True)
    training = TimeSeriesDataSet(
        df[df.time_idx < train_cutoff],
        time_idx="time_idx",
        target="cf",
        group_ids=["site"],
        max_encoder_length=24,
        max_prediction_length=24,
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
    test = TimeSeriesDataSet.from_dataset(
        training, df[df.time_idx < test_cutoff],
        stop_randomization=True,
    )
    test_loader = test.to_dataloader(train=False, batch_size=16, num_workers=0)
    print(f"  test batches: {len(test_loader)}", flush=True)

    # 체크포인트 로드
    print("\n[3/4] 체크포인트 로드...", flush=True)
    ckpts = sorted((ROOT / "checkpoints").glob("*.ckpt"))
    print(f"  available: {[c.name for c in ckpts]}", flush=True)
    # 가장 최신 (epoch 큰 것) 선택
    best_ckpt = ckpts[-1]
    print(f"  load: {best_ckpt}", flush=True)

    tft = TemporalFusionTransformer.load_from_checkpoint(str(best_ckpt))
    print(f"  params: {sum(p.numel() for p in tft.parameters())/1e3:.1f}k", flush=True)

    # 추론
    print("\n[4/4] Test 추론...", flush=True)
    raw_predictions = tft.predict(
        test_loader,
        mode="raw",
        return_index=True,
        return_x=False,
        trainer_kwargs=dict(accelerator="gpu", precision="bf16-mixed", logger=False),
    )
    pred = raw_predictions.output["prediction"].cpu().numpy()
    idx = raw_predictions.index
    print(f"  predictions shape: {pred.shape}, index rows: {len(idx)}", flush=True)

    # 평가
    rows = []
    for i, row in idx.iterrows():
        site, t0 = row["site"], int(row["time_idx"])
        for h in range(24):
            r = {"site": site, "time_idx": t0 + h}
            for j, q in enumerate(QUANTILES):
                r[f"q{int(q*100):02d}"] = float(pred[i, h, j])
            rows.append(r)
    pred_df = pd.DataFrame(rows)

    eval_df = df[["site", "time_idx", "datetime_kst", "cf", "site_capacity_kw", "is_daytime", "zenith_center"]].copy()
    merged = pred_df.merge(eval_df, on=["site", "time_idx"], how="inner")
    merged = merged[merged.datetime_kst >= pd.Timestamp("2025-01-01")]
    merged_day = merged[merged.is_daytime == 1].copy()
    print(f"  daytime test rows: {len(merged_day):,}", flush=True)

    # NMAE
    merged_day["pred_med"] = merged_day["q50"].clip(lower=0)
    merged_day["abs_err"] = (merged_day["cf"] - merged_day["pred_med"]).abs()
    merged_day["abs_err_kwh"] = merged_day["abs_err"] * merged_day["site_capacity_kw"]
    nmae_cap = merged_day["abs_err_kwh"].sum() / merged_day["site_capacity_kw"].sum()

    # Coverage
    in_80 = ((merged_day["cf"] >= merged_day["q10"]) & (merged_day["cf"] <= merged_day["q90"])).mean()
    in_95 = ((merged_day["cf"] >= merged_day["q05"]) & (merged_day["cf"] <= merged_day["q95"])).mean()

    print(f"\n=== TEST 전체 (daytime) — TFT ===", flush=True)
    print(f"  rows:           {len(merged_day):,}", flush=True)
    print(f"  MAE (cf):       {merged_day['abs_err'].mean():.4f}", flush=True)
    print(f"  NMAE (capacity): {nmae_cap*100:.2f}%   ← NGBoost 6.20%", flush=True)
    print(f"  80% coverage:   {in_80*100:.1f}%   ← NGBoost 80.6%", flush=True)
    print(f"  95% coverage:   {in_95*100:.1f}%   ← NGBoost 92.9%", flush=True)

    # 사이트별
    print(f"\n=== 사이트별 NMAE ===", flush=True)
    print(f"{'site':<12} {'rows':>7} {'NMAE %':>8} {'cov80 %':>9}", flush=True)
    for s in sorted(merged_day.site.unique()):
        sub = merged_day[merged_day.site == s]
        nmae_s = (sub["abs_err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum()
        cov80_s = ((sub["cf"] >= sub["q10"]) & (sub["cf"] <= sub["q90"])).mean()
        print(f"{s:<12} {len(sub):>7,} {nmae_s*100:>7.2f}% {cov80_s*100:>8.1f}%", flush=True)

    # 포트폴리오
    portfolio = merged_day.groupby("datetime_kst").apply(
        lambda g: pd.Series({
            "actual_kwh": (g["cf"] * g["site_capacity_kw"]).sum(),
            "pred_kwh": (g["pred_med"] * g["site_capacity_kw"]).sum(),
            "cap": g["site_capacity_kw"].sum(),
        }), include_groups=False
    )
    nmae_port = (portfolio["actual_kwh"] - portfolio["pred_kwh"]).abs().sum() / portfolio["cap"].sum()
    print(f"\n=== 포트폴리오 ===", flush=True)
    print(f"  rows: {len(portfolio):,}, NMAE: {nmae_port*100:.2f}%   ← NGBoost 5.18%", flush=True)

    # Save
    out_dir = ROOT / "pv" / "experiments" / "tft_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_day.to_parquet(out_dir / "test_predictions.parquet", index=False)
    print(f"\n저장: {out_dir / 'test_predictions.parquet'}", flush=True)


if __name__ == "__main__":
    main()
