"""Hybrid 모델 D-1 mode 평가 — autoregressive rollout.

규칙:
  - D-1 23:59까지 cf 실측 사용 (D 시작 직전까지의 history는 real)
  - D-day 야간 (cf=0인 시간)은 0으로 알려진 값으로 채움
  - D-day daytime 시간 t의 예측 시:
      local history = [cf[t-6], ..., cf[t-1]]
      이때 hour < D 00:00 → real cf
      hour >= D 00:00 → 예측된 cf (또는 야간 0)
  - global / cond features는 *예보 가능*으로 가정 (test set의 실측값 사용)

비교:
  ResMLP+AdaLN (D-1)        : 5.12% / 4.13% / cov80 82.6%
  Hybrid (D-1 rollout)      : 이 결과
  Hybrid (intraday 1h-ahead): 4.27% / 3.74% / cov80 79.6%
  NGBoost + Bayesian reweight: 별도 평가 (아래)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# Same config as training script
MAIN_FEATURES = ["dsr_mean", "zenith_center", "ta", "hm", "ws"]
COND_NUM_FEATURES = ["dc10Tca", "hour_sin", "hour_cos", "month_sin", "month_cos"]
LOCAL_FEATURES = ["cf", "dsr_mean", "delta_cf", "delta_dsr"]
LOCAL_HOURS = 6
TARGET = "cf"
TRAIN_END = pd.Timestamp("2024-01-01")
VAL_END = pd.Timestamp("2025-01-01")

DIM_GLOBAL = 64
N_BLOCKS = 4
DIM_LOCAL = 16


# Model classes (must match training)
class AdaLNBlock(nn.Module):
    def __init__(self, dim, n_cond):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_to_ss = nn.Linear(n_cond, 2 * dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.cond_to_ss(cond).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        h = self.mlp(h)
        return x + h


class HybridResMLPCNN(nn.Module):
    def __init__(self, n_main, n_cond, n_local_feat,
                 dim_g=DIM_GLOBAL, n_blocks=N_BLOCKS, dim_l=DIM_LOCAL):
        super().__init__()
        self.global_input = nn.Linear(n_main, dim_g)
        self.global_blocks = nn.ModuleList([AdaLNBlock(dim_g, n_cond) for _ in range(n_blocks)])
        self.global_final_norm = nn.LayerNorm(dim_g, elementwise_affine=False)
        self.global_final_cond = nn.Linear(n_cond, 2 * dim_g)
        self.local_conv = nn.Sequential(
            nn.Conv1d(n_local_feat, dim_l, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(dim_l, dim_l, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(dim_g + dim_l, dim_g), nn.GELU(),
            nn.Linear(dim_g, dim_g), nn.GELU(),
        )
        self.mu_head = nn.Linear(dim_g, 1)
        self.log_sig_head = nn.Linear(dim_g, 1)

    def forward(self, x_global, x_cond, x_local):
        h_g = self.global_input(x_global)
        for blk in self.global_blocks:
            h_g = blk(h_g, x_cond)
        h_g = self.global_final_norm(h_g)
        scale, shift = self.global_final_cond(x_cond).chunk(2, dim=-1)
        h_g = h_g * (1 + scale) + shift
        h_l = self.local_conv(x_local.transpose(1, 2)).squeeze(-1)
        h = torch.cat([h_g, h_l], dim=-1)
        h = self.fusion(h)
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.log_sig_head(h)).squeeze(-1) + 1e-3
        return mu, sigma


def load_data():
    df = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    df = df.sort_values(["site", "datetime_kst"]).reset_index(drop=True)
    df["is_daytime"] = (df["dsr_mean"].notna() & df["cf"].notna()).astype(np.int32)
    for col in ["ta", "hm", "ws", "dc10Tca", "dsr_mean", "zenith_center"]:
        df[col] = df.groupby("site")[col].transform(lambda s: s.fillna(s.median()))
    df["rn"] = df["rn"].fillna(0)
    df["cf"] = df["cf"].fillna(0)
    return df


def main():
    print("=" * 70)
    print("Hybrid D-1 Rollout 평가 (autoregressive)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(ROOT / "pv/experiments/hybrid_resmlp_cnn/model.pt",
                      map_location=device, weights_only=False)
    all_sites = ckpt["all_sites"]
    mean_g, std_g, mean_l, std_l = ckpt["scalers"]

    df = load_data()
    print(f"  data rows: {len(df):,}")

    site_idx = {s: i for i, s in enumerate(all_sites)}
    n_sites = len(all_sites)
    n_main = len(MAIN_FEATURES)
    n_cond = len(COND_NUM_FEATURES) + n_sites
    n_local = len(LOCAL_FEATURES)
    model = HybridResMLPCNN(n_main, n_cond, n_local).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  model loaded — {sum(p.numel() for p in model.parameters())/1e3:.1f}k params")

    # === Rollout per (site, test day) ===
    print("\n[Rollout] D-1 17:00 → D 24h 시뮬레이션...")
    test_predictions = []
    n_processed = 0

    for site, sub in df.groupby("site"):
        sub = sub.sort_values("datetime_kst").reset_index(drop=True)
        # Real arrays
        gm = sub[MAIN_FEATURES].values.astype(np.float32)
        gm_s = (gm - mean_g) / std_g
        cn = sub[COND_NUM_FEATURES].values.astype(np.float32)
        cn[:, 0] /= 10.0
        soh = np.zeros((len(sub), n_sites), dtype=np.float32)
        soh[:, site_idx[site]] = 1.0
        cond_arr = np.concatenate([cn, soh], axis=1)
        cf_real = sub["cf"].values.astype(np.float32)
        dsr_real = sub["dsr_mean"].values.astype(np.float32)
        is_day = sub["is_daytime"].values
        # Date-of-each-row to identify D boundaries
        dt = sub["datetime_kst"].values
        dates = pd.to_datetime(dt).normalize()

        # cf_pred: array we fill recursively for D-day hours
        cf_pred = cf_real.copy()   # we'll overwrite test day hours

        # Process each test day in 2025
        unique_test_dates = pd.to_datetime(sub["datetime_kst"]).dt.normalize().unique()
        unique_test_dates = [d for d in unique_test_dates if d >= VAL_END]

        for day in unique_test_dates:
            day_mask = dates == day
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue
            # For each hour of D from start, predict if daytime
            for idx in day_indices:
                if not is_day[idx]:
                    cf_pred[idx] = 0.0   # night = 0
                    continue
                if idx < LOCAL_HOURS:
                    cf_pred[idx] = cf_real[idx]   # not enough history (early data)
                    continue
                # Build local from cf_pred (recursive: D hours use pred, D-1 hours use real)
                hist_idx = slice(idx - LOCAL_HOURS, idx)
                cf_hist = cf_pred[hist_idx]
                dsr_hist = dsr_real[hist_idx]   # DSR forecast 가용 가정
                delta_cf = np.diff(cf_hist, prepend=cf_pred[idx - LOCAL_HOURS - 1] if idx >= LOCAL_HOURS + 1 else cf_hist[0])
                delta_dsr = np.diff(dsr_hist, prepend=dsr_real[idx - LOCAL_HOURS - 1] if idx >= LOCAL_HOURS + 1 else dsr_hist[0])
                local = np.stack([cf_hist, dsr_hist, delta_cf, delta_dsr], axis=-1).astype(np.float32)
                local_s = (local - mean_l) / std_l

                # Predict
                xg = torch.from_numpy(gm_s[idx:idx+1]).to(device)
                xc = torch.from_numpy(cond_arr[idx:idx+1]).to(device)
                xl = torch.from_numpy(local_s[None, :, :]).to(device)
                with torch.no_grad():
                    mu, sig = model(xg, xc, xl)
                mu_v = float(mu.cpu().numpy()[0])
                sig_v = float(sig.cpu().numpy()[0])
                mu_v = max(mu_v, 0.0)
                cf_pred[idx] = mu_v   # 다음 시점 history에 사용
                test_predictions.append({
                    "datetime_kst": sub["datetime_kst"].iloc[idx],
                    "site": site,
                    "site_capacity_kw": float(sub["site_capacity_kw"].iloc[idx]),
                    "cf": float(cf_real[idx]),
                    "pred_cf": mu_v,
                    "pred_std_cf": sig_v,
                })

        n_processed += 1
        print(f"  {site}: {n_processed}/8 사이트 처리됨", flush=True)

    pred_df = pd.DataFrame(test_predictions)
    print(f"\n  총 {len(pred_df):,} 예측 (rollout)")

    # === Metrics ===
    cap = pred_df["site_capacity_kw"].values
    y = pred_df["cf"].values
    mu = pred_df["pred_cf"].values
    sig = pred_df["pred_std_cf"].values
    err = np.abs(y - mu)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu - 1.28 * sig) & (y <= mu + 1.28 * sig)).mean() * 100
    cov95 = ((y >= mu - 1.96 * sig) & (y <= mu + 1.96 * sig)).mean() * 100
    print(f"\n=== Hybrid D-1 Rollout ===")
    print(f"  rows: {len(pred_df):,}")
    print(f"  NMAE: {nmae:.2f}% / cov80: {cov80:.1f}% / cov95: {cov95:.1f}%")

    print(f"\n  사이트별 NMAE:")
    pred_df["err"] = err
    for s in sorted(pred_df["site"].unique()):
        sub = pred_df[pred_df["site"] == s]
        n = (sub["err"] * sub["site_capacity_kw"]).sum() / sub["site_capacity_kw"].sum() * 100
        print(f"    {s:<14} {n:>6.2f}%")

    # Portfolio
    pred_df["pred_kwh"] = mu * pred_df["site_capacity_kw"]
    pred_df["actual_kwh"] = y * pred_df["site_capacity_kw"]
    pred_df["var_kwh2"] = (sig * pred_df["site_capacity_kw"]) ** 2
    port = pred_df.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        var_sum=("var_kwh2", "sum"), cap=("site_capacity_kw", "sum"))
    port["sigma"] = np.sqrt(port["var_sum"])
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    pcov80 = ((port["actual"] >= port["pred"] - 1.28 * port["sigma"])
              & (port["actual"] <= port["pred"] + 1.28 * port["sigma"])).mean() * 100
    print(f"\n  포트폴리오 NMAE: {pnmae:.2f}% / cov80: {pcov80:.1f}%")

    out_dir = ROOT / "pv/experiments/hybrid_d1_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)

    # === 종합 비교 표 ===
    print("\n" + "=" * 90)
    print("종합 비교 표")
    print("=" * 90)
    print(f"{'Mode':<35} {'Site NMAE':>10} {'Port NMAE':>10} {'Cov80':>8}")
    print("-" * 90)
    print(f"{'ResMLP+AdaLN (D-1, no cf history)':<35} {'5.12%':>10} {'4.13%':>10} {'82.6%':>8}")
    print(f"{'Hybrid (D-1 rollout, recursive)':<35} {f'{nmae:.2f}%':>10} {f'{pnmae:.2f}%':>10} {f'{cov80:.1f}%':>8}")
    print(f"{'Hybrid (intraday 1h, real cf hist)':<35} {'4.27%':>10} {'3.74%':>10} {'79.6%':>8}")
    print(f"{'NGBoost + Bayesian reweight (1h)':<35} {'?':>10} {'?':>10} {'?':>8}")
    print("-" * 90)
    print("\n주: NGBoost + reweight는 1h MAE 3.83 MWh (5% 개선) / σ -15% / cov80 별도 측정")


if __name__ == "__main__":
    main()
