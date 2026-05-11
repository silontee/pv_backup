"""Stage 2 robustness ablation — variance 구조적 감소 목표.

전략:
  1. Stage 1 trunk 한 번 학습 (seed 42) → 저장
  2. Stage 2 configurations 각각:
     - 6 configurations × 3 seeds (42, 123, 7)
     - 각 설정마다 Stage 2만 학습 (trunk frozen, freshly init)
     - test eval → mean / std 집계
  3. 가장 안정적 (낮은 std + 낮은 mean NMAE) 설정 추천

Configurations:
  A. baseline       (full capacity, no reg, 4 ch)
  B. capacity_down  (dim_local 16→8, fusion 64→32, no reg, 4 ch)
  C. light_reg      (full capacity, dropout 0.05/0.1, wd 5e-5, 4 ch)
  D. combined       (capacity_down + light_reg, 4 ch)
  E. ch2            (full capacity, no reg, 2 ch [cf, dsr_mean])
  F. ch2_cap_down   (capacity_down, no reg, 2 ch)

각 config × seed → Stage 2 학습 → test NMAE / Portfolio NMAE / Cov80
"""
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/models"))

# Reuse base classes
from train_two_mode_pv import (
    AdaLNBlock, SharedTrunk, GaussianHead,
    MAIN_FEATURES, COND_NUM_FEATURES, LOCAL_FEATURES, LOCAL_HOURS,
    TARGET, TRAIN_END, VAL_END, DIM, N_BLOCKS, LR, BATCH_SIZE,
    MAX_EPOCHS_S1, MAX_EPOCHS_S2, PATIENCE,
    load_data, build_d1, build_intraday, gaussian_nll,
    stage1_train, evaluate_d1,
)


# ===== Configurable Stage 2 components =====

class LocalCNNEncoderV(nn.Module):
    def __init__(self, n_local_feat, dim_local, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv1d(n_local_feat, dim_local, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Conv1d(dim_local, dim_local, kernel_size=3, padding=1),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)

    def forward(self, x_local):
        return self.conv(x_local.transpose(1, 2)).squeeze(-1)


class IntradayHeadV(nn.Module):
    def __init__(self, dim_global, dim_local, fusion_hidden=64, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(dim_global + dim_local, fusion_hidden), nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(fusion_hidden, fusion_hidden), nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.fusion = nn.Sequential(*layers)
        self.gaussian = GaussianHead(fusion_hidden)

    def forward(self, h_global, h_local):
        h = torch.cat([h_global, h_local], dim=-1)
        h = self.fusion(h)
        return self.gaussian(h)


class TwoModeAblation(nn.Module):
    """trunk + d1_head loaded externally; local_encoder + intraday_head per config."""
    def __init__(self, trunk, d1_head, local_encoder, intraday_head):
        super().__init__()
        self.trunk = trunk
        self.d1_head = d1_head
        self.local_encoder = local_encoder
        self.intraday_head = intraday_head

    def forward(self, x_main, x_cond, mode, x_local=None):
        h_global = self.trunk(x_main, x_cond)
        if mode == "d1":
            return self.d1_head(h_global)
        h_local = self.local_encoder(x_local)
        return self.intraday_head(h_global, h_local)


# ===== Seed control =====

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===== Stage 2 training =====

def stage2_train_config(model, train_loader, val_loader, device, weight_decay, max_epochs=MAX_EPOCHS_S2):
    # Freeze trunk + d1
    for p in model.trunk.parameters():
        p.requires_grad = False
    for p in model.d1_head.parameters():
        p.requires_grad = False
    params = list(model.local_encoder.parameters()) + list(model.intraday_head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(max_epochs):
        model.train()
        model.trunk.eval()
        model.d1_head.eval()
        for xm, xc, xl, y in train_loader:
            xm, xc, xl, y = xm.to(device), xc.to(device), xl.to(device), y.to(device)
            mu, sig = model(xm, xc, mode="intraday", x_local=xl)
            loss = gaussian_nll(y, mu, sig).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sch.step()
        model.eval()
        v = 0; n = 0
        with torch.no_grad():
            for xm, xc, xl, y in val_loader:
                xm, xc, xl, y = xm.to(device), xc.to(device), xl.to(device), y.to(device)
                mu, sig = model(xm, xc, mode="intraday", x_local=xl)
                v += gaussian_nll(y, mu, sig).mean().item() * len(y); n += len(y)
        v /= n
        if v < best:
            best = v
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model


def eval_stage2(model, g, c, l, y, meta, device):
    model.eval()
    ds = TensorDataset(torch.from_numpy(g), torch.from_numpy(c), torch.from_numpy(l), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=BATCH_SIZE * 4)
    mus, sigs = [], []
    with torch.no_grad():
        for xm, xc, xl, _ in loader:
            mu, sig = model(xm.to(device), xc.to(device), mode="intraday", x_local=xl.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)
    cap = meta["site_capacity_kw"].values
    err = np.abs(y - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((y >= mu_arr - 1.282 * sig_arr) & (y <= mu_arr + 1.282 * sig_arr)).mean() * 100
    df = meta.copy(); df["pred_kwh"] = mu_arr * df["site_capacity_kw"]
    df["actual_kwh"] = y * df["site_capacity_kw"]
    port = df.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"), cap=("site_capacity_kw", "sum"))
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    return nmae, pnmae, cov80


# ===== Configs =====

CONFIGS = {
    "A_baseline":      dict(dim_local=16, fusion_hidden=64, dropout_cnn=0.0, dropout_fusion=0.0, wd=1e-4, n_channels=4),
    "B_capacity_down": dict(dim_local=8,  fusion_hidden=32, dropout_cnn=0.0, dropout_fusion=0.0, wd=1e-4, n_channels=4),
    "C_light_reg":     dict(dim_local=16, fusion_hidden=64, dropout_cnn=0.05, dropout_fusion=0.1, wd=5e-5, n_channels=4),
    "D_combined":      dict(dim_local=8,  fusion_hidden=32, dropout_cnn=0.05, dropout_fusion=0.1, wd=5e-5, n_channels=4),
    "E_2ch":           dict(dim_local=16, fusion_hidden=64, dropout_cnn=0.0, dropout_fusion=0.0, wd=1e-4, n_channels=2),
    "F_2ch_cap_down":  dict(dim_local=8,  fusion_hidden=32, dropout_cnn=0.0, dropout_fusion=0.0, wd=1e-4, n_channels=2),
}
SEEDS = [42, 123, 7]


def main():
    print("=" * 80, flush=True)
    print("Stage 2 Robustness Ablation", flush=True)
    print("=" * 80, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ===== Load data once =====
    print("\n[Data] Loading...", flush=True)
    df = load_data()
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]
    all_sites = sorted(df["site"].unique())

    main_tr, cond_tr, y_tr, meta_tr, scalers_d1 = build_d1(train_df, all_sites)
    main_v, cond_v, y_v, meta_v, _ = build_d1(val_df, all_sites, *scalers_d1)
    main_te, cond_te, y_te, meta_te, _ = build_d1(test_df, all_sites, *scalers_d1)

    g_tr, c_tr, l_tr, y_i_tr, meta_i_tr, scalers_intra = build_intraday(train_df, all_sites, scalers_d1)
    g_v, c_v, l_v, y_i_v, meta_i_v, _ = build_intraday(val_df, all_sites, scalers_intra)
    g_te, c_te, l_te, y_i_te, meta_i_te, _ = build_intraday(test_df, all_sites, scalers_intra)

    n_main = main_tr.shape[1]; n_cond = cond_tr.shape[1]
    print(f"  D-1: train {len(y_tr):,}, val {len(y_v):,}, test {len(y_te):,}", flush=True)
    print(f"  Intraday: train {len(y_i_tr):,}, val {len(y_i_v):,}, test {len(y_i_te):,}", flush=True)

    # ===== Stage 1 trunk: train once with seed 42 =====
    print("\n" + "=" * 70, flush=True)
    print("[Stage 1] Trunk + d1_head 학습 (seed 42, 1회)", flush=True)
    print("=" * 70, flush=True)
    set_all_seeds(42)

    from train_two_mode_pv import TwoModePV, LocalCNNEncoder, IntradayHead
    model0 = TwoModePV(n_main, n_cond, n_local_feat=4).to(device)

    train_d1_ds = TensorDataset(torch.from_numpy(main_tr), torch.from_numpy(cond_tr), torch.from_numpy(y_tr))
    val_d1_ds = TensorDataset(torch.from_numpy(main_v), torch.from_numpy(cond_v), torch.from_numpy(y_v))
    g_d1 = torch.Generator(); g_d1.manual_seed(42)
    train_d1_loader = DataLoader(train_d1_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_d1)
    val_d1_loader = DataLoader(val_d1_ds, batch_size=BATCH_SIZE)

    model0 = stage1_train(model0, train_d1_loader, val_d1_loader, device)
    df_d1, nmae_d1, pnmae_d1, cov_d1 = evaluate_d1(model0, main_te, cond_te, y_te, meta_te, device, "Stage 1")
    print(f"\n  Stage 1 final: NMAE {nmae_d1:.2f}% / Port {pnmae_d1:.2f}% / Cov80 {cov_d1:.1f}%", flush=True)

    # Save trunk + d1_head state
    trunk_state = {k: v.cpu().clone() for k, v in model0.trunk.state_dict().items()}
    d1_state = {k: v.cpu().clone() for k, v in model0.d1_head.state_dict().items()}

    # ===== Stage 2 ablation =====
    print("\n" + "=" * 70, flush=True)
    print("[Stage 2] Configuration × Seed ablation", flush=True)
    print("=" * 70, flush=True)

    results = []
    for cfg_name, cfg in CONFIGS.items():
        n_ch = cfg["n_channels"]
        # Build local features per channel count
        if n_ch == 4:
            l_train, l_val, l_test = l_tr, l_v, l_te
        elif n_ch == 2:
            l_train = l_tr[:, :, :2]   # cf, dsr_mean
            l_val = l_v[:, :, :2]
            l_test = l_te[:, :, :2]
        else:
            raise ValueError

        for seed in SEEDS:
            print(f"\n  [{cfg_name} × seed={seed}] dim_local={cfg['dim_local']}, "
                  f"fusion={cfg['fusion_hidden']}, dropout={cfg['dropout_cnn']}/{cfg['dropout_fusion']}, "
                  f"wd={cfg['wd']}, n_ch={n_ch}", flush=True)
            set_all_seeds(seed)

            # Fresh trunk + d1_head (load saved state)
            trunk = SharedTrunk(n_main, n_cond, dim=DIM, n_blocks=N_BLOCKS).to(device)
            trunk.load_state_dict(trunk_state)
            d1_head = GaussianHead(DIM).to(device)
            d1_head.load_state_dict(d1_state)
            # Fresh local + intraday heads (per config)
            local_enc = LocalCNNEncoderV(
                n_local_feat=n_ch, dim_local=cfg["dim_local"], dropout=cfg["dropout_cnn"]
            ).to(device)
            intra_head = IntradayHeadV(
                dim_global=DIM, dim_local=cfg["dim_local"],
                fusion_hidden=cfg["fusion_hidden"], dropout=cfg["dropout_fusion"]
            ).to(device)
            model = TwoModeAblation(trunk, d1_head, local_enc, intra_head).to(device)

            # Loaders
            train_i_ds = TensorDataset(torch.from_numpy(g_tr), torch.from_numpy(c_tr),
                                       torch.from_numpy(l_train), torch.from_numpy(y_i_tr))
            val_i_ds = TensorDataset(torch.from_numpy(g_v), torch.from_numpy(c_v),
                                     torch.from_numpy(l_val), torch.from_numpy(y_i_v))
            g_i = torch.Generator(); g_i.manual_seed(seed)
            train_i_loader = DataLoader(train_i_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_i)
            val_i_loader = DataLoader(val_i_ds, batch_size=BATCH_SIZE)

            model = stage2_train_config(model, train_i_loader, val_i_loader, device, cfg["wd"])
            nmae, pnmae, cov80 = eval_stage2(model, g_te, c_te, l_test, y_i_te, meta_i_te, device)
            print(f"    → NMAE {nmae:.3f}% / Port {pnmae:.3f}% / Cov80 {cov80:.1f}%", flush=True)
            results.append({
                "config": cfg_name, "seed": seed,
                **cfg,
                "nmae": nmae, "pnmae": pnmae, "cov80": cov80,
            })

    # ===== Aggregation =====
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 80, flush=True)
    print("종합 결과 (mean ± std over 3 seeds)", flush=True)
    print("=" * 80, flush=True)

    summary = res_df.groupby("config").agg(
        nmae_mean=("nmae", "mean"),
        nmae_std=("nmae", "std"),
        pnmae_mean=("pnmae", "mean"),
        pnmae_std=("pnmae", "std"),
        cov80_mean=("cov80", "mean"),
        cov80_std=("cov80", "std"),
    ).round(3)
    print(summary.to_string(), flush=True)

    # Stability score: nmae_mean + 2 * nmae_std (worst-case)
    summary["score"] = summary["nmae_mean"] + 2 * summary["nmae_std"]
    summary = summary.sort_values("score")
    print("\n[Recommendation] Stability score (lower is better) = mean + 2*std:", flush=True)
    print(summary[["nmae_mean", "nmae_std", "pnmae_mean", "pnmae_std", "score"]].to_string(), flush=True)

    best_cfg = summary.index[0]
    print(f"\n→ 가장 안정적: **{best_cfg}** (score {summary.loc[best_cfg, 'score']:.3f})", flush=True)
    print(f"   NMAE {summary.loc[best_cfg, 'nmae_mean']:.2f}% ± {summary.loc[best_cfg, 'nmae_std']:.2f}%", flush=True)
    print(f"   Portfolio {summary.loc[best_cfg, 'pnmae_mean']:.2f}% ± {summary.loc[best_cfg, 'pnmae_std']:.2f}%", flush=True)

    # Save
    out_dir = ROOT / "pv/experiments/stage2_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "raw_results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    print(f"\n저장: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
