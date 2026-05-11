"""Full Pipeline Ablation — Stage 1 + Stage 2 combined evaluation.

구조:
  for seed in [42, 123, 7]:
      Stage 1 학습 (해당 seed) → trunk_seed
      for config in [A, B, C, D, E, F]:
          Stage 2 학습 (trunk_seed 위에)
          → record (Stage 1 metric, Stage 2 metric)

→ 각 config가 *Stage 1 variance*까지 포함한 full pipeline 평가
→ mean/std 로 final 추천

Total: 3 Stage 1 + 18 Stage 2 = 21 trainings
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

from train_two_mode_pv import (
    SharedTrunk, GaussianHead, TwoModePV, LocalCNNEncoder, IntradayHead,
    MAIN_FEATURES, COND_NUM_FEATURES, LOCAL_FEATURES, LOCAL_HOURS,
    TARGET, TRAIN_END, VAL_END, DIM, N_BLOCKS, LR, BATCH_SIZE,
    MAX_EPOCHS_S1, MAX_EPOCHS_S2, PATIENCE,
    load_data, build_d1, build_intraday, gaussian_nll,
    stage1_train, evaluate_d1,
)
from stage2_ablation import (
    LocalCNNEncoderV, IntradayHeadV, TwoModeAblation,
    stage2_train_config, eval_stage2,
)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CONFIGS = {
    "A_baseline":      dict(dim_local=16, fusion_hidden=64, dropout_cnn=0.0,  dropout_fusion=0.0, wd=1e-4, n_channels=4),
    "B_capacity_down": dict(dim_local=8,  fusion_hidden=32, dropout_cnn=0.0,  dropout_fusion=0.0, wd=1e-4, n_channels=4),
    "C_light_reg":     dict(dim_local=16, fusion_hidden=64, dropout_cnn=0.05, dropout_fusion=0.1, wd=5e-5, n_channels=4),
}
# 2 seeds 먼저, 결과 애매하면 3번째 추가
SEEDS = [42, 123]
TRUNK_CACHE_DIR = ROOT / "pv/experiments/trunk_cache"


def main():
    print("=" * 80, flush=True)
    print("Full Pipeline Ablation — Stage 1 + Stage 2 combined evaluation", flush=True)
    print("=" * 80, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

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

    # ===== Stage 1 per seed (3 trunks) =====
    print("\n" + "=" * 70, flush=True)
    print("[Stage 1] 3개 trunk 학습 (seed 42, 123, 7)", flush=True)
    print("=" * 70, flush=True)

    TRUNK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    stage1_results = []
    trunks = {}
    for seed in SEEDS:
        cache_path = TRUNK_CACHE_DIR / f"trunk_seed{seed}.pt"
        if cache_path.exists():
            print(f"\n[Stage 1 × seed={seed}] CACHE HIT — loading {cache_path.name}", flush=True)
            ckpt = torch.load(cache_path, map_location=device, weights_only=False)
            trunks[seed] = {"trunk": ckpt["trunk"], "d1_head": ckpt["d1_head"]}
            stage1_results.append({"seed": seed, **ckpt["metrics"]})
            print(f"  cached metrics: NMAE {ckpt['metrics']['nmae']:.3f}% / "
                  f"Port {ckpt['metrics']['pnmae']:.3f}% / Cov80 {ckpt['metrics']['cov80']:.1f}%", flush=True)
            continue

        print(f"\n[Stage 1 × seed={seed}] (training)", flush=True)
        set_all_seeds(seed)
        model = TwoModePV(n_main, n_cond, n_local_feat=4).to(device)
        train_d1_ds = TensorDataset(torch.from_numpy(main_tr), torch.from_numpy(cond_tr), torch.from_numpy(y_tr))
        val_d1_ds = TensorDataset(torch.from_numpy(main_v), torch.from_numpy(cond_v), torch.from_numpy(y_v))
        g = torch.Generator(); g.manual_seed(seed)
        tl = DataLoader(train_d1_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
        vl = DataLoader(val_d1_ds, batch_size=BATCH_SIZE)
        model = stage1_train(model, tl, vl, device)
        df_d1, nmae, pnmae, cov80 = evaluate_d1(model, main_te, cond_te, y_te, meta_te, device, f"S1 seed={seed}")
        print(f"  Stage 1 final: NMAE {nmae:.3f}% / Port {pnmae:.3f}% / Cov80 {cov80:.1f}%", flush=True)
        metrics = {"stage": "1", "nmae": nmae, "pnmae": pnmae, "cov80": cov80}
        stage1_results.append({"seed": seed, **metrics})
        trunks[seed] = {
            "trunk": {k: v.cpu().clone() for k, v in model.trunk.state_dict().items()},
            "d1_head": {k: v.cpu().clone() for k, v in model.d1_head.state_dict().items()},
        }
        torch.save({"trunk": trunks[seed]["trunk"], "d1_head": trunks[seed]["d1_head"], "metrics": metrics},
                   cache_path)
        print(f"  trunk cached: {cache_path.name}", flush=True)

    # ===== Stage 2 per config × seed =====
    print("\n" + "=" * 70, flush=True)
    print("[Stage 2] Configurations × Seeds (18 runs)", flush=True)
    print("=" * 70, flush=True)

    stage2_results = []
    for cfg_name, cfg in CONFIGS.items():
        n_ch = cfg["n_channels"]
        if n_ch == 4:
            l_train, l_val, l_test = l_tr, l_v, l_te
        else:
            l_train = l_tr[:, :, :2]; l_val = l_v[:, :, :2]; l_test = l_te[:, :, :2]

        for seed in SEEDS:
            print(f"\n  [{cfg_name} × seed={seed}] dim_local={cfg['dim_local']}, "
                  f"fusion={cfg['fusion_hidden']}, dropout={cfg['dropout_cnn']}/{cfg['dropout_fusion']}, "
                  f"wd={cfg['wd']}, n_ch={n_ch}", flush=True)
            set_all_seeds(seed)

            trunk = SharedTrunk(n_main, n_cond, dim=DIM, n_blocks=N_BLOCKS).to(device)
            trunk.load_state_dict(trunks[seed]["trunk"])
            d1_head = GaussianHead(DIM).to(device)
            d1_head.load_state_dict(trunks[seed]["d1_head"])
            local_enc = LocalCNNEncoderV(
                n_local_feat=n_ch, dim_local=cfg["dim_local"], dropout=cfg["dropout_cnn"]
            ).to(device)
            intra_head = IntradayHeadV(
                dim_global=DIM, dim_local=cfg["dim_local"],
                fusion_hidden=cfg["fusion_hidden"], dropout=cfg["dropout_fusion"]
            ).to(device)
            model = TwoModeAblation(trunk, d1_head, local_enc, intra_head).to(device)

            train_i_ds = TensorDataset(torch.from_numpy(g_tr), torch.from_numpy(c_tr),
                                       torch.from_numpy(l_train), torch.from_numpy(y_i_tr))
            val_i_ds = TensorDataset(torch.from_numpy(g_v), torch.from_numpy(c_v),
                                     torch.from_numpy(l_val), torch.from_numpy(y_i_v))
            g_i = torch.Generator(); g_i.manual_seed(seed)
            tl = DataLoader(train_i_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_i)
            vl = DataLoader(val_i_ds, batch_size=BATCH_SIZE)
            model = stage2_train_config(model, tl, vl, device, cfg["wd"])
            nmae, pnmae, cov80 = eval_stage2(model, g_te, c_te, l_test, y_i_te, meta_i_te, device)
            print(f"    → S2 NMAE {nmae:.3f}% / Port {pnmae:.3f}% / Cov80 {cov80:.1f}%", flush=True)
            stage2_results.append({
                "config": cfg_name, "seed": seed,
                **cfg, "nmae": nmae, "pnmae": pnmae, "cov80": cov80,
            })

    # ===== Aggregation =====
    s1_df = pd.DataFrame(stage1_results)
    s2_df = pd.DataFrame(stage2_results)

    print("\n" + "=" * 80, flush=True)
    print("Stage 1 results (trunk variance, 3 seeds)", flush=True)
    print("=" * 80, flush=True)
    print(s1_df[["seed", "nmae", "pnmae", "cov80"]].to_string(index=False), flush=True)
    print(f"\n  Stage 1 mean: NMAE {s1_df['nmae'].mean():.3f}% ± {s1_df['nmae'].std():.3f}%, "
          f"Port {s1_df['pnmae'].mean():.3f}% ± {s1_df['pnmae'].std():.3f}%", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("Stage 2 — config별 mean ± std (3 seeds, full pipeline)", flush=True)
    print("=" * 80, flush=True)
    summary = s2_df.groupby("config").agg(
        nmae_mean=("nmae", "mean"), nmae_std=("nmae", "std"),
        pnmae_mean=("pnmae", "mean"), pnmae_std=("pnmae", "std"),
        cov80_mean=("cov80", "mean"), cov80_std=("cov80", "std"),
    ).round(3)
    print(summary.to_string(), flush=True)

    summary["score"] = summary["nmae_mean"] + 2 * summary["nmae_std"]
    summary = summary.sort_values("score")
    print("\n[Recommendation] Stability-aware score (mean + 2*std, lower=better):", flush=True)
    print(summary[["nmae_mean", "nmae_std", "pnmae_mean", "pnmae_std", "score"]].to_string(), flush=True)

    best = summary.index[0]
    print(f"\n→ 가장 안정적: **{best}** (score {summary.loc[best, 'score']:.3f})", flush=True)
    print(f"   Stage 2 NMAE {summary.loc[best, 'nmae_mean']:.2f}% ± {summary.loc[best, 'nmae_std']:.2f}%", flush=True)

    out_dir = ROOT / "pv/experiments/full_pipeline_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_df.to_csv(out_dir / "stage1_results.csv", index=False)
    s2_df.to_csv(out_dir / "stage2_results.csv", index=False)
    summary.to_csv(out_dir / "summary.csv")
    print(f"\n저장: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
