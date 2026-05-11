"""Boundary fix 검증 — A_baseline (winner) 1개만 재평가.

Build_intraday를 *full_df 기반 + target time split*으로 수정:
  - History는 전체 데이터에서 자유롭게 가져옴 (cross-split 가능)
  - Target만 split의 시간 범위로 필터

A_baseline (NMAE 3.86%, Port 3.28%, Cov80 79.1%) winner를 *수정된 데이터*로
seed=42 trunk + seed=42 random init Stage 2 학습 → 비교.

전체 ablation 다시 안 돌림 — 최종안 검증만.
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
    MAX_EPOCHS_S2, PATIENCE,
    load_data, build_d1, build_intraday, gaussian_nll,
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


def build_intraday_v2(full_df, all_sites, target_start, target_end, scalers=None):
    """V2: full_df 기반 + target time filter.

    History 가능한 범위:
      - target_dt at 2025-01-01 09:00
      - history = 2024-12-31 12:00 ~ 2025-01-01 08:00 (cross-split OK)
    """
    site_idx = {s: i for i, s in enumerate(all_sites)}
    n_sites = len(all_sites)
    if scalers is not None:
        mean_g, std_g, mean_l, std_l = scalers
    else:
        mean_g = std_g = mean_l = std_l = None

    g_list, c_list, l_list, y_list, meta = [], [], [], [], []
    for site, sub in full_df.groupby("site"):
        sub = sub.sort_values("datetime_kst").reset_index(drop=True)
        gm = sub[MAIN_FEATURES].values.astype(np.float32)
        cn = sub[COND_NUM_FEATURES].values.astype(np.float32)
        cn[:, 0] /= 10.0
        soh = np.zeros((len(sub), n_sites), dtype=np.float32)
        soh[:, site_idx[site]] = 1.0
        cond = np.concatenate([cn, soh], axis=1)
        loc = sub[LOCAL_FEATURES].values.astype(np.float32)
        is_day = sub["is_daytime"].values
        dt = sub["datetime_kst"].values

        for t in range(LOCAL_HOURS, len(sub)):
            if not is_day[t]:
                continue
            target_dt = sub["datetime_kst"].iloc[t]
            if target_dt < target_start or target_dt >= target_end:
                continue
            g_list.append(gm[t])
            c_list.append(cond[t])
            l_list.append(loc[t - LOCAL_HOURS:t])
            y_list.append(sub["cf"].iloc[t])
            meta.append({"datetime_kst": target_dt, "site": site,
                         "site_capacity_kw": sub["site_capacity_kw"].iloc[t]})

    g = np.stack(g_list); c = np.stack(c_list); l = np.stack(l_list)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta)

    if mean_g is None:
        mean_g = g.mean(axis=0); std_g = g.std(axis=0) + 1e-6
        mean_l = l.reshape(-1, l.shape[-1]).mean(axis=0)
        std_l = l.reshape(-1, l.shape[-1]).std(axis=0) + 1e-6
    g = (g - mean_g) / std_g
    l = (l - mean_l) / std_l
    return g, c, l, y, meta_df, (mean_g, std_g, mean_l, std_l)


# Stage 2 train (same as ablation)
def stage2_train_local(model, train_loader, val_loader, device, weight_decay=1e-4):
    for p in model.trunk.parameters():
        p.requires_grad = False
    for p in model.d1_head.parameters():
        p.requires_grad = False
    params = list(model.local_encoder.parameters()) + list(model.intraday_head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS_S2)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS_S2):
        model.train(); model.trunk.eval(); model.d1_head.eval()
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
            print(f"    early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


def evaluate_intra(model, g, c, l, y, meta, device):
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


def main():
    print("=" * 70)
    print("Boundary Fix Verification — A_baseline (winner) 검증")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_data()
    train_df = df[df.datetime_kst < TRAIN_END]
    val_df = df[(df.datetime_kst >= TRAIN_END) & (df.datetime_kst < VAL_END)]
    test_df = df[df.datetime_kst >= VAL_END]
    all_sites = sorted(df["site"].unique())

    # Stage 1 scalers (D-1)
    main_tr, cond_tr, y_tr, meta_tr, scalers_d1 = build_d1(train_df, all_sites)

    # ===== V1: original (test 내부만) =====
    print("\n[V1] Original — test_df 내부에서만 history")
    g_tr_v1, c_tr_v1, l_tr_v1, y_i_tr_v1, meta_i_tr_v1, scalers_v1 = build_intraday(train_df, all_sites, scalers_d1)
    g_v_v1, c_v_v1, l_v_v1, y_i_v_v1, meta_i_v_v1, _ = build_intraday(val_df, all_sites, scalers_v1)
    g_te_v1, c_te_v1, l_te_v1, y_i_te_v1, meta_i_te_v1, _ = build_intraday(test_df, all_sites, scalers_v1)
    print(f"  V1 sample 수: train {len(y_i_tr_v1):,}, val {len(y_i_v_v1):,}, test {len(y_i_te_v1):,}")

    # ===== V2: full_df + target filter =====
    print("\n[V2] Fixed — full_df + target time filter")
    g_tr_v2, c_tr_v2, l_tr_v2, y_i_tr_v2, meta_i_tr_v2, scalers_v2 = build_intraday_v2(
        df, all_sites, target_start=df["datetime_kst"].min(), target_end=TRAIN_END)
    g_v_v2, c_v_v2, l_v_v2, y_i_v_v2, meta_i_v_v2, _ = build_intraday_v2(
        df, all_sites, target_start=TRAIN_END, target_end=VAL_END, scalers=scalers_v2)
    g_te_v2, c_te_v2, l_te_v2, y_i_te_v2, meta_i_te_v2, _ = build_intraday_v2(
        df, all_sites, target_start=VAL_END, target_end=df["datetime_kst"].max() + pd.Timedelta(hours=1),
        scalers=scalers_v2)
    print(f"  V2 sample 수: train {len(y_i_tr_v2):,}, val {len(y_i_v_v2):,}, test {len(y_i_te_v2):,}")
    print(f"  V2가 V1보다 많은 sample 수 (boundary 전 LOCAL_HOURS=6 sample 추가):")
    print(f"    train: +{len(y_i_tr_v2) - len(y_i_tr_v1):,}, val: +{len(y_i_v_v2) - len(y_i_v_v1):,}, test: +{len(y_i_te_v2) - len(y_i_te_v1):,}")

    # 첫 sample 확인 — 2024-12-31 → 2025-01-01 cross-boundary가 일어나는지
    print("\n[V2 첫 5 test sample]:")
    site_check = "광양항세방"
    sub_full = df[df.site == site_check].sort_values("datetime_kst").reset_index(drop=True)
    sub_te_idx = []
    for i, r in sub_full.iterrows():
        if i < LOCAL_HOURS: continue
        if not r["is_daytime"]: continue
        if r["datetime_kst"] < VAL_END: continue
        sub_te_idx.append(i)
    for n, i in enumerate(sub_te_idx[:5]):
        target_dt = sub_full.iloc[i]["datetime_kst"]
        history_dt = sub_full.iloc[i - LOCAL_HOURS:i]["datetime_kst"].tolist()
        history_cf = sub_full.iloc[i - LOCAL_HOURS:i]["cf"].tolist()
        print(f"  Sample #{n}: target={target_dt}, history first={history_dt[0]}")
        print(f"    history cf: {[f'{c:.2f}' for c in history_cf]}")

    # Load trunk seed=42
    print("\n[Load] trunk_seed42")
    ckpt = torch.load(ROOT / "pv/experiments/trunk_cache/trunk_seed42.pt",
                      map_location=device, weights_only=False)
    n_main = main_tr.shape[1]; n_cond = cond_tr.shape[1]

    # ===== Train Stage 2 with V2 dataset (A_baseline config, seed=42) =====
    print("\n[Train] A_baseline + V2 dataset (seed=42)")
    set_all_seeds(42)

    trunk = SharedTrunk(n_main, n_cond, dim=DIM, n_blocks=N_BLOCKS).to(device)
    trunk.load_state_dict(ckpt["trunk"])
    d1_head = GaussianHead(DIM).to(device)
    d1_head.load_state_dict(ckpt["d1_head"])
    local_enc = LocalCNNEncoder(n_local_feat=4, dim_local=16).to(device)
    intra_head = IntradayHead(dim_global=DIM, dim_local=16).to(device)

    class Model(nn.Module):
        def __init__(self, trunk, d1_head, local_encoder, intraday_head):
            super().__init__()
            self.trunk = trunk; self.d1_head = d1_head
            self.local_encoder = local_encoder; self.intraday_head = intraday_head
        def forward(self, x_main, x_cond, mode, x_local=None):
            h_global = self.trunk(x_main, x_cond)
            if mode == "d1":
                return self.d1_head(h_global)
            h_local = self.local_encoder(x_local)
            return self.intraday_head(h_global, h_local)

    model = Model(trunk, d1_head, local_enc, intra_head).to(device)

    train_ds = TensorDataset(torch.from_numpy(g_tr_v2), torch.from_numpy(c_tr_v2),
                             torch.from_numpy(l_tr_v2), torch.from_numpy(y_i_tr_v2))
    val_ds = TensorDataset(torch.from_numpy(g_v_v2), torch.from_numpy(c_v_v2),
                           torch.from_numpy(l_v_v2), torch.from_numpy(y_i_v_v2))
    g_loader = torch.Generator(); g_loader.manual_seed(42)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_loader)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = stage2_train_local(model, train_loader, val_loader, device)

    # ===== Eval on V2 test =====
    print("\n[Eval] V2 test set")
    nmae_v2, pnmae_v2, cov80_v2 = evaluate_intra(model, g_te_v2, c_te_v2, l_te_v2, y_i_te_v2, meta_i_te_v2, device)
    print(f"  V2: NMAE {nmae_v2:.3f}% / Port {pnmae_v2:.3f}% / Cov80 {cov80_v2:.1f}%")

    # ===== Eval also on V1 test (same model) for fair comparison =====
    print("\n[Eval] V1 test set (same model, same scalers re-applied)")
    # V1 test data with V2 scalers
    g_te_v1_norm = (g_te_v1 * (scalers_v1[1]) + scalers_v1[0])    # de-normalize
    g_te_v1_norm = (g_te_v1_norm - scalers_v2[0]) / scalers_v2[1]  # re-normalize with V2 scalers
    l_te_v1_norm = l_te_v1 * scalers_v1[3] + scalers_v1[2]
    l_te_v1_norm = (l_te_v1_norm - scalers_v2[2]) / scalers_v2[3]
    nmae_v1, pnmae_v1, cov80_v1 = evaluate_intra(model, g_te_v1_norm, c_te_v1, l_te_v1_norm,
                                                  y_i_te_v1, meta_i_te_v1, device)
    print(f"  V1: NMAE {nmae_v1:.3f}% / Port {pnmae_v1:.3f}% / Cov80 {cov80_v1:.1f}%")

    print("\n" + "=" * 70)
    print("최종 비교")
    print("=" * 70)
    print(f"  Original (V1, A_baseline seed=42): NMAE 3.838% / Port 3.254% / Cov80 80.8%")
    print(f"  Boundary Fix (V2, same model):     NMAE {nmae_v2:.3f}% / Port {pnmae_v2:.3f}% / Cov80 {cov80_v2:.1f}%")
    print(f"  Δ NMAE: {nmae_v2 - 3.838:+.3f}%, Δ Port: {pnmae_v2 - 3.254:+.3f}%, Δ Cov80: {cov80_v2 - 80.8:+.1f}%p")


if __name__ == "__main__":
    main()
