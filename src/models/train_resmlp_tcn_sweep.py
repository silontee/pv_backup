"""ResMLP+AdaLN + TCN — batch sensitivity 마지막 sweep.

기각 전 마지막 검증:
  batch 32  + lr 5e-4   {NLL, MAE+0.2·NLL}
  batch 64  + lr 7e-4   {NLL, MAE+0.2·NLL}
  batch 256 + lr 1e-3   {NLL, MAE+0.2·NLL}

총 6 runs. ResMLP baseline 5.12% 넘는 config 있는지 확인.
없으면 hybrid 정리.
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

from train_resmlp_tcn import (
    AdaLNBlock, ResMLP_AdaLN_TCN,
    MAIN_FEATURES, COND_NUM_FEATURES, TEMPORAL_FEATURES,
    WINDOW, WINDOW_OFFSET_LEFT, TARGET, TRAIN_END, VAL_END,
    DIM, N_BLOCKS, DIM_TEMPORAL,
    load_data, build, gaussian_nll,
)

WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50          # 작은 batch는 epoch 줄임
PATIENCE = 8
SEED = 42

CONFIGS = [
    {"name": "batch32_NLL",      "batch": 32,  "lr": 5e-4, "alpha": None},
    {"name": "batch32_MAE+02NLL","batch": 32,  "lr": 5e-4, "alpha": 0.20},
    {"name": "batch64_NLL",      "batch": 64,  "lr": 7e-4, "alpha": None},
    {"name": "batch64_MAE+02NLL","batch": 64,  "lr": 7e-4, "alpha": 0.20},
    {"name": "batch256_NLL",     "batch": 256, "lr": 1e-3, "alpha": None},
    {"name": "batch256_MAE+02NLL","batch": 256,"lr": 1e-3, "alpha": 0.20},
]


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_combined(y, mu, sigma, alpha):
    mae = (y - mu).abs().mean()
    nll = gaussian_nll(y, mu, sigma).mean()
    if alpha is None:
        return nll, mae.item(), nll.item()
    return mae + alpha * nll, mae.item(), nll.item()


def train_one(name, batch, lr, alpha, X_m_tr, X_c_tr, X_t_tr, Y_tr,
              X_m_v, X_c_v, X_t_v, Y_v, n_main, n_cond, n_temp, device):
    print(f"\n{'='*70}")
    print(f"Config: {name}  batch={batch}, lr={lr}, alpha={alpha}")
    print(f"{'='*70}")
    set_all_seeds(SEED)
    model = ResMLP_AdaLN_TCN(n_main, n_cond, n_temp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e3:.1f}k", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_m_tr), torch.from_numpy(X_c_tr),
                             torch.from_numpy(X_t_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_m_v), torch.from_numpy(X_c_v),
                           torch.from_numpy(X_t_v), torch.from_numpy(Y_v))
    g_loader = torch.Generator(); g_loader.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, generator=g_loader)
    val_loader = DataLoader(val_ds, batch_size=2048)   # eval batch 크게

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        for xm, xc, xt, y in train_loader:
            xm, xc, xt, y = xm.to(device), xc.to(device), xt.to(device), y.to(device)
            mu, sig = model(xm, xc, xt)
            loss, _, _ = loss_combined(y, mu, sig, alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

        model.eval()
        v_mae = 0; v_nll = 0; nv = 0
        with torch.no_grad():
            for xm, xc, xt, y in val_loader:
                xm, xc, xt, y = xm.to(device), xc.to(device), xt.to(device), y.to(device)
                mu, sig = model(xm, xc, xt)
                v_mae += (y - mu).abs().sum().item()
                v_nll += gaussian_nll(y, mu, sig).sum().item()
                nv += len(y)
        v_mae /= nv; v_nll /= nv
        # tracking metric: val MAE (point accuracy 우선)
        primary = v_mae
        m = ""
        if primary < best:
            best = primary
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            bad = 0; m = "★"
        else:
            bad += 1
        if ep % 5 == 0 or m:
            print(f"  ep {ep:>3}: val_mae {v_mae:.4f} val_nll {v_nll:.4f} {m}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


def evaluate(model, X_m, X_c, X_t, Y, meta, device):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_m), torch.from_numpy(X_c),
                       torch.from_numpy(X_t), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=2048)
    mus, sigs = [], []
    with torch.no_grad():
        for xm, xc, xt, _ in loader:
            mu, sig = model(xm.to(device), xc.to(device), xt.to(device))
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)
    cap = meta["site_capacity_kw"].values
    err = np.abs(Y - mu_arr)
    nmae = (err * cap).sum() / cap.sum() * 100
    cov80 = ((Y >= mu_arr - 1.282 * sig_arr) & (Y <= mu_arr + 1.282 * sig_arr)).mean() * 100
    cov95 = ((Y >= mu_arr - 1.96 * sig_arr) & (Y <= mu_arr + 1.96 * sig_arr)).mean() * 100
    nll = (0.5 * np.log(2 * np.pi * sig_arr ** 2) + (Y - mu_arr) ** 2 / (2 * sig_arr ** 2)).mean()

    df_eval = meta.copy(); df_eval["mu"] = mu_arr; df_eval["err"] = err
    df_eval["pred_kwh"] = mu_arr * df_eval["site_capacity_kw"]
    df_eval["actual_kwh"] = Y * df_eval["site_capacity_kw"]
    port = df_eval.groupby("datetime_kst", as_index=False).agg(
        pred=("pred_kwh", "sum"), actual=("actual_kwh", "sum"),
        cap=("site_capacity_kw", "sum"))
    pnmae = (port["pred"] - port["actual"]).abs().sum() / port["cap"].sum() * 100
    return {"nmae": nmae, "pnmae": pnmae, "cov80": cov80, "cov95": cov95, "nll": float(nll)}


def main():
    print("=" * 70, flush=True)
    print("ResMLP+TCN batch-sensitivity sweep (6 configs)", flush=True)
    print("=" * 70, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    df = load_data()
    all_sites = sorted(df["site"].unique())
    print(f"\n[Build] sequences (window={WINDOW}, centered)...", flush=True)
    X_m_tr, X_c_tr, X_t_tr, Y_tr, M_tr, scalers = build(
        df, all_sites, df.datetime_kst.min(), TRAIN_END)
    X_m_v, X_c_v, X_t_v, Y_v, M_v, _ = build(df, all_sites, TRAIN_END, VAL_END, scalers=scalers)
    X_m_te, X_c_te, X_t_te, Y_te, M_te, _ = build(
        df, all_sites, VAL_END, df.datetime_kst.max() + pd.Timedelta(hours=1), scalers=scalers)
    print(f"  train {len(Y_tr):,}, val {len(Y_v):,}, test {len(Y_te):,}", flush=True)
    n_main = X_m_tr.shape[1]; n_cond = X_c_tr.shape[1]; n_temp = X_t_tr.shape[2]

    results = []
    for cfg in CONFIGS:
        model = train_one(cfg["name"], cfg["batch"], cfg["lr"], cfg["alpha"],
                          X_m_tr, X_c_tr, X_t_tr, Y_tr,
                          X_m_v, X_c_v, X_t_v, Y_v,
                          n_main, n_cond, n_temp, device)
        m = evaluate(model, X_m_te, X_c_te, X_t_te, Y_te, M_te, device)
        print(f"\n  → {cfg['name']}: NMAE {m['nmae']:.3f}% / Port {m['pnmae']:.3f}% / "
              f"Cov80 {m['cov80']:.1f}% / NLL {m['nll']:.4f}", flush=True)
        results.append({**cfg, **m})

    print("\n" + "=" * 80, flush=True)
    print("SWEEP 결과 (vs baseline)", flush=True)
    print("=" * 80, flush=True)
    print(f"  {'Config':<28} {'batch':>6} {'lr':>7} {'alpha':>6} {'NMAE %':>8} "
          f"{'Port %':>8} {'Cov80':>7} {'NLL':>8}", flush=True)
    print("-" * 90, flush=True)
    print(f"  {'ResMLP+AdaLN baseline':<28} {4096:>6} {1e-3:>7.0e} {'-':>6} "
          f"{5.12:>7.2f}% {4.13:>7.2f}% {82.6:>6.1f}% {' ?':>8}", flush=True)
    print(f"  {'ResMLP+TCN (b=4096, NLL)':<28} {4096:>6} {1e-3:>7.0e} {'-':>6} "
          f"{5.21:>7.2f}% {4.25:>7.2f}% {81.5:>6.1f}% {-1.139:>8.4f}", flush=True)
    print("-" * 90, flush=True)
    for r in results:
        a = r["alpha"] if r["alpha"] else "-"
        print(f"  {'ResMLP+TCN ' + r['name']:<28} {r['batch']:>6} {r['lr']:>7.0e} "
              f"{a if isinstance(a, str) else f'{a:.2f}':>6} "
              f"{r['nmae']:>7.2f}% {r['pnmae']:>7.2f}% {r['cov80']:>6.1f}% {r['nll']:>8.4f}", flush=True)

    print()
    best = min(results, key=lambda r: r["nmae"])
    base_nmae = 5.12
    print(f"Best TCN config: {best['name']} → NMAE {best['nmae']:.3f}%")
    if best["nmae"] < base_nmae:
        print(f"  ✅ Beat baseline ({base_nmae:.2f}%) by {base_nmae - best['nmae']:.3f}%p")
    else:
        print(f"  ❌ NOT beat baseline ({base_nmae:.2f}%) — gap {best['nmae'] - base_nmae:+.3f}%p")
        print(f"  → Hybrid 정리 권장")


if __name__ == "__main__":
    main()
