"""Step A — Winner reproducibility check.

Architecture: ResMLP+AdaLN + Shallow TCN hybrid (구조 변경 없음).

Locked hyperparams:
  batch=64, lr=7e-4, wd=1e-4, AdamW, CosineAnnealingLR(T_max=100),
  max_epochs=100, patience=10, grad_clip=1.0,
  loss = MAE + 0.2 * GaussianNLL.

Seeds: 42, 123, 7 → mean±std report.
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
    ResMLP_AdaLN_TCN,
    MAIN_FEATURES, COND_NUM_FEATURES, TEMPORAL_FEATURES,
    WINDOW, WINDOW_OFFSET_LEFT, TARGET, TRAIN_END, VAL_END,
    DIM, N_BLOCKS, DIM_TEMPORAL,
    load_data, build, gaussian_nll,
)

# === Locked hyperparams ===
BATCH_SIZE = 64
LR = 7e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0
ALPHA_NLL = 0.20

SEEDS = [42, 123, 7]


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_combined(y, mu, sigma):
    """MAE + α·NLL (α = 0.20)."""
    mae = (y - mu).abs().mean()
    nll = gaussian_nll(y, mu, sigma).mean()
    return mae + ALPHA_NLL * nll, mae.item(), nll.item()


def train_one_seed(seed, X_m_tr, X_c_tr, X_t_tr, Y_tr,
                    X_m_v, X_c_v, X_t_v, Y_v,
                    n_main, n_cond, n_temp, device):
    print(f"\n{'='*70}", flush=True)
    print(f"Seed {seed} — Winner repro", flush=True)
    print(f"{'='*70}", flush=True)
    set_all_seeds(seed)

    model = ResMLP_AdaLN_TCN(n_main, n_cond, n_temp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e3:.1f}k", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_m_tr), torch.from_numpy(X_c_tr),
                             torch.from_numpy(X_t_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_m_v), torch.from_numpy(X_c_v),
                           torch.from_numpy(X_t_v), torch.from_numpy(Y_v))
    g_loader = torch.Generator(); g_loader.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_loader)
    val_loader = DataLoader(val_ds, batch_size=2048)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        for xm, xc, xt, y in train_loader:
            xm, xc, xt, y = xm.to(device), xc.to(device), xt.to(device), y.to(device)
            mu, sig = model(xm, xc, xt)
            loss, _, _ = loss_combined(y, mu, sig)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
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
        # tracking metric: val_mae (point accuracy 우선, sweep과 동일)
        primary = v_mae
        m = ""
        if primary < best:
            best = primary
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            bad = 0; m = "★"
        else:
            bad += 1
        if ep % 10 == 0 or m:
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
    print("Step A — ResMLP+AdaLN+TCN winner reproducibility (3 seeds)", flush=True)
    print(f"  Locked: batch={BATCH_SIZE}, lr={LR}, wd={WEIGHT_DECAY}, "
          f"max_epochs={MAX_EPOCHS}, patience={PATIENCE}, "
          f"loss=MAE+{ALPHA_NLL}·NLL", flush=True)
    print("=" * 70, flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    for seed in SEEDS:
        model = train_one_seed(seed, X_m_tr, X_c_tr, X_t_tr, Y_tr,
                                X_m_v, X_c_v, X_t_v, Y_v,
                                n_main, n_cond, n_temp, device)
        m = evaluate(model, X_m_te, X_c_te, X_t_te, Y_te, M_te, device)
        print(f"\n  → seed={seed}: NMAE {m['nmae']:.3f}% / Port {m['pnmae']:.3f}% / "
              f"Cov80 {m['cov80']:.1f}% / Cov95 {m['cov95']:.1f}% / NLL {m['nll']:.4f}",
              flush=True)
        results.append({"seed": seed, **m})

        out_dir = ROOT / f"pv/experiments/resmlp_tcn_winner/seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "metrics": m,
                    "scalers": scalers}, out_dir / "model.pt")

    # Aggregate
    res_df = pd.DataFrame(results)
    print("\n" + "=" * 70, flush=True)
    print("Step A 결과 (3 seeds)", flush=True)
    print("=" * 70, flush=True)
    print(res_df.round(3).to_string(index=False), flush=True)

    print("\n  Mean ± Std:", flush=True)
    for col in ["nmae", "pnmae", "cov80", "cov95", "nll"]:
        vals = res_df[col].values
        print(f"    {col:<10} {vals.mean():>7.3f} ± {vals.std(ddof=1):>5.3f}", flush=True)

    # Output for Step B reference
    out_path = ROOT / "pv/experiments/resmlp_tcn_winner/repro_summary.csv"
    res_df.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}", flush=True)

    print(f"\n  Reference baseline: ResMLP+AdaLN single-run = NMAE 5.12% / Port 4.13%", flush=True)
    print(f"  Reference batch=64 winner (sweep, seed 42): NMAE 5.072% / Port 4.165%", flush=True)


if __name__ == "__main__":
    main()
