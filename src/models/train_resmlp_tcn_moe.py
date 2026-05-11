"""Step B — ResMLP+AdaLN+TCN with K=3 expert heads + soft gating.

Architecture:
  ResMLP+AdaLN backbone + TCN branch (Step A 동일, 유지)
  → fusion MLP → h_fused (B, 64)

  *변경*: single (μ_head, σ_head) → K=3 expert heads + gating

  Expert k (k=1..3):
    Linear(64 → 1) for μ_k
    Linear(64 → 1) for log_σ_k → softplus → σ_k

  Gating:
    Linear(64 → 64) → GELU → Linear(64 → 3) → softmax → (w_1, w_2, w_3)

  Soft combination:
    μ = Σ_k w_k · μ_k
    σ² = Σ_k w_k (σ_k² + μ_k²) - μ²   (mixture variance)

학습 조건 (Step A 동일):
  batch=64, lr=7e-4, wd=1e-4, AdamW, CosineAnnealingLR(T_max=100),
  max_epochs=100, patience=10, grad_clip=1.0,
  loss = MAE + 0.2 * GaussianNLL on (μ_mix, σ_mix).

Diagnostic (gating collapse 검증):
  - 각 head 평균 weight (이상적 1/3 ≈ 0.333)
  - Weight entropy (max log 3 ≈ 1.10, collapsed → 낮음)
  - Dominance: max weight > 0.8 sample 비율
  - Site / hour / cloud별 conditional weight 분포

seed=42 1회 먼저, condition 만족 시 123/7 확장.
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
    AdaLNBlock,
    MAIN_FEATURES, COND_NUM_FEATURES, TEMPORAL_FEATURES,
    WINDOW, WINDOW_OFFSET_LEFT, TARGET, TRAIN_END, VAL_END,
    DIM, N_BLOCKS, DIM_TEMPORAL,
    load_data, build, gaussian_nll,
)

# Step A locked hyperparams
BATCH_SIZE = 64
LR = 7e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0
ALPHA_NLL = 0.20
N_EXPERTS = 3
SEED = 42


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ========== Model with K experts + gating ==========

class ResMLP_AdaLN_TCN_MoE(nn.Module):
    def __init__(self, n_main, n_cond, n_temporal_feat,
                 dim=DIM, n_blocks=N_BLOCKS, dim_temporal=DIM_TEMPORAL,
                 n_experts=N_EXPERTS):
        super().__init__()
        # ===== Static branch =====
        self.input_proj = nn.Linear(n_main, dim)
        self.blocks = nn.ModuleList([AdaLNBlock(dim, n_cond) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.cond_final = nn.Linear(n_cond, 2 * dim)

        # ===== Temporal branch =====
        self.tcn = nn.Sequential(
            nn.Conv1d(n_temporal_feat, dim_temporal, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(dim_temporal, dim_temporal, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ===== Fusion =====
        self.fusion = nn.Sequential(
            nn.Linear(dim + dim_temporal, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
        )

        # ===== K expert heads =====
        self.n_experts = n_experts
        self.mu_heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(n_experts)])
        self.log_sig_heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(n_experts)])

        # ===== Gating network =====
        self.gate = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, n_experts),
        )

    def forward(self, x_main, x_cond, x_temporal, return_gates=False):
        # Static
        x = self.input_proj(x_main)
        for blk in self.blocks:
            x = blk(x, x_cond)
        h = self.final_norm(x)
        scale, shift = self.cond_final(x_cond).chunk(2, dim=-1)
        h_static = h * (1 + scale) + shift

        # Temporal
        h_temp = self.tcn(x_temporal.transpose(1, 2)).squeeze(-1)

        # Fusion
        h_fused = self.fusion(torch.cat([h_static, h_temp], dim=-1))

        # K experts
        mus = torch.stack([head(h_fused).squeeze(-1) for head in self.mu_heads], dim=-1)   # (B, K)
        sigs = torch.stack(
            [F.softplus(head(h_fused)).squeeze(-1) + 1e-3 for head in self.log_sig_heads],
            dim=-1,
        )   # (B, K)

        # Gating
        gate_logits = self.gate(h_fused)        # (B, K)
        gate_w = F.softmax(gate_logits, dim=-1)  # (B, K)

        # Mixture aggregation
        mu_mix = (gate_w * mus).sum(dim=-1)
        # E[X²] = Σ w_k (σ_k² + μ_k²)
        ex2 = (gate_w * (sigs ** 2 + mus ** 2)).sum(dim=-1)
        var_mix = ex2 - mu_mix ** 2
        var_mix = torch.clamp(var_mix, min=1e-6)
        sigma_mix = torch.sqrt(var_mix)

        if return_gates:
            return mu_mix, sigma_mix, gate_w, mus, sigs
        return mu_mix, sigma_mix


def loss_combined(y, mu, sigma):
    mae = (y - mu).abs().mean()
    nll = gaussian_nll(y, mu, sigma).mean()
    return mae + ALPHA_NLL * nll


# ========== Train ==========

def train(model, train_loader, val_loader, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for xm, xc, xt, y in train_loader:
            xm, xc, xt, y = xm.to(device), xc.to(device), xt.to(device), y.to(device)
            mu, sig = model(xm, xc, xt)
            loss = loss_combined(y, mu, sig)
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


def evaluate_with_gates(model, X_m, X_c, X_t, Y, meta, device):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X_m), torch.from_numpy(X_c),
                       torch.from_numpy(X_t), torch.from_numpy(Y))
    loader = DataLoader(ds, batch_size=2048)
    mus, sigs, gates_all = [], [], []
    with torch.no_grad():
        for xm, xc, xt, _ in loader:
            mu, sig, gate, _, _ = model(xm.to(device), xc.to(device), xt.to(device), return_gates=True)
            mus.append(mu.cpu().numpy()); sigs.append(sig.cpu().numpy())
            gates_all.append(gate.cpu().numpy())
    mu_arr = np.clip(np.concatenate(mus), 0, None)
    sig_arr = np.concatenate(sigs)
    gates = np.concatenate(gates_all)   # (N, K)

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

    metrics = {"nmae": nmae, "pnmae": pnmae, "cov80": cov80, "cov95": cov95, "nll": float(nll)}
    return metrics, gates, df_eval


def diagnose_gating(gates: np.ndarray, df_eval: pd.DataFrame, raw_test_df: pd.DataFrame):
    """gates: (N, K). 각 column별 mean weight / entropy / dominance."""
    K = gates.shape[1]
    mean_w = gates.mean(axis=0)
    # Entropy per sample, then mean
    eps = 1e-12
    ent = -(gates * np.log(gates + eps)).sum(axis=-1).mean()
    max_ent = np.log(K)
    dominance_08 = (gates.max(axis=-1) > 0.8).mean() * 100
    dominance_09 = (gates.max(axis=-1) > 0.9).mean() * 100

    print("\n--- Gating Diagnostic ---", flush=True)
    print(f"  Mean weight per head: {[f'{w:.3f}' for w in mean_w]}  (이상적 {1/K:.3f})", flush=True)
    print(f"  Mean entropy: {ent:.3f} / max {max_ent:.3f} → {ent/max_ent*100:.1f}%", flush=True)
    print(f"  Dominance > 0.8: {dominance_08:.1f}% of samples", flush=True)
    print(f"  Dominance > 0.9: {dominance_09:.1f}% of samples", flush=True)

    collapse = (mean_w.max() > 0.8) or (ent / max_ent < 0.5)
    if collapse:
        print(f"  ⚠️ COLLAPSE 의심 (entropy 비율 {ent/max_ent*100:.1f}% / max weight {mean_w.max():.3f})", flush=True)
    else:
        print(f"  ✅ 정상 (entropy 비율 {ent/max_ent*100:.1f}%, max weight {mean_w.max():.3f})", flush=True)

    # Site-conditional
    df_eval = df_eval.reset_index(drop=True)
    print("\n--- Site별 평균 weight ---", flush=True)
    print(f"  {'site':<14} " + " ".join([f'h{k}' for k in range(K)]), flush=True)
    for s in sorted(df_eval["site"].unique()):
        idx = df_eval.index[df_eval["site"] == s].values
        sw = gates[idx].mean(axis=0)
        print(f"  {s:<14} " + " ".join([f'{w:.3f}' for w in sw]), flush=True)

    # Hour bin
    df_eval["hour"] = pd.to_datetime(df_eval["datetime_kst"]).dt.hour
    df_eval["hour_bin"] = pd.cut(df_eval["hour"], bins=[-1, 9, 13, 17, 24],
                                   labels=["dawn(7-9)", "morning(10-13)", "afternoon(14-17)", "evening(18+)"])
    print("\n--- Hour 구간별 평균 weight ---", flush=True)
    for label, sub in df_eval.groupby("hour_bin", observed=True):
        idx = sub.index.values
        sw = gates[idx].mean(axis=0)
        print(f"  {str(label):<22} (n={len(idx):>5}): " + " ".join([f'{w:.3f}' for w in sw]), flush=True)

    # Cloud level — join from raw test data
    if "dc10Tca" in raw_test_df.columns:
        raw_test_df = raw_test_df.reset_index(drop=True)
        df_eval = df_eval.merge(
            raw_test_df[["datetime_kst", "site", "dc10Tca"]],
            on=["datetime_kst", "site"], how="left",
        )
        df_eval["cloud_bin"] = pd.cut(df_eval["dc10Tca"], bins=[-1, 3, 7, 10.1],
                                       labels=["clear(<3)", "mild(3-7)", "heavy(>=7)"])
        print("\n--- Cloud level별 평균 weight ---", flush=True)
        for label, sub in df_eval.groupby("cloud_bin", observed=True):
            if pd.isna(label):
                continue
            idx = sub.index.values
            sw = gates[idx].mean(axis=0)
            print(f"  {str(label):<14} (n={len(idx):>5}): " + " ".join([f'{w:.3f}' for w in sw]), flush=True)

    return {"mean_weights": mean_w.tolist(), "entropy": float(ent),
            "max_entropy": float(max_ent), "dominance_08": float(dominance_08),
            "dominance_09": float(dominance_09), "collapse_flag": bool(collapse)}


def main():
    print("=" * 70, flush=True)
    print(f"Step B — ResMLP+TCN with K={N_EXPERTS} experts + gating (seed={SEED})", flush=True)
    print(f"  Locked: batch={BATCH_SIZE}, lr={LR}, wd={WEIGHT_DECAY}, "
          f"max_epochs={MAX_EPOCHS}, patience={PATIENCE}, "
          f"loss=MAE+{ALPHA_NLL}·NLL", flush=True)
    print("=" * 70, flush=True)
    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data()
    all_sites = sorted(df["site"].unique())
    raw_test_df = df[df.datetime_kst >= VAL_END].copy()

    print(f"\n[Build] sequences (window={WINDOW}, centered)...", flush=True)
    X_m_tr, X_c_tr, X_t_tr, Y_tr, M_tr, scalers = build(
        df, all_sites, df.datetime_kst.min(), TRAIN_END)
    X_m_v, X_c_v, X_t_v, Y_v, M_v, _ = build(df, all_sites, TRAIN_END, VAL_END, scalers=scalers)
    X_m_te, X_c_te, X_t_te, Y_te, M_te, _ = build(
        df, all_sites, VAL_END, df.datetime_kst.max() + pd.Timedelta(hours=1), scalers=scalers)
    print(f"  train {len(Y_tr):,}, val {len(Y_v):,}, test {len(Y_te):,}", flush=True)

    n_main = X_m_tr.shape[1]; n_cond = X_c_tr.shape[1]; n_temp = X_t_tr.shape[2]
    model = ResMLP_AdaLN_TCN_MoE(n_main, n_cond, n_temp).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] params: {n_params/1e3:.1f}k (3 experts + gate)", flush=True)

    train_ds = TensorDataset(torch.from_numpy(X_m_tr), torch.from_numpy(X_c_tr),
                             torch.from_numpy(X_t_tr), torch.from_numpy(Y_tr))
    val_ds = TensorDataset(torch.from_numpy(X_m_v), torch.from_numpy(X_c_v),
                           torch.from_numpy(X_t_v), torch.from_numpy(Y_v))
    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=2048)

    print("\n[Train]", flush=True)
    model = train(model, train_loader, val_loader, device)

    print("\n[Test eval + gating diagnostic]", flush=True)
    metrics, gates, df_eval = evaluate_with_gates(model, X_m_te, X_c_te, X_t_te, Y_te, M_te, device)
    print(f"\n  → seed={SEED}: NMAE {metrics['nmae']:.3f}% / Port {metrics['pnmae']:.3f}% / "
          f"Cov80 {metrics['cov80']:.1f}% / Cov95 {metrics['cov95']:.1f}% / NLL {metrics['nll']:.4f}",
          flush=True)

    diag = diagnose_gating(gates, df_eval, raw_test_df)

    # Reference
    print("\n" + "=" * 70, flush=True)
    print("비교 (seed=42)", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Model':<32} {'NMAE %':>8} {'Port %':>8} {'Cov80':>7} {'NLL':>8}", flush=True)
    print(f"  {'ResMLP+AdaLN baseline':<32} {5.12:>7.2f}% {4.13:>7.2f}% {82.6:>6.1f}% {' ?':>8}", flush=True)
    print(f"  {'Hybrid TCN winner (seed=42)':<32} {5.151:>7.2f}% {4.174:>7.2f}% {81.2:>6.1f}% {-1.207:>8.4f}", flush=True)
    print(f"  {'MoE 3-expert (seed=42)':<32} {metrics['nmae']:>7.2f}% {metrics['pnmae']:>7.2f}% "
          f"{metrics['cov80']:>6.1f}% {metrics['nll']:>8.4f}", flush=True)

    # Decision
    winner_nmae = 5.151
    if metrics["nmae"] < winner_nmae and not diag["collapse_flag"]:
        print(f"\n  ✅ winner 능가 + collapse 없음 → seed 123, 7 확장 권장", flush=True)
    elif metrics["nmae"] >= winner_nmae:
        print(f"\n  ❌ winner 못 넘음 (Δ {metrics['nmae'] - winner_nmae:+.3f}%p)", flush=True)
    elif diag["collapse_flag"]:
        print(f"\n  ❌ collapse 발생 → 정리 권장", flush=True)

    out_dir = ROOT / f"pv/experiments/resmlp_tcn_moe_seed{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metrics": metrics,
                "gating_diag": diag, "scalers": scalers}, out_dir / "model.pt")
    print(f"\n저장: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
