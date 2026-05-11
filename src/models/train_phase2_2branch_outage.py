"""Phase 2 with integrated outage-probability head (실험).

기존 train_phase2_2branch.py 와 동일 구조 + 추가:
  - outage probability head: combined → linear → sigmoid (per-lead)
  - outage pseudo-label: data/processed/phase2_outage_labels.parquet
  - loss = forecast_loss + λ_out * BCE(p_outage, y_out, pos_weight)
  - inference: mu_final = (1 - p_outage) * mu_phase2 (soft suppression)

OUT_DIR: pv/experiments/phase2_2branch_outage_lo{X}_pw{Y}/
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/models"))

from train_phase2_intraday_tcn import (
    set_all_seeds, normalize_past, normalize_future,
    load_phase1_ensemble,
    H_PAST, ALL_SITES,
    BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, GRAD_CLIP, ALPHA_NLL,
    PHASE1_DIR, SEEDS_PHASE1, PHASE2_VAL_FROM,
)
from train_phase2_2branch import (
    IntradayTCN_2Branch as Base2Branch,
    SamplesDataset,
)

L_FUTURE = 12  # 일몰까지
OUT_DIR_BASE = ROOT / "pv/experiments/phase2_2branch_outage"
LABEL_PATH = ROOT / "data/processed/phase2_outage_labels.parquet"


# ============== Sample construction (with outage label) ==============

def build_samples_with_outage(phase1_df, weather_df, outage_lookup, label):
    """v1과 동일한 구조 + outage_label per lead."""
    src_cols = ['datetime_kst','site','dsr_mean','dc10Tca','ta','hm','ws','zenith_center']
    weather_df = weather_df[src_cols].copy()
    weather_df['datetime_kst'] = pd.to_datetime(weather_df['datetime_kst'])

    df = phase1_df.copy()
    df['datetime_kst'] = pd.to_datetime(df['datetime_kst'])
    df = df.merge(weather_df, on=['datetime_kst','site'], how='left')
    df['hour']  = df.datetime_kst.dt.hour
    df['month'] = df.datetime_kst.dt.month
    df['date']  = df.datetime_kst.dt.normalize()
    df = df.sort_values(['site','datetime_kst']).reset_index(drop=True)

    site_to_idx = {s: i for i, s in enumerate(ALL_SITES)}

    rows = []
    for (date, site), g in df.groupby(['date','site']):
        g = g.set_index('hour')
        site_idx = site_to_idx[site]
        for issue_hour in range(7, 19):
            past_hours = list(range(issue_hour - (H_PAST - 1), issue_hour + 1))
            past_feat = np.zeros((H_PAST, 8), dtype=np.float32)
            past_mask = np.zeros(H_PAST, dtype=np.float32)
            past_cf = []
            past_resid = []
            for i, h in enumerate(past_hours):
                if h in g.index:
                    r = g.loc[h]
                    feats = [r['cf'], r['cf'] - r['mu_mean'],
                             r['dsr_mean'], r['dc10Tca'], r['ta'], r['hm'], r['ws'], r['zenith_center']]
                    # 모든 weather 피처가 non-null이어야 mask=1
                    if all(pd.notna(v) for v in feats):
                        past_feat[i] = feats
                        past_mask[i] = 1.0
                        past_cf.append(float(r['cf']))
                        past_resid.append(float(r['cf'] - r['mu_mean']))
            if past_mask[-1] < 1.0:
                continue
            recent_cf = past_cf[-3:] if len(past_cf) >= 3 else past_cf
            if len(recent_cf) >= 2:
                drop_now_3h = max(0.0, -float(np.diff(recent_cf).min()))
            else:
                drop_now_3h = 0.0
            recent_resid = past_resid[-3:] if len(past_resid) >= 3 else past_resid
            abs_realized_gap_3h = float(abs(np.mean(recent_resid))) if recent_resid else 0.0

            # === Future + outage label ===
            future_feat = np.zeros((L_FUTURE, 5), dtype=np.float32)
            future_mask = np.zeros(L_FUTURE, dtype=np.float32)
            future_mu_p1 = np.zeros(L_FUTURE, dtype=np.float32)
            future_sigma_p1 = np.ones(L_FUTURE, dtype=np.float32)
            future_cf = np.zeros(L_FUTURE, dtype=np.float32)
            future_cap = np.zeros(L_FUTURE, dtype=np.float32)
            future_dt = [pd.NaT] * L_FUTURE
            future_dc = np.full(L_FUTURE, np.nan, dtype=np.float32)
            future_outage = np.zeros(L_FUTURE, dtype=np.float32)
            for i, h in enumerate(range(issue_hour + 1, issue_hour + 1 + L_FUTURE)):
                if h in g.index:
                    r = g.loc[h]
                    fut_feats = [r['mu_mean'], r['sigma_total'],
                                 r['dsr_mean'], r['dc10Tca'], r['zenith_center']]
                    if all(pd.notna(v) for v in fut_feats) and pd.notna(r.get('cf', np.nan)):
                        future_feat[i] = fut_feats
                        future_mask[i] = 1.0
                        future_mu_p1[i] = r['mu_mean']
                        future_sigma_p1[i] = max(float(r['sigma_total']), 0.03)
                        future_cf[i] = r['cf']
                        future_cap[i] = r['site_capacity_kw']
                        future_dt[i] = r['datetime_kst']
                        future_dc[i] = r['dc10Tca']
                        future_outage[i] = float(outage_lookup.get((r['datetime_kst'], site), 0))
            if future_mask.sum() == 0:
                continue

            sigma_p1_lead1 = float(future_sigma_p1[0])
            gap_x_sigma = abs_realized_gap_3h * sigma_p1_lead1
            gate_input = np.array([drop_now_3h, abs_realized_gap_3h, gap_x_sigma], dtype=np.float32)
            ih_sin = np.sin(2*np.pi*issue_hour/24); ih_cos = np.cos(2*np.pi*issue_hour/24)
            month = int(g.iloc[0]['month'])
            m_sin = np.sin(2*np.pi*month/12); m_cos = np.cos(2*np.pi*month/12)
            site_oh = np.zeros(len(ALL_SITES), dtype=np.float32); site_oh[site_idx] = 1.0
            static = np.concatenate([site_oh, [ih_sin, ih_cos, m_sin, m_cos]]).astype(np.float32)

            rows.append({
                'date': date, 'site': site, 'issue_hour': issue_hour,
                'past_feat': past_feat, 'past_mask': past_mask,
                'future_feat': future_feat, 'future_mask': future_mask,
                'future_mu_p1': future_mu_p1, 'future_sigma_p1': future_sigma_p1,
                'future_cf': future_cf, 'future_cap': future_cap,
                'future_dc10Tca': future_dc, 'future_dt': future_dt,
                'future_outage': future_outage,
                'static': static, 'gate_input_raw': gate_input,
            })
    pos_count = sum(int(s['future_outage'].sum()) for s in rows)
    print(f"  [{label}] samples: {len(rows):,}, outage labels (lead-level): {pos_count}")
    return rows


# ============== Model with outage head ==============

class IntradayTCN_2Branch_Outage(Base2Branch):
    def __init__(self, n_past=8, n_future=5, n_static=12, n_horizon=L_FUTURE, n_gate=3,
                 dim_past=32, dim_future=16, dim_head=64,
                 event_gain=1.0, gate_temp=1.0, gate_threshold=0.0):
        super().__init__(n_past=n_past, n_future=n_future, n_static=n_static,
                         n_horizon=n_horizon, n_gate=n_gate,
                         dim_past=dim_past, dim_future=dim_future, dim_head=dim_head,
                         event_gain=event_gain, gate_temp=gate_temp,
                         gate_threshold=gate_threshold)
        combined_dim = dim_past + dim_future*n_horizon + n_static
        # outage probability head — bias init to negative for low base rate
        self.head_outage = nn.Sequential(
            nn.Linear(combined_dim, dim_head), nn.GELU(),
            nn.Linear(dim_head, n_horizon),
        )
        with torch.no_grad():
            self.head_outage[-1].bias.fill_(-5.0)  # sigmoid(-5) ≈ 0.007 (base rate ~0.2%)

    def forward(self, past_feat, past_mask, future_feat, future_mask, static, gate_in):
        past_in  = torch.cat([past_feat, past_mask.unsqueeze(-1)], dim=-1)
        past_emb = self.past_tcn(past_in.transpose(1, 2)).squeeze(-1)
        future_in  = torch.cat([future_feat, future_mask.unsqueeze(-1)], dim=-1)
        future_emb = self.future_mlp(future_in)
        future_flat = future_emb.reshape(future_emb.size(0), -1)
        combined = torch.cat([past_emb, future_flat, static], dim=-1)
        delta_base  = self.head_base(combined)
        delta_event = self.head_event(combined)
        gate_logits = self.gate(gate_in)
        g_raw = torch.sigmoid(gate_logits / self.gate_temp)
        gate = torch.clamp((g_raw - self.gate_threshold) / (1.0 - self.gate_threshold), min=0.0) \
               if self.gate_threshold > 0 else g_raw
        delta_total = delta_base + self.event_gain * gate * delta_event
        outage_logit = self.head_outage(combined)
        return delta_total, delta_base, delta_event, gate, outage_logit


# ============== Loss ==============

def compute_loss_outage(cf, mu_p1, sigma_p1, delta_total, mask, outage_logit, outage_y,
                        lambda_out=0.10, pos_weight=20.0):
    # 안정성: sigma 최소값 0.03 (이전 0.001), delta clamp
    sigma_safe = sigma_p1.clamp(min=0.03)
    delta_safe = delta_total.clamp(min=-1.0, max=1.0)

    mu_new = mu_p1 + delta_safe
    err = (cf - mu_new).abs()
    mae = (err * mask).sum() / (mask.sum() + 1e-8)
    nll_per = 0.5 * (torch.log(2 * np.pi * sigma_safe ** 2) + (cf - mu_new) ** 2 / sigma_safe ** 2)
    nll = (nll_per * mask).sum() / (mask.sum() + 1e-8)
    forecast_loss = mae + ALPHA_NLL * nll

    logit_safe = outage_logit.clamp(min=-15.0, max=15.0)
    pw = torch.tensor(pos_weight, device=outage_logit.device, dtype=outage_logit.dtype)
    bce_per = F.binary_cross_entropy_with_logits(logit_safe, outage_y, pos_weight=pw, reduction='none')
    bce = (bce_per * mask).sum() / (mask.sum() + 1e-8)
    total = forecast_loss + lambda_out * bce
    return total, mae.detach(), bce.detach()


# ============== Tensor / Dataset ==============

def to_tensors_outage(samples, gate_scalers=None):
    keys = ['past_feat','past_mask','future_feat','future_mask',
            'future_mu_p1','future_sigma_p1','future_cf','future_cap','future_outage','static']
    out = {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in keys}
    raw = np.stack([s['gate_input_raw'] for s in samples])
    if gate_scalers is None:
        mu = raw.mean(axis=0); sd = raw.std(axis=0) + 1e-6
        gate_scalers = (mu, sd)
    else:
        mu, sd = gate_scalers
    out['gate_input'] = torch.from_numpy(((raw - mu)/sd).astype(np.float32))
    return out, gate_scalers


class SamplesDatasetOutage(torch.utils.data.Dataset):
    KEYS = ['past_feat','past_mask','future_feat','future_mask',
            'future_mu_p1','future_sigma_p1','future_cf','future_cap','future_outage',
            'static','gate_input']
    def __init__(self, t): self.t = t; self.n = self.t['past_feat'].shape[0]
    def __len__(self): return self.n
    def __getitem__(self, i): return tuple(self.t[k][i] for k in self.KEYS)


# ============== Train ==============

def train_one(seed, train_samples, val_samples, device,
              event_gain=2.0, lambda_out=0.15, pos_weight=300.0):
    print(f"\n[Train seed={seed}] outage-aware (gain={event_gain}, λ_out={lambda_out}, pw={pos_weight})")
    set_all_seeds(seed)
    for s in train_samples + val_samples:
        s['past_feat']   = normalize_past(s['past_feat'][None])[0]
        s['future_feat'] = normalize_future(s['future_feat'][None])[0]
    tr_t, gate_sc = to_tensors_outage(train_samples)
    vl_t, _       = to_tensors_outage(val_samples, gate_scalers=gate_sc)

    train_ds = SamplesDatasetOutage(tr_t); val_ds = SamplesDatasetOutage(vl_t)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader   = DataLoader(val_ds, batch_size=2048)

    model = IntradayTCN_2Branch_Outage(event_gain=event_gain, gate_temp=1.0,
                                         gate_threshold=0.0, n_horizon=L_FUTURE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            pf, pm, ff, fm, mu_p1, sig_p1, cf, _cap, y_out, st, gi = [b.to(device) for b in batch]
            delta, _, _, _, out_logit = model(pf, pm, ff, fm, st, gi)
            loss, _, _ = compute_loss_outage(cf, mu_p1, sig_p1, delta, fm, out_logit, y_out,
                                              lambda_out=lambda_out, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()
        model.eval()
        v_mae_sum = 0.0; v_n = 0.0; out_means = []
        with torch.no_grad():
            for batch in val_loader:
                pf, pm, ff, fm, mu_p1, sig_p1, cf, _cap, y_out, st, gi = [b.to(device) for b in batch]
                delta, _, _, _, out_logit = model(pf, pm, ff, fm, st, gi)
                err = (cf - (mu_p1 + delta)).abs() * fm
                v_mae_sum += err.sum().item(); v_n += fm.sum().item()
                out_means.append(torch.sigmoid(out_logit).mean().item())
        v_mae = v_mae_sum / max(v_n, 1)
        if v_mae < best:
            best = v_mae; bad = 0
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            mark = "★"
        else:
            bad += 1; mark = ""
        if ep % 5 == 0 or mark:
            print(f"  ep {ep:>3}: val_mae {v_mae:.4f}  out_p_mean {np.mean(out_means):.4f}  {mark}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model, gate_sc


def predict(model, samples, gate_sc, device):
    for s in samples:
        s['past_feat'] = normalize_past(s['past_feat'][None])[0]
        s['future_feat'] = normalize_future(s['future_feat'][None])[0]
    t, _ = to_tensors_outage(samples, gate_scalers=gate_sc)
    ds = SamplesDatasetOutage(t)
    loader = DataLoader(ds, batch_size=2048)
    deltas, gates, out_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pf, pm, ff, fm, mu_p1, sig_p1, cf, cap, y_out, st, gi = [b.to(device) for b in batch]
            delta, _, _, gate, out_logit = model(pf, pm, ff, fm, st, gi)
            deltas.append(delta.cpu().numpy())
            gates.append(gate.cpu().numpy())
            out_probs.append(torch.sigmoid(out_logit).cpu().numpy())
    return np.concatenate(deltas), np.concatenate(gates), np.concatenate(out_probs)


def collect_pred_df(samples, deltas, gates, out_probs):
    rows = []
    for idx, s in enumerate(samples):
        for k in range(L_FUTURE):
            if s['future_mask'][k] < 1.0: continue
            target_dt = s['future_dt'][k]
            mu_p1 = float(s['future_mu_p1'][k])
            delta = float(deltas[idx, k])
            mu_p2 = max(0.0, mu_p1 + delta)
            p_out = float(out_probs[idx, k])
            mu_final = (1.0 - p_out) * mu_p2
            rows.append({
                'date': s['date'], 'site': s['site'], 'issue_hour': s['issue_hour'],
                'lead': k+1, 'target_dt': target_dt,
                'cf': float(s['future_cf'][k]),
                'cap': float(s['future_cap'][k]),
                'dc10Tca_target': float(s['future_dc10Tca'][k]),
                'mu_phase1': mu_p1, 'sigma_phase1': float(s['future_sigma_p1'][k]),
                'delta': delta, 'gate': float(gates[idx, k]),
                'mu_phase2': mu_p2,
                'p_outage': p_out,
                'mu_final': mu_final,
                'is_outage_label': float(s['future_outage'][k]),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gain", type=float, default=2.0)
    ap.add_argument("--lambda-out", type=float, default=0.15)
    ap.add_argument("--pos-weight", type=float, default=300.0)
    args = ap.parse_args()

    suffix = f"lo{int(args.lambda_out*100)}_pw{int(args.pos_weight)}"
    out_dir = OUT_DIR_BASE.parent / f"phase2_2branch_outage_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Phase 2 outage-aware | seed={args.seed} | gain={args.gain} | "
          f"λ_out={args.lambda_out} | pw={args.pos_weight}")
    print(f"  OUT: {out_dir}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # data
    p1_val  = load_phase1_ensemble("val")
    p1_test = load_phase1_ensemble("test")
    p1_val['datetime_kst'] = pd.to_datetime(p1_val['datetime_kst'])
    p1_test['datetime_kst'] = pd.to_datetime(p1_test['datetime_kst'])
    weather = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    label_df = pd.read_parquet(LABEL_PATH)
    label_df['datetime_kst'] = pd.to_datetime(label_df.datetime_kst)
    outage_lookup = dict(zip(zip(label_df.datetime_kst, label_df.site), label_df.is_outage))
    print(f"outage labels loaded: {sum(outage_lookup.values())} positive")

    # 원본 train_phase2_2branch.py 와 동일 split: PHASE2_VAL_FROM = 2024-10-16
    p1_val_train = p1_val[p1_val.datetime_kst < PHASE2_VAL_FROM]
    p1_val_eval  = p1_val[p1_val.datetime_kst >= PHASE2_VAL_FROM]

    print("\n[build train samples]")
    tr = build_samples_with_outage(p1_val_train, weather, outage_lookup, label="train")
    print("\n[build val samples]")
    vl = build_samples_with_outage(p1_val_eval, weather, outage_lookup, label="val")
    print("\n[build test samples]")
    test_samples = build_samples_with_outage(p1_test, weather, outage_lookup, label="test")

    # train
    model, gate_sc = train_one(args.seed, tr, vl, device,
                                event_gain=args.gain,
                                lambda_out=args.lambda_out,
                                pos_weight=args.pos_weight)

    # predict on test
    print("\n[predict — test 2025]")
    deltas, gates, out_probs = predict(model, test_samples, gate_sc, device)
    pred_df = collect_pred_df(test_samples, deltas, gates, out_probs)
    pred_df.to_parquet(out_dir / f"test_predictions_seed{args.seed}.parquet", index=False)
    print(f"  saved: test_predictions_seed{args.seed}.parquet ({len(pred_df):,} rows)")

    # quick metrics
    p1_nmae   = ((pred_df.cf - pred_df.mu_phase1).abs() * pred_df.cap).sum() / pred_df.cap.sum() * 100
    p2_nmae   = ((pred_df.cf - pred_df.mu_phase2).abs() * pred_df.cap).sum() / pred_df.cap.sum() * 100
    fin_nmae  = ((pred_df.cf - pred_df.mu_final).abs()  * pred_df.cap).sum() / pred_df.cap.sum() * 100
    print(f"\n  P1 NMAE: {p1_nmae:.3f}%   P2 NMAE: {p2_nmae:.3f}%   P2*soft: {fin_nmae:.3f}%")
    print(f"  outage prob — mean: {pred_df.p_outage.mean():.4f}, "
          f"median: {pred_df.p_outage.median():.4f}, max: {pred_df.p_outage.max():.4f}")


if __name__ == "__main__":
    main()
