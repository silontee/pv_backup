"""Phase 2 — 2-branch (base + event) with soft gate.

Architecture:
  shared_encoder = past TCN + future MLP + static
  delta_base    = head_base(shared)
  delta_event   = head_event(shared)
  gate          = sigmoid(MLP(gate_input)) ∈ (0, 1) per-lead
  delta_total   = delta_base + gate · delta_event

Gate inputs (모두 issue time t에서 계산 가능 — Phase 2 결과 미사용):
  1. drop_now_3h          : past 3h actual cf의 최대 하락 폭
  2. abs_realized_gap_3h  : |mean(cf - μ_p1) over past 3h|  (recent residual)
  3. gap_x_sigma          : abs_realized_gap_3h × sigma_phase1_lead1

목적: 평상시는 base branch만 사용 (gate≈0). 급격한 drop / cloud-pass에서 gate↑로 event branch
      활성화 → 추가 correction. 평균 NMAE는 H=6 baseline 유지하면서 top event에서 추가 개선.

설계 디테일:
  - gate 마지막 layer bias = -3 → 초기 sigmoid≈0.05 (event branch 영향 작게 시작)
  - event head 마지막 layer 초기 weight × 0.1 (보수적)
  - loss / opt / scheduler / batch / lr 모두 baseline H=6 그대로

UNIFIED config: batch=64, lr=7e-4, max_ep=50, patience=8, MAE+0.2*NLL.
H_PAST=6, L_FUTURE=3.
"""
import argparse
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

# Reuse base utilities from Phase 2 v1 script
from train_phase2_intraday_tcn import (
    set_all_seeds, normalize_past, normalize_future,
    load_phase1_ensemble,
    H_PAST, L_FUTURE, ALL_SITES,
    BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, GRAD_CLIP, ALPHA_NLL,
    PHASE1_DIR, SEEDS_PHASE1, PHASE2_VAL_FROM,
)

OUT_DIR_BASE = ROOT / "pv/experiments/phase2_2branch"


# ============== Sample construction (gate features 추가) ==============

def build_samples_2branch(phase1_df, weather_df, label):
    """v1과 동일한 구조 + gate_input 3차원 추가."""
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
        for issue_hour in range(7, 19):  # 7..18 inclusive (sunset 19시까지 cover)
            past_hours = list(range(issue_hour - (H_PAST - 1), issue_hour + 1))
            past_feat = np.zeros((H_PAST, 8), dtype=np.float32)
            past_mask = np.zeros(H_PAST, dtype=np.float32)
            past_cf = []
            past_resid = []
            for i, h in enumerate(past_hours):
                if h in g.index:
                    r = g.loc[h]
                    if pd.notna(r['dsr_mean']):
                        past_feat[i] = [
                            r['cf'], r['cf'] - r['mu_mean'],
                            r['dsr_mean'], r['dc10Tca'], r['ta'], r['hm'], r['ws'], r['zenith_center']
                        ]
                        past_mask[i] = 1.0
                        past_cf.append(float(r['cf']))
                        past_resid.append(float(r['cf'] - r['mu_mean']))
            if past_mask[-1] < 1.0:
                continue

            # === Gate features (계산은 raw scale, 정규화는 따로) ===
            # drop_now_3h: 최근 3h 내 cf 가장 큰 하락 폭
            recent_cf = past_cf[-3:] if len(past_cf) >= 3 else past_cf
            if len(recent_cf) >= 2:
                diffs = np.diff(recent_cf)  # cf의 1h 변화량
                drop_now_3h = max(0.0, -float(diffs.min()))  # 가장 큰 하락
            else:
                drop_now_3h = 0.0
            # abs_realized_gap_3h: 최근 3h residual 평균의 절댓값
            recent_resid = past_resid[-3:] if len(past_resid) >= 3 else past_resid
            abs_realized_gap_3h = float(abs(np.mean(recent_resid))) if recent_resid else 0.0

            # === Future ===
            future_feat = np.zeros((L_FUTURE, 5), dtype=np.float32)
            future_mask = np.zeros(L_FUTURE, dtype=np.float32)
            future_mu_p1 = np.zeros(L_FUTURE, dtype=np.float32)
            future_sigma_p1 = np.ones(L_FUTURE, dtype=np.float32)
            future_cf = np.zeros(L_FUTURE, dtype=np.float32)
            future_cap = np.zeros(L_FUTURE, dtype=np.float32)
            future_dt = [pd.NaT] * L_FUTURE
            future_dc = np.full(L_FUTURE, np.nan, dtype=np.float32)
            for i, h in enumerate(range(issue_hour + 1, issue_hour + 1 + L_FUTURE)):
                if h in g.index:
                    r = g.loc[h]
                    if pd.notna(r['dsr_mean']):
                        future_feat[i] = [r['mu_mean'], r['sigma_total'],
                                           r['dsr_mean'], r['dc10Tca'], r['zenith_center']]
                        future_mask[i] = 1.0
                        future_mu_p1[i] = r['mu_mean']
                        future_sigma_p1[i] = max(float(r['sigma_total']), 1e-3)
                        future_cf[i] = r['cf']
                        future_cap[i] = r['site_capacity_kw']
                        future_dt[i] = r['datetime_kst']
                        future_dc[i] = r['dc10Tca']
            if future_mask.sum() == 0:
                continue

            # gate input scalars (raw scale; 학습 전 표준화 적용)
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
                'static': static, 'gate_input_raw': gate_input,
            })
    print(f"  [{label}] samples: {len(rows)}")
    return rows


# ============== 2-branch model ==============

class IntradayTCN_2Branch(nn.Module):
    def __init__(self, n_past=8, n_future=5, n_static=12, n_horizon=L_FUTURE, n_gate=3,
                 dim_past=32, dim_future=16, dim_head=64,
                 event_gain=1.0, gate_temp=1.0, gate_threshold=0.0):
        super().__init__()
        self.event_gain = event_gain
        self.gate_temp = gate_temp           # T: 작을수록 sharper (sigmoid(z/T))
        self.gate_threshold = gate_threshold # τ: 작은 activation 잘라냄
        # shared encoder (same as v1)
        self.past_tcn = nn.Sequential(
            nn.Conv1d(n_past + 1, dim_past, kernel_size=3, padding=1, dilation=1), nn.GELU(),
            nn.Conv1d(dim_past, dim_past, kernel_size=3, padding=2, dilation=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.future_mlp = nn.Sequential(
            nn.Linear(n_future + 1, dim_future), nn.GELU(),
            nn.Linear(dim_future, dim_future), nn.GELU(),
        )
        combined = dim_past + dim_future*n_horizon + n_static
        # base branch head
        self.head_base = nn.Sequential(
            nn.Linear(combined, dim_head), nn.GELU(),
            nn.Linear(dim_head, dim_head), nn.GELU(),
            nn.Linear(dim_head, n_horizon),
        )
        # event branch head
        self.head_event = nn.Sequential(
            nn.Linear(combined, dim_head), nn.GELU(),
            nn.Linear(dim_head, dim_head), nn.GELU(),
            nn.Linear(dim_head, n_horizon),
        )
        # event head 마지막 layer 초기 weight × 0.1 (보수적)
        with torch.no_grad():
            self.head_event[-1].weight.mul_(0.1)
            self.head_event[-1].bias.zero_()
        # gate MLP
        self.gate = nn.Sequential(
            nn.Linear(n_gate, 16), nn.GELU(),
            nn.Linear(16, n_horizon),
        )
        # gate 마지막 layer bias = -3 → sigmoid(-3) ≈ 0.047
        with torch.no_grad():
            self.gate[-1].bias.fill_(-3.0)

    def forward(self, past_feat, past_mask, future_feat, future_mask, static, gate_in):
        past_in = torch.cat([past_feat, past_mask.unsqueeze(-1)], dim=-1)
        past_emb = self.past_tcn(past_in.transpose(1, 2)).squeeze(-1)
        future_in = torch.cat([future_feat, future_mask.unsqueeze(-1)], dim=-1)
        future_emb = self.future_mlp(future_in)
        future_flat = future_emb.reshape(future_emb.size(0), -1)
        combined = torch.cat([past_emb, future_flat, static], dim=-1)
        delta_base  = self.head_base(combined)
        delta_event = self.head_event(combined)
        gate_logits = self.gate(gate_in)
        g_raw = torch.sigmoid(gate_logits / self.gate_temp)
        if self.gate_threshold > 0:
            # clamp(min=0) of (g_raw - τ) / (1 - τ)  → 작은 activation 0으로 잘라냄
            gate = torch.clamp((g_raw - self.gate_threshold) / (1.0 - self.gate_threshold), min=0.0)
        else:
            gate = g_raw
        delta_total = delta_base + self.event_gain * gate * delta_event
        return delta_total, delta_base, delta_event, gate


# ============== Loss / Tensors ==============

def compute_loss(cf, mu_p1, sigma_p1, delta_total, mask):
    mu_new = mu_p1 + delta_total
    err = (cf - mu_new).abs()
    mae = (err * mask).sum() / (mask.sum() + 1e-8)
    nll_per = 0.5 * (torch.log(2 * np.pi * sigma_p1 ** 2) + (cf - mu_new) ** 2 / sigma_p1 ** 2)
    nll = (nll_per * mask).sum() / (mask.sum() + 1e-8)
    return mae + ALPHA_NLL * nll, mae.detach(), nll.detach()


def to_tensors(samples, gate_scalers=None):
    """gate_input은 train set 통계로 z-score 표준화."""
    keys = ['past_feat','past_mask','future_feat','future_mask',
            'future_mu_p1','future_sigma_p1','future_cf','future_cap','static']
    out = {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in keys}
    raw = np.stack([s['gate_input_raw'] for s in samples])
    if gate_scalers is None:
        mu = raw.mean(axis=0); sd = raw.std(axis=0) + 1e-6
        gate_scalers = (mu, sd)
    else:
        mu, sd = gate_scalers
    norm = (raw - mu) / sd
    out['gate_input'] = torch.from_numpy(norm.astype(np.float32))
    return out, gate_scalers


class SamplesDataset(torch.utils.data.Dataset):
    KEYS = ['past_feat','past_mask','future_feat','future_mask',
            'future_mu_p1','future_sigma_p1','future_cf','future_cap','static','gate_input']
    def __init__(self, tensors): self.t = tensors; self.n = self.t['past_feat'].shape[0]
    def __len__(self): return self.n
    def __getitem__(self, i): return tuple(self.t[k][i] for k in self.KEYS)


# ============== Train ==============

def train_one(seed, train_samples, val_samples, device,
              event_gain=1.0, gate_temp=1.0, gate_threshold=0.0):
    print(f"\n[Train seed={seed}] 2-branch gain={event_gain} T={gate_temp} τ={gate_threshold}")
    set_all_seeds(seed)
    # normalize past/future scalars
    for s in train_samples + val_samples:
        s['past_feat'] = normalize_past(s['past_feat'][None])[0]
        s['future_feat'] = normalize_future(s['future_feat'][None])[0]
    tr_t, gate_sc = to_tensors(train_samples)
    vl_t, _ = to_tensors(val_samples, gate_scalers=gate_sc)
    print(f"  gate scalers (mean, std): drop_now {gate_sc[0][0]:.3f}/{gate_sc[1][0]:.3f}, "
          f"realized_gap {gate_sc[0][1]:.3f}/{gate_sc[1][1]:.3f}, "
          f"gap_x_sigma {gate_sc[0][2]:.3f}/{gate_sc[1][2]:.3f}")

    train_ds = SamplesDataset(tr_t); val_ds = SamplesDataset(vl_t)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=2048)

    model = IntradayTCN_2Branch(event_gain=event_gain, gate_temp=gate_temp,
                                  gate_threshold=gate_threshold,
                                  n_horizon=L_FUTURE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e3:.1f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            pf, pm, ff, fm, mu_p1, sig_p1, cf, _cap, st, gi = [b.to(device) for b in batch]
            delta, _, _, _ = model(pf, pm, ff, fm, st, gi)
            loss, _, _ = compute_loss(cf, mu_p1, sig_p1, delta, fm)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()
        model.eval()
        v_mae_sum = 0.0; v_n = 0.0
        gate_means = []
        with torch.no_grad():
            for batch in val_loader:
                pf, pm, ff, fm, mu_p1, sig_p1, cf, _cap, st, gi = [b.to(device) for b in batch]
                delta, _, _, gate = model(pf, pm, ff, fm, st, gi)
                err = (cf - (mu_p1 + delta)).abs() * fm
                v_mae_sum += err.sum().item(); v_n += fm.sum().item()
                gate_means.append(gate.mean().item())
        v_mae = v_mae_sum / max(v_n, 1)
        gate_mean = float(np.mean(gate_means))
        if v_mae < best:
            best = v_mae; bad = 0
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            mark = "★"
        else:
            bad += 1; mark = ""
        if ep % 5 == 0 or mark:
            print(f"  ep {ep:>3}: val_mae {v_mae:.4f}  gate_mean {gate_mean:.3f}  {mark}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model, gate_sc


def predict(model, samples, gate_sc, device):
    for s in samples:
        s['past_feat'] = normalize_past(s['past_feat'][None])[0]
        s['future_feat'] = normalize_future(s['future_feat'][None])[0]
    t, _ = to_tensors(samples, gate_scalers=gate_sc)
    ds = SamplesDataset(t)
    loader = DataLoader(ds, batch_size=2048)
    deltas, deltas_base, deltas_event, gates = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pf, pm, ff, fm, mu_p1, sig_p1, cf, cap, st, gi = [b.to(device) for b in batch]
            delta, db, de, g = model(pf, pm, ff, fm, st, gi)
            deltas.append(delta.cpu().numpy())
            deltas_base.append(db.cpu().numpy())
            deltas_event.append(de.cpu().numpy())
            gates.append(g.cpu().numpy())
    return (np.concatenate(deltas), np.concatenate(deltas_base),
            np.concatenate(deltas_event), np.concatenate(gates))


def collect_pred_df(samples, deltas, deltas_base, deltas_event, gates):
    rows = []
    for idx, s in enumerate(samples):
        for k in range(L_FUTURE):
            if s['future_mask'][k] < 1.0: continue
            target_dt = s['future_dt'][k] if isinstance(s.get('future_dt'), list) else s.get(f'future_dt{k}')
            rows.append({
                'date': s['date'], 'site': s['site'], 'issue_hour': s['issue_hour'],
                'lead': k+1, 'target_dt': target_dt,
                'cf': float(s['future_cf'][k]),
                'cap': float(s['future_cap'][k]),
                'dc10Tca_target': float(s['future_dc10Tca'][k]),
                'mu_phase1': float(s['future_mu_p1'][k]),
                'sigma_phase1': float(s['future_sigma_p1'][k]),
                'delta': float(deltas[idx,k]),
                'delta_base': float(deltas_base[idx,k]),
                'delta_event': float(deltas_event[idx,k]),
                'gate': float(gates[idx,k]),
                'mu_phase2': float(s['future_mu_p1'][k]) + float(deltas[idx,k]),
                'drop_now_3h_raw': float(s['gate_input_raw'][0]),
                'realized_gap_3h_raw': float(s['gate_input_raw'][1]),
            })
    return pd.DataFrame(rows)


def evaluate(test_df):
    print("\n" + "=" * 70)
    print("Phase 1 vs Phase 1+2 (2-branch)")
    print("=" * 70)
    print(f"  total tuples: {len(test_df):,}")

    test_df['mu_phase2'] = test_df['mu_phase2'].clip(lower=0)
    cap = test_df['cap'].values; cf = test_df['cf'].values
    mu1 = test_df['mu_phase1'].values; mu2 = test_df['mu_phase2'].values
    sig = test_df['sigma_phase1'].values

    def line(name, arr_mu):
        nmae = (np.abs(cf - arr_mu) * cap).sum() / cap.sum() * 100
        cov80 = ((cf >= arr_mu - 1.282*sig) & (cf <= arr_mu + 1.282*sig)).mean() * 100
        cov95 = ((cf >= arr_mu - 1.96*sig) & (cf <= arr_mu + 1.96*sig)).mean() * 100
        bias = ((arr_mu - cf) * cap).sum() / cap.sum() * 100
        print(f"  [{name:<12}] NMAE {nmae:.3f}%  bias {bias:+.3f}%  Cov80 {cov80:.1f}%  Cov95 {cov95:.1f}%")
        return nmae

    n1 = line('Phase 1', mu1)
    n2 = line('Phase 1+2', mu2)
    print(f"  Δ NMAE: {n2-n1:+.3f}%p")

    print("\n[By lead]")
    for L in [1,2,3]:
        sub = test_df[test_df.lead == L]
        c, m1l, m2l, capl = sub.cf.values, sub.mu_phase1.values, sub.mu_phase2.values, sub.cap.values
        a = (np.abs(c-m1l)*capl).sum()/capl.sum()*100
        b = (np.abs(c-m2l)*capl).sum()/capl.sum()*100
        print(f"  lead {L}h: P1 {a:.3f}% → P1+2 {b:.3f}%  ({b-a:+.3f}%p)  n={len(sub)}")

    print("\n[Partial cloud (target dc10Tca 3~7)]")
    pc = test_df[(test_df.dc10Tca_target>=3)&(test_df.dc10Tca_target<=7)]
    if len(pc)>0:
        a = (np.abs(pc.cf-pc.mu_phase1)*pc.cap).sum()/pc.cap.sum()*100
        b = (np.abs(pc.cf-pc.mu_phase2)*pc.cap).sum()/pc.cap.sum()*100
        print(f"  PC NMAE: P1 {a:.3f}% → P1+2 {b:.3f}%  ({b-a:+.3f}%p)  (n={len(pc)})")

    test_df['target_hour'] = pd.to_datetime(test_df.target_dt).dt.hour
    test_df['target_date'] = pd.to_datetime(test_df.target_dt).dt.normalize()
    print("\n[Top cloud-pass events 정오~15시]")
    for ev_str in ['2025-03-23','2025-04-26','2025-05-04']:
        ev = pd.Timestamp(ev_str)
        sub = test_df[(test_df.target_date==ev)&test_df.target_hour.between(12,15)]
        if len(sub)==0: print(f"  {ev_str}: no data"); continue
        psub = sub.assign(p1=sub.mu_phase1*sub.cap, p2=sub.mu_phase2*sub.cap, a=sub.cf*sub.cap)
        port = psub.groupby('target_dt',as_index=False).agg(p1=('p1','sum'),p2=('p2','sum'),
                                                              a=('a','sum'),c=('cap','sum'))
        pn1 = (port.p1-port.a).abs().sum()/port.c.sum()*100
        pn2 = (port.p2-port.a).abs().sum()/port.c.sum()*100
        gate_mean_event = sub.gate.mean()
        print(f"  {ev_str}: Port {pn1:.2f}% → {pn2:.2f}%  ({pn2-pn1:+.2f}%p)  | gate_mean={gate_mean_event:.3f}")

    # gate analysis
    print("\n[Gate analysis]")
    print(f"  overall gate mean: {test_df.gate.mean():.3f}  (median {test_df.gate.median():.3f})")
    print(f"  gate quartiles: {test_df.gate.quantile(0.25):.3f} / {test_df.gate.median():.3f} / {test_df.gate.quantile(0.75):.3f} / {test_df.gate.quantile(0.95):.3f} (q95)")
    # top events vs others
    ev_dates = pd.to_datetime(['2025-03-23','2025-04-26','2025-05-04'])
    ev_mask = test_df.target_date.isin(ev_dates) & test_df.target_hour.between(12,15)
    print(f"  gate mean — top event 12~15h: {test_df[ev_mask].gate.mean():.3f}  (n={ev_mask.sum()})")
    print(f"  gate mean — non-event hours:  {test_df[~ev_mask].gate.mean():.3f}  (n={(~ev_mask).sum()})")
    # quantile of drop_now_3h vs gate
    print(f"  drop_now_3h q90 = {test_df.drop_now_3h_raw.quantile(0.9):.3f}")
    high_drop = test_df[test_df.drop_now_3h_raw > test_df.drop_now_3h_raw.quantile(0.9)]
    print(f"  gate mean — drop_now_3h > q90: {high_drop.gate.mean():.3f}  (n={len(high_drop)})")


def main():
    global L_FUTURE
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=1.0,
                   help="gate temperature T: gate = sigmoid(z/T). 작을수록 sharper")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="gate threshold τ: max(0, g_raw - τ)/(1 - τ)")
    p.add_argument("--l-future", type=int, default=12,
                   help="output horizon length (default 12 = issue+1 ~ 일몰)")
    args = p.parse_args()

    L_FUTURE = args.l_future

    suffix = f"g{int(args.gain*10)}"
    if args.temp != 1.0: suffix += f"_T{int(args.temp*10)}"
    if args.threshold > 0: suffix += f"_th{int(args.threshold*100)}"
    if L_FUTURE != 3: suffix += f"_L{L_FUTURE}"
    out_dir = OUT_DIR_BASE if (args.gain==1.0 and args.temp==1.0 and args.threshold==0 and L_FUTURE==3) else \
              OUT_DIR_BASE.parent / f"phase2_2branch_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Phase 2 2-branch | seed={args.seed} | gain={args.gain} T={args.temp} τ={args.threshold}")
    print(f"  H_PAST={H_PAST}, L_FUTURE={L_FUTURE}, batch={BATCH_SIZE}, lr={LR}, "
          f"max_ep={MAX_EPOCHS}, patience={PATIENCE}, loss=MAE+{ALPHA_NLL}·NLL")
    print(f"  gate inputs: drop_now_3h, abs_realized_gap_3h, gap_x_sigma")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    p1_val = load_phase1_ensemble("val")
    p1_test = load_phase1_ensemble("test")
    p1_val['datetime_kst'] = pd.to_datetime(p1_val['datetime_kst'])
    p1_test['datetime_kst'] = pd.to_datetime(p1_test['datetime_kst'])

    weather = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    weather['datetime_kst'] = pd.to_datetime(weather['datetime_kst'])
    for col in ["ta","hm","ws","dc10Tca"]:
        weather[col] = weather.groupby("site")[col].transform(lambda s: s.fillna(s.median()))

    print("\n[Build samples — Phase 2 train (2024.01~2024.10.15)]")
    p1_val_train = p1_val[p1_val.datetime_kst < PHASE2_VAL_FROM]
    samples_train = build_samples_2branch(p1_val_train, weather, "train")
    print("\n[Build samples — Phase 2 val (2024.10.16~2024.12.31)]")
    p1_val_eval = p1_val[p1_val.datetime_kst >= PHASE2_VAL_FROM]
    samples_val = build_samples_2branch(p1_val_eval, weather, "val")
    print("\n[Build samples — Phase 2 test (2025)]")
    samples_test = build_samples_2branch(p1_test, weather, "test")

    model, gate_sc = train_one(args.seed, samples_train, samples_val, device,
                                  event_gain=args.gain, gate_temp=args.temp,
                                  gate_threshold=args.threshold)

    print("\n[Predict — test 2025]")
    deltas, db, de, gates = predict(model, samples_test, gate_sc, device)
    test_df = collect_pred_df(samples_test, deltas, db, de, gates)
    test_df.to_parquet(out_dir / f"test_predictions_seed{args.seed}.parquet", index=False)

    evaluate(test_df)
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
