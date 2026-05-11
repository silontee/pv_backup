"""Phase 2 — intraday actual-anchored TCN correction module.

Phase 1 (frozen): ResMLP+AdaLN v2 ensemble baseline. 5-seed predictions in
  pv/experiments/resmlp_adaln_v2_ensemble/seed_*/{val,test}_predictions.parquet
Phase 2 train  : 2024 (val of Phase 1)
Phase 2 test   : 2025 (test of Phase 1)

설계 원칙 (확정):
  - residual correction: μ' = μ_phase1 + Δ. TCN은 Δ만 학습.
  - actual-anchored: 항상 frozen base + 최근 actual + 최근 weather + perfect-foresight future weather가 입력.
                     recursive autoregressive rollout 금지.
  - short-horizon: next 1, 2, 3h 동시 출력. mask로 결측 horizon 처리.
  - σ는 Phase 1 그대로 (update 안 함).
  - 예천 anomaly_zero 행은 마스킹하지 않고 그대로 사용.
  - perfect-foresight: future weather는 observed GK-2A + ASOS 그대로.

UNIFIED config:
  batch=64, lr=7e-4, wd=1e-4, AdamW + CosineAnnealingLR(T_max=50),
  max_ep=50, patience=8, grad_clip=1.0, loss = MAE + 0.2*GaussianNLL,
  σ_phase1 fixed (Δμ만 학습).
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
PHASE1_DIR = ROOT / "pv/experiments/resmlp_adaln_v2_ensemble"
OUT_DIR = ROOT / "pv/experiments/phase2_intraday_tcn"
SEEDS_PHASE1 = [42, 123, 7, 202, 999]

# ===== UNIFIED CONFIG =====
BATCH_SIZE = 64
LR = 7e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 8
GRAD_CLIP = 1.0
ALPHA_NLL = 0.20

H_PAST = 6        # 6시간 past (issue_time 포함)
L_FUTURE = 12     # issue+1 ~ 일몰까지 cover (max sunset 19h - earliest issue 7h = 12 leads)
ALL_SITES = ['경상대','고흥만수상','광양항세방','구미','삼천포','영흥','예천','창원']
PHASE2_VAL_FROM = pd.Timestamp("2024-10-16")  # 2024 안에서 train/val split


def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============== Phase 1 ensemble loading ==============

def load_phase1_ensemble(split):
    """split ∈ {'val','test'}. Returns df with mu_mean, sigma_total."""
    fname = f"{split}_predictions.parquet"
    dfs = []
    for s in SEEDS_PHASE1:
        d = pd.read_parquet(PHASE1_DIR / f"seed_{s}" / fname); d["seed"] = s
        dfs.append(d)
    all_df = pd.concat(dfs, ignore_index=True)
    g = all_df.groupby(["datetime_kst", "site"])
    ens = g.agg(
        cf=("cf", "first"),
        site_capacity_kw=("site_capacity_kw", "first"),
        mu_mean=("pred_mu", "mean"),
        mu_var=("pred_mu", "var"),
        sigma_sq_mean=("pred_sigma", lambda s: (s ** 2).mean()),
    ).reset_index()
    ens["sigma_total"] = np.sqrt(ens["sigma_sq_mean"] + ens["mu_var"])
    return ens


# ============== Sample construction ==============

def normalize_past(arr):
    """arr: (N, H, 8) [cf, residual, dsr_mean, dc10Tca, ta, hm, ws, zenith]"""
    arr = arr.copy()
    arr[..., 2] /= 1000.0  # dsr_mean
    arr[..., 3] /= 10.0    # dc10Tca
    arr[..., 4] = (arr[..., 4] - 15.0) / 15.0  # ta z-score
    arr[..., 5] /= 100.0   # hm
    arr[..., 6] /= 10.0    # ws
    arr[..., 7] /= 90.0    # zenith
    return arr


def normalize_future(arr):
    """arr: (N, L, 5) [mu, sigma, dsr_mean, dc10Tca, zenith]"""
    arr = arr.copy()
    arr[..., 2] /= 1000.0
    arr[..., 3] /= 10.0
    arr[..., 4] /= 90.0
    return arr


def build_samples(phase1_df, weather_df, label):
    """For each (date, site, issue_hour), build sample.
    issue_hour ∈ [9, 16] (allow up to 16 since output 17 covered).
    """
    src_cols = ['datetime_kst','site','dsr_mean','dc10Tca','ta','hm','ws','zenith_center']
    weather_df = weather_df[src_cols].copy()
    weather_df['datetime_kst'] = pd.to_datetime(weather_df['datetime_kst'])

    df = phase1_df.copy()
    df['datetime_kst'] = pd.to_datetime(df['datetime_kst'])
    df = df.merge(weather_df, on=['datetime_kst','site'], how='left')
    df['hour'] = df.datetime_kst.dt.hour
    df['month'] = df.datetime_kst.dt.month
    df['date'] = df.datetime_kst.dt.normalize()
    df = df.sort_values(['site','datetime_kst']).reset_index(drop=True)

    site_to_idx = {s: i for i, s in enumerate(ALL_SITES)}

    rows = []
    for (date, site), g in df.groupby(['date','site']):
        g = g.set_index('hour')
        site_idx = site_to_idx[site]
        for issue_hour in range(9, 17):  # 9..16 inclusive
            # past: t-(H_PAST-1) .. t (H_PAST steps including issue hour)
            past_hours = list(range(issue_hour - (H_PAST - 1), issue_hour + 1))
            past_feat = np.zeros((H_PAST, 8), dtype=np.float32)
            past_mask = np.zeros(H_PAST, dtype=np.float32)
            for i, h in enumerate(past_hours):
                if h in g.index:
                    r = g.loc[h]
                    if pd.notna(r['dsr_mean']):
                        past_feat[i] = [
                            r['cf'], r['cf'] - r['mu_mean'],
                            r['dsr_mean'], r['dc10Tca'], r['ta'], r['hm'], r['ws'], r['zenith_center']
                        ]
                        past_mask[i] = 1.0
            # need at least the issue_hour itself (t) past valid
            if past_mask[-1] < 1.0:
                continue
            # future: t+1 .. t+3
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
            # static cond
            ih_sin = np.sin(2 * np.pi * issue_hour / 24)
            ih_cos = np.cos(2 * np.pi * issue_hour / 24)
            month = int(g.iloc[0]['month'])
            m_sin = np.sin(2 * np.pi * month / 12)
            m_cos = np.cos(2 * np.pi * month / 12)
            site_oh = np.zeros(len(ALL_SITES), dtype=np.float32)
            site_oh[site_idx] = 1.0
            static = np.concatenate([site_oh, [ih_sin, ih_cos, m_sin, m_cos]]).astype(np.float32)

            rows.append({
                'date': date, 'site': site, 'issue_hour': issue_hour,
                'past_feat': past_feat, 'past_mask': past_mask,
                'future_feat': future_feat, 'future_mask': future_mask,
                'future_mu_p1': future_mu_p1, 'future_sigma_p1': future_sigma_p1,
                'future_cf': future_cf, 'future_cap': future_cap,
                'future_dc10Tca': future_dc, 'future_dt0': future_dt[0],
                'future_dt1': future_dt[1], 'future_dt2': future_dt[2],
                'static': static,
            })
    print(f"  [{label}] samples: {len(rows)}")
    return rows


# ============== Model ==============

class IntradayTCN(nn.Module):
    def __init__(self, n_past=8, n_future=5, n_static=12, n_horizon=L_FUTURE,
                 dim_past=32, dim_future=16, dim_head=64):
        super().__init__()
        # past TCN (input has +1 mask channel = 9 channels)
        self.past_tcn = nn.Sequential(
            nn.Conv1d(n_past + 1, dim_past, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(dim_past, dim_past, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.future_mlp = nn.Sequential(
            nn.Linear(n_future + 1, dim_future), nn.GELU(),
            nn.Linear(dim_future, dim_future), nn.GELU(),
        )
        combined = dim_past + dim_future * n_horizon + n_static
        self.head = nn.Sequential(
            nn.Linear(combined, dim_head), nn.GELU(),
            nn.Linear(dim_head, dim_head), nn.GELU(),
            nn.Linear(dim_head, n_horizon),
        )

    def forward(self, past_feat, past_mask, future_feat, future_mask, static):
        # past_feat (B, H, 8), past_mask (B, H) → concat mask as channel
        past_in = torch.cat([past_feat, past_mask.unsqueeze(-1)], dim=-1)  # (B, H, 9)
        past_emb = self.past_tcn(past_in.transpose(1, 2)).squeeze(-1)       # (B, dim_past)
        # future per-step MLP, mask as extra channel
        future_in = torch.cat([future_feat, future_mask.unsqueeze(-1)], dim=-1)  # (B, L, 6)
        future_emb = self.future_mlp(future_in)                                    # (B, L, dim_future)
        future_flat = future_emb.reshape(future_emb.size(0), -1)
        combined = torch.cat([past_emb, future_flat, static], dim=-1)
        delta = self.head(combined)  # (B, L)
        return delta


# ============== Loss / Metrics ==============

def compute_loss(cf, mu_p1, sigma_p1, delta, mask):
    """Combined MAE + 0.2*NLL on updated mu, σ frozen."""
    mu_new = mu_p1 + delta
    err = (cf - mu_new).abs()
    mae = (err * mask).sum() / (mask.sum() + 1e-8)
    nll_per = 0.5 * (torch.log(2 * np.pi * sigma_p1 ** 2) + (cf - mu_new) ** 2 / sigma_p1 ** 2)
    nll = (nll_per * mask).sum() / (mask.sum() + 1e-8)
    return mae + ALPHA_NLL * nll, mae.detach(), nll.detach()


def metrics_from_arrays(cf, mu, sigma, cap, label=""):
    """capacity-weighted NMAE and coverage."""
    nmae = (np.abs(cf - mu) * cap).sum() / cap.sum() * 100
    cov80 = ((cf >= mu - 1.282 * sigma) & (cf <= mu + 1.282 * sigma)).mean() * 100
    cov95 = ((cf >= mu - 1.96 * sigma) & (cf <= mu + 1.96 * sigma)).mean() * 100
    bias = ((mu - cf) * cap).sum() / cap.sum() * 100
    if label:
        print(f"  [{label}] NMAE {nmae:.3f}%  bias {bias:+.3f}%  Cov80 {cov80:.1f}%  Cov95 {cov95:.1f}%")
    return {'nmae': nmae, 'bias': bias, 'cov80': cov80, 'cov95': cov95}


# ============== Train / Eval ==============

def to_tensors(samples):
    keys_arr = ['past_feat','past_mask','future_feat','future_mask',
                 'future_mu_p1','future_sigma_p1','future_cf','future_cap','static']
    return {k: torch.from_numpy(np.stack([s[k] for s in samples])) for k in keys_arr}


class SamplesDataset(torch.utils.data.Dataset):
    def __init__(self, tensors):
        self.t = tensors
        self.n = self.t['past_feat'].shape[0]
    def __len__(self): return self.n
    def __getitem__(self, i):
        return tuple(self.t[k][i] for k in ['past_feat','past_mask','future_feat','future_mask',
                                              'future_mu_p1','future_sigma_p1','future_cf','future_cap','static'])


def train_loop(seed, train_samples, val_samples, device):
    print(f"\n[Train seed={seed}]")
    set_all_seeds(seed)

    # normalize
    for s in train_samples + val_samples:
        s['past_feat'] = normalize_past(s['past_feat'][None])[0]
        s['future_feat'] = normalize_future(s['future_feat'][None])[0]

    tr_t = to_tensors(train_samples)
    vl_t = to_tensors(val_samples)
    train_ds = SamplesDataset(tr_t)
    val_ds = SamplesDataset(vl_t)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=2048)

    model = IntradayTCN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params/1e3:.1f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

    best = float("inf"); best_state = None; bad = 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for batch in train_loader:
            pf, pm, ff, fm, mu_p1, sig_p1, cf, _cap, st = [b.to(device) for b in batch]
            delta = model(pf, pm, ff, fm, st)
            loss, _, _ = compute_loss(cf, mu_p1, sig_p1, delta, fm)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        sch.step()
        # val
        model.eval()
        v_mae_sum = 0.0; v_n = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pf, pm, ff, fm, mu_p1, sig_p1, cf, _cap, st = [b.to(device) for b in batch]
                delta = model(pf, pm, ff, fm, st)
                err = (cf - (mu_p1 + delta)).abs() * fm
                v_mae_sum += err.sum().item(); v_n += fm.sum().item()
        v_mae = v_mae_sum / max(v_n, 1)
        if v_mae < best:
            best = v_mae; bad = 0
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            mark = "★"
        else:
            bad += 1; mark = ""
        if ep % 5 == 0 or mark:
            print(f"  ep {ep:>3}: val_mae {v_mae:.4f} {mark}", flush=True)
        if bad >= PATIENCE:
            print(f"  early stop @ ep {ep}", flush=True); break
    model.load_state_dict(best_state)
    return model


def predict(model, samples, device):
    """Run model on samples, return delta + updated mu per (sample, horizon)."""
    for s in samples:
        s['past_feat'] = normalize_past(s['past_feat'][None])[0]
        s['future_feat'] = normalize_future(s['future_feat'][None])[0]
    t = to_tensors(samples)
    ds = SamplesDataset(t)
    loader = DataLoader(ds, batch_size=2048)
    deltas = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pf, pm, ff, fm, mu_p1, sig_p1, cf, cap, st = [b.to(device) for b in batch]
            d = model(pf, pm, ff, fm, st)
            deltas.append(d.cpu().numpy())
    return np.concatenate(deltas)


def collect_pred_df(samples, deltas):
    """Flatten (sample, horizon) → row table with phase1 & phase2 mu, σ, cf, cap."""
    rows = []
    for s, dd in zip(samples, deltas):
        for k in range(L_FUTURE):
            if s['future_mask'][k] < 1.0:
                continue
            target_dt = s.get(f'future_dt{k}')
            rows.append({
                'date': s['date'], 'site': s['site'], 'issue_hour': s['issue_hour'],
                'lead': k + 1, 'target_dt': target_dt,
                'cf': float(s['future_cf'][k]),
                'cap': float(s['future_cap'][k]),
                'dc10Tca_target': float(s['future_dc10Tca'][k]),
                'mu_phase1': float(s['future_mu_p1'][k]),
                'sigma_phase1': float(s['future_sigma_p1'][k]),
                'delta': float(dd[k]),
                'mu_phase2': float(s['future_mu_p1'][k]) + float(dd[k]),
            })
    return pd.DataFrame(rows)


def evaluate(test_df):
    print("\n" + "=" * 70)
    print("Phase 1 vs Phase 1+2 (rolling, all (issue, lead, site) tuples)")
    print("=" * 70)
    print(f"  total tuples: {len(test_df):,}")

    # clip phase2 to [0, ∞)
    test_df['mu_phase2'] = test_df['mu_phase2'].clip(lower=0)

    cap = test_df['cap'].values
    cf = test_df['cf'].values
    mu1 = test_df['mu_phase1'].values
    mu2 = test_df['mu_phase2'].values
    sig = test_df['sigma_phase1'].values

    print("\n[Overall]")
    m1 = metrics_from_arrays(cf, mu1, sig, cap, "Phase 1 only")
    m2 = metrics_from_arrays(cf, mu2, sig, cap, "Phase 1+2")
    print(f"  Δ NMAE: {m2['nmae']-m1['nmae']:+.3f}%p")

    print("\n[By lead]")
    for lead in range(1, L_FUTURE + 1):
        sub = test_df[test_df.lead == lead]
        if len(sub) == 0: continue
        c, m1l, m2l, sl, capl = sub.cf.values, sub.mu_phase1.values, sub.mu_phase2.values, sub.sigma_phase1.values, sub.cap.values
        a = metrics_from_arrays(c, m1l, sl, capl)
        b = metrics_from_arrays(c, m2l, sl, capl)
        print(f"  lead {lead}h: P1 NMAE {a['nmae']:.3f}% → P1+2 {b['nmae']:.3f}%  ({b['nmae']-a['nmae']:+.3f}%p)  n={len(sub)}")

    print("\n[Partial cloud (target dc10Tca 3~7)]")
    pc = test_df[(test_df.dc10Tca_target >= 3) & (test_df.dc10Tca_target <= 7)]
    if len(pc) > 0:
        a = metrics_from_arrays(pc.cf.values, pc.mu_phase1.values, pc.sigma_phase1.values, pc.cap.values, "Phase 1 only")
        b = metrics_from_arrays(pc.cf.values, pc.mu_phase2.values, pc.sigma_phase1.values, pc.cap.values, "Phase 1+2")
        print(f"  Δ NMAE: {b['nmae']-a['nmae']:+.3f}%p   (n={len(pc)})")

    test_df['target_hour'] = pd.to_datetime(test_df['target_dt']).dt.hour
    test_df['target_date'] = pd.to_datetime(test_df['target_dt']).dt.normalize()
    test_df['pred1_kwh']  = test_df['mu_phase1'] * test_df['cap']
    test_df['pred2_kwh']  = test_df['mu_phase2'] * test_df['cap']
    test_df['actual_kwh'] = test_df['cf']        * test_df['cap']

    print("\n[Top cloud-pass events: 2025-03-23 / 04-26 / 05-04, 정오~15시 = target_hour 12~15]")
    events = pd.to_datetime(['2025-03-23','2025-04-26','2025-05-04'])
    for ev in events:
        ev_sub = test_df[(test_df.target_date == ev) & test_df.target_hour.between(12, 15)]
        if len(ev_sub) == 0:
            print(f"  {ev.date()}: no data"); continue
        port = ev_sub.groupby('target_dt', as_index=False).agg(
            pred1=('pred1_kwh','sum'), pred2=('pred2_kwh','sum'),
            actual=('actual_kwh','sum'), cap=('cap','sum'))
        pn1 = (port.pred1 - port.actual).abs().sum() / port.cap.sum() * 100
        pn2 = (port.pred2 - port.actual).abs().sum() / port.cap.sum() * 100
        n_tuples = len(ev_sub); n_hours = port.target_dt.nunique()
        print(f"  {ev.date()}: Port P1 {pn1:.2f}% → P1+2 {pn2:.2f}%  ({pn2-pn1:+.2f}%p)  "
              f"target_hours={n_hours} tuples={n_tuples}")

    print("\n[Portfolio (per target_dt, all events)]")
    port = test_df.groupby('target_dt', as_index=False).agg(
        pred1=('pred1_kwh','sum'), pred2=('pred2_kwh','sum'),
        actual=('actual_kwh','sum'), cap=('cap','sum'))
    pn1 = (port.pred1 - port.actual).abs().sum() / port.cap.sum() * 100
    pn2 = (port.pred2 - port.actual).abs().sum() / port.cap.sum() * 100
    print(f"  Port NMAE: P1 {pn1:.3f}% → P1+2 {pn2:.3f}%  ({pn2-pn1:+.3f}%p)")


def main():
    global H_PAST
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--h-past", type=int, default=6,
                   help="past window length (1~6 추천). default=6.")
    args = p.parse_args()

    # Override module-level constant if requested
    H_PAST = args.h_past

    out_dir = OUT_DIR if args.h_past == 6 else (
        OUT_DIR.parent / f"phase2_intraday_tcn_h{args.h_past}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Phase 2 intraday TCN | seed={args.seed} | H_PAST={H_PAST}")
    print(f"  H_PAST={H_PAST}, L_FUTURE={L_FUTURE}, batch={BATCH_SIZE}, lr={LR}, "
          f"max_ep={MAX_EPOCHS}, patience={PATIENCE}, loss=MAE+{ALPHA_NLL}·NLL")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n[Phase 1 ensemble loading]")
    p1_val = load_phase1_ensemble("val")    # 2024
    p1_test = load_phase1_ensemble("test")  # 2025
    p1_val['datetime_kst'] = pd.to_datetime(p1_val['datetime_kst'])
    p1_test['datetime_kst'] = pd.to_datetime(p1_test['datetime_kst'])
    print(f"  val rows  (2024): {len(p1_val)}")
    print(f"  test rows (2025): {len(p1_test)}")

    print("\n[Weather source]")
    weather = pd.read_parquet(ROOT / "data/processed/training_set.parquet")
    weather['datetime_kst'] = pd.to_datetime(weather['datetime_kst'])
    # imputation (ASOS gap 처리)
    for col in ["ta", "hm", "ws", "dc10Tca"]:
        weather[col] = weather.groupby("site")[col].transform(lambda s: s.fillna(s.median()))

    print("\n[Build samples — Phase 2 train (2024.01~2024.10.15)]")
    p1_val_train = p1_val[p1_val.datetime_kst < PHASE2_VAL_FROM]
    samples_train = build_samples(p1_val_train, weather, "train")

    print("\n[Build samples — Phase 2 val (2024.10.16~2024.12.31)]")
    p1_val_eval = p1_val[p1_val.datetime_kst >= PHASE2_VAL_FROM]
    samples_val = build_samples(p1_val_eval, weather, "val")

    print("\n[Build samples — Phase 2 test (2025)]")
    samples_test = build_samples(p1_test, weather, "test")

    model = train_loop(args.seed, samples_train, samples_val, device)

    # predict on test
    print("\n[Predict — test 2025]")
    deltas = predict(model, samples_test, device)
    test_df = collect_pred_df(samples_test, deltas)
    test_df.to_parquet(out_dir / f"test_predictions_seed{args.seed}.parquet", index=False)

    evaluate(test_df)
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
