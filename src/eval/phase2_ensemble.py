"""Phase 2 3-seed ensemble + 공식 메트릭 표.

Inputs:
  pv/experiments/phase2_intraday_tcn/test_predictions_seed{42,123,7}.parquet

Outputs:
  pv/experiments/phase2_intraday_tcn/ensemble_test.parquet
  pv/experiments/phase2_intraday_tcn/metrics_summary.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
P2_DIR = ROOT / "pv/experiments/phase2_intraday_tcn"
SEEDS = [42, 123, 7]


def cap_weighted_nmae(cf, mu, cap):
    return (np.abs(cf - mu) * cap).sum() / cap.sum() * 100


def cov(cf, mu, sigma, k):
    return ((cf >= mu - k * sigma) & (cf <= mu + k * sigma)).mean() * 100


def metrics_block(df, label):
    cap = df.cap.values; cf = df.cf.values
    mu_p1 = df.mu_phase1.values; mu_p2 = df.mu_phase2.values; sig = df.sigma_phase1.values
    return {
        'label': label, 'n': len(df),
        'P1_nmae': cap_weighted_nmae(cf, mu_p1, cap),
        'P2_nmae': cap_weighted_nmae(cf, mu_p2, cap),
        'P1_cov80': cov(cf, mu_p1, sig, 1.282),
        'P2_cov80': cov(cf, mu_p2, sig, 1.282),
        'P1_cov95': cov(cf, mu_p1, sig, 1.96),
        'P2_cov95': cov(cf, mu_p2, sig, 1.96),
        'P1_bias': ((mu_p1 - cf) * cap).sum() / cap.sum() * 100,
        'P2_bias': ((mu_p2 - cf) * cap).sum() / cap.sum() * 100,
    }


def portfolio_metric(df):
    g = df.assign(p1=df.mu_phase1*df.cap, p2=df.mu_phase2*df.cap, a=df.cf*df.cap)
    port = g.groupby('target_dt', as_index=False).agg(p1=('p1','sum'), p2=('p2','sum'),
                                                        a=('a','sum'), c=('cap','sum'))
    pn1 = (port.p1 - port.a).abs().sum() / port.c.sum() * 100
    pn2 = (port.p2 - port.a).abs().sum() / port.c.sum() * 100
    return pn1, pn2


def main():
    print("=" * 70)
    print("Phase 2 ensemble — 3 seeds [42, 123, 7]")
    print("=" * 70)

    # 1. per-seed predictions
    seed_dfs = {}
    for s in SEEDS:
        d = pd.read_parquet(P2_DIR / f"test_predictions_seed{s}.parquet")
        d['target_dt'] = pd.to_datetime(d['target_dt'])
        seed_dfs[s] = d
        print(f"  seed {s}: {len(d):,} rows")

    # 2. ensemble: same (date, site, issue_hour, lead) → mean delta
    base = seed_dfs[SEEDS[0]][['date','site','issue_hour','lead','target_dt',
                                 'cf','cap','dc10Tca_target','mu_phase1','sigma_phase1']].copy()
    delta_stack = np.stack([seed_dfs[s].sort_values(['date','site','issue_hour','lead'])['delta'].values
                            for s in SEEDS], axis=0)  # (3, N)
    # ensure same row order across seeds
    for s in SEEDS:
        seed_dfs[s] = seed_dfs[s].sort_values(['date','site','issue_hour','lead']).reset_index(drop=True)
    base = base.sort_values(['date','site','issue_hour','lead']).reset_index(drop=True)
    delta_stack = np.stack([seed_dfs[s]['delta'].values for s in SEEDS], axis=0)
    base['delta_mean'] = delta_stack.mean(axis=0)
    base['delta_std'] = delta_stack.std(axis=0, ddof=1)
    base['mu_phase2'] = (base['mu_phase1'] + base['delta_mean']).clip(lower=0)
    base.to_parquet(P2_DIR / "ensemble_test.parquet", index=False)
    print(f"  saved ensemble: {len(base):,} rows")

    # 3. per-seed individual + ensemble in one table
    rows = []
    for s, d in seed_dfs.items():
        d['mu_phase2'] = (d['mu_phase1'] + d['delta']).clip(lower=0)
        rows.append({**metrics_block(d, f'seed-{s}')})
    rows.append({**metrics_block(base, 'ENSEMBLE-3')})
    summary = pd.DataFrame(rows)
    print("\n[Per-seed + Ensemble — Overall]")
    print(summary[['label','n','P1_nmae','P2_nmae','P1_cov80','P2_cov80','P1_cov95','P2_cov95','P2_bias']]
          .round(3).to_string(index=False))

    # 4. ensemble breakdowns
    ens = base.copy()
    cap = ens.cap.values; cf = ens.cf.values
    mu1 = ens.mu_phase1.values; mu2 = ens.mu_phase2.values; sig = ens.sigma_phase1.values

    print("\n" + "=" * 70)
    print("ENSEMBLE Phase 2 (3-seed mean delta) — 공식 메트릭")
    print("=" * 70)

    def line(name, p1, p2):
        d = p2 - p1
        pct = d / p1 * 100 if p1 != 0 else 0
        print(f"  {name:<30} P1 {p1:>7.3f}% → P1+2 {p2:>7.3f}%   Δ {d:>+7.3f}%p ({pct:>+5.1f}%)")

    print("\n[Overall]")
    line("Site NMAE",
         cap_weighted_nmae(cf, mu1, cap), cap_weighted_nmae(cf, mu2, cap))
    pn1, pn2 = portfolio_metric(ens)
    line("Portfolio NMAE", pn1, pn2)
    print(f"  Cov80                          P1 {cov(cf,mu1,sig,1.282):>6.1f}% → P1+2 {cov(cf,mu2,sig,1.282):>6.1f}%")
    print(f"  Cov95                          P1 {cov(cf,mu1,sig,1.96):>6.1f}% → P1+2 {cov(cf,mu2,sig,1.96):>6.1f}%")
    print(f"  bias                           P1 {((mu1-cf)*cap).sum()/cap.sum()*100:>+6.3f}% → P1+2 {((mu2-cf)*cap).sum()/cap.sum()*100:>+6.3f}%")

    print("\n[By lead horizon]")
    for lead in [1, 2, 3]:
        sub = ens[ens.lead == lead]
        line(f"lead {lead}h NMAE",
             cap_weighted_nmae(sub.cf.values, sub.mu_phase1.values, sub.cap.values),
             cap_weighted_nmae(sub.cf.values, sub.mu_phase2.values, sub.cap.values))

    print("\n[Partial cloud (target dc10Tca 3~7)]")
    pc = ens[(ens.dc10Tca_target >= 3) & (ens.dc10Tca_target <= 7)]
    line(f"PC NMAE (n={len(pc)})",
         cap_weighted_nmae(pc.cf.values, pc.mu_phase1.values, pc.cap.values),
         cap_weighted_nmae(pc.cf.values, pc.mu_phase2.values, pc.cap.values))
    print(f"  PC Cov80                       P1 {cov(pc.cf.values, pc.mu_phase1.values, pc.sigma_phase1.values, 1.282):>6.1f}% → P1+2 {cov(pc.cf.values, pc.mu_phase2.values, pc.sigma_phase1.values, 1.282):>6.1f}%")

    print("\n[Top cloud-pass events: 정오~15시 portfolio]")
    ens['target_hour'] = pd.to_datetime(ens.target_dt).dt.hour
    ens['target_date'] = pd.to_datetime(ens.target_dt).dt.normalize()
    events = pd.to_datetime(['2025-03-23','2025-04-26','2025-05-04'])
    for ev in events:
        sub = ens[(ens.target_date == ev) & ens.target_hour.between(12, 15)]
        if len(sub) == 0: continue
        pn1, pn2 = portfolio_metric(sub)
        line(f"{ev.date()}", pn1, pn2)

    # 5. save final summary
    summary.to_csv(P2_DIR / "metrics_summary.csv", index=False)
    print(f"\n저장: {P2_DIR}")


if __name__ == "__main__":
    main()
