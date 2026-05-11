"""MOS (Model Output Statistics) Correction.

Per-variable additive bias correction:
  corrected_v = raw_v + δ_v
  δ_v = MLP(raw_v, site_emb, hour_sin, hour_cos, month_sin, month_cos, lead_time)

5 variables:
  dsr_mean   (W/m², shortwave_radiation)
  ta         (°C, temperature_2m)
  hm         (%, relative_humidity_2m)
  ws         (m/s, wind_speed_10m)
  cloud      (0-100% or 0-10, cloud_cover/dc10Tca)

학습 데이터: (raw_forecast, observed_truth) 쌍.
  raw forecast: Open-Meteo historical-forecast-api에서 D-1 17:00 시점 issue
  truth:        ASOS + GK-2A 실측 (build_training_set 출력)

Conditioning:
  site_emb     (Embedding 8 → 8)
  hour_sin/cos (시간 cyclic)
  month_sin/cos (월 cyclic)
  lead_time    (D-1 17:00 → target 시각까지 hours, 정규화)

Loss: MSE on residual (obs - raw)
  → δ가 obs - raw 잘 fit하면 corrected ≈ obs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MOSCorrection(nn.Module):
    """Single-variable additive bias correction.

    Input:
        raw       : (B,) raw forecast value
        site_id   : (B,) long
        hour_sin/cos, month_sin/cos : (B,) float
        lead_time : (B,) float  (hours since D-1 17:00, normalized)

    Output:
        corrected : (B,) = raw + δ
    """

    def __init__(self, n_sites: int, site_emb_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.site_emb = nn.Embedding(n_sites, site_emb_dim)
        # Input: raw(1) + site_emb + hour_sin/cos(2) + month_sin/cos(2) + lead(1)
        in_dim = 1 + site_emb_dim + 2 + 2 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, raw, site_id, hour_sin, hour_cos, month_sin, month_cos, lead_time):
        s_e = self.site_emb(site_id)
        x = torch.cat([
            raw.unsqueeze(-1), s_e,
            hour_sin.unsqueeze(-1), hour_cos.unsqueeze(-1),
            month_sin.unsqueeze(-1), month_cos.unsqueeze(-1),
            lead_time.unsqueeze(-1),
        ], dim=-1)
        delta = self.net(x).squeeze(-1)
        return raw + delta, delta


class MOSStack(nn.Module):
    """5 MOS 모듈 (변수별 따로 학습)."""

    VARIABLES = ["dsr", "ta", "hm", "ws", "cloud"]

    def __init__(self, n_sites: int, site_emb_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.mos = nn.ModuleDict({
            v: MOSCorrection(n_sites, site_emb_dim, hidden) for v in self.VARIABLES
        })

    def forward(self, raw_dict: dict, site_id, hour_sin, hour_cos, month_sin, month_cos, lead_time):
        """
        raw_dict: {"dsr": (B,), "ta": (B,), "hm": (B,), "ws": (B,), "cloud": (B,)}
        Returns:
          corrected_dict: same structure (corrected values)
          delta_dict: (residuals applied)
        """
        corrected = {}
        deltas = {}
        for v in self.VARIABLES:
            c, d = self.mos[v](
                raw_dict[v], site_id,
                hour_sin, hour_cos, month_sin, month_cos, lead_time,
            )
            corrected[v] = c
            deltas[v] = d
        return corrected, deltas


def mos_loss(corrected_dict, observed_dict, weights=None):
    """MSE per-variable, sum (or weighted)."""
    if weights is None:
        weights = {v: 1.0 for v in corrected_dict}
    total = 0
    parts = {}
    for v, c in corrected_dict.items():
        if v not in observed_dict:
            continue
        diff = (c - observed_dict[v]) ** 2
        loss_v = diff.mean()
        parts[v] = loss_v.item()
        total = total + weights[v] * loss_v
    return total, parts


# ===== Training utility =====

def train_mos(
    train_loader,
    val_loader,
    n_sites: int,
    device,
    *,
    hidden: int = 32,
    site_emb_dim: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 8,
    var_weights: dict = None,
):
    """Train MOSStack.

    DataLoader yields dicts with keys:
        raw    : {"dsr": tensor, "ta": tensor, ...} each (B,)
        obs    : {"dsr": tensor, "ta": tensor, ...} each (B,)
        site   : (B,) long
        hour_sin, hour_cos, month_sin, month_cos, lead_time : (B,)
    """
    model = MOSStack(n_sites=n_sites, site_emb_dim=site_emb_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best = float("inf"); best_state = None; bad = 0

    for ep in range(max_epochs):
        model.train()
        tr_loss = 0; n = 0
        for batch in train_loader:
            raw = {v: t.to(device) for v, t in batch["raw"].items()}
            obs = {v: t.to(device) for v, t in batch["obs"].items()}
            site = batch["site"].to(device)
            hs = batch["hour_sin"].to(device); hc = batch["hour_cos"].to(device)
            ms = batch["month_sin"].to(device); mc = batch["month_cos"].to(device)
            lt = batch["lead_time"].to(device)
            corrected, _ = model(raw, site, hs, hc, ms, mc, lt)
            loss, _ = mos_loss(corrected, obs, var_weights)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * site.size(0); n += site.size(0)
        tr_loss /= n; sch.step()

        model.eval()
        v_loss = 0; nv = 0; v_parts = {v: 0.0 for v in MOSStack.VARIABLES}
        with torch.no_grad():
            for batch in val_loader:
                raw = {v: t.to(device) for v, t in batch["raw"].items()}
                obs = {v: t.to(device) for v, t in batch["obs"].items()}
                site = batch["site"].to(device)
                hs = batch["hour_sin"].to(device); hc = batch["hour_cos"].to(device)
                ms = batch["month_sin"].to(device); mc = batch["month_cos"].to(device)
                lt = batch["lead_time"].to(device)
                corrected, _ = model(raw, site, hs, hc, ms, mc, lt)
                loss, parts = mos_loss(corrected, obs, var_weights)
                v_loss += loss.item() * site.size(0); nv += site.size(0)
                for v, p in parts.items():
                    v_parts[v] += p * site.size(0)
        v_loss /= nv
        for v in v_parts: v_parts[v] /= nv

        if v_loss < best:
            best = v_loss
            best_state = {k: w.cpu().clone() for k, w in model.state_dict().items()}
            bad = 0; mark = "★"
        else:
            bad += 1; mark = ""
        print(f"  ep {ep:>3}: tr {tr_loss:.4f} val {v_loss:.4f} "
              f"(dsr {v_parts['dsr']:.3f}, ta {v_parts['ta']:.3f}, "
              f"hm {v_parts['hm']:.3f}, ws {v_parts['ws']:.3f}, cloud {v_parts['cloud']:.3f}) {mark}")
        if bad >= patience:
            print(f"  early stop @ ep {ep}"); break
    model.load_state_dict(best_state)
    return model
