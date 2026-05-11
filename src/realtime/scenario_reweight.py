"""Scenario reweight — distribution correction.

Wrapper around existing src/scenarios/bayesian_reweight.py.
LSTM baseline의 (μ, σ)에서 시나리오 풀 생성 + Bayesian filter로 가중치 갱신.

분포는 *Student-t* 우선:
  ν=3.73, loc=0.09, scale=0.688 (post_hoc_student_t.py에서 fit)
  → multivariate t로 trajectory 1000개 생성
  → 매 시간 obs 들어오면 likelihood-based reweight
"""
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.linalg import toeplitz, cholesky

ROOT = Path(__file__).resolve().parents[2]


# Student-t prior parameters (post_hoc_student_t.py에서 fit)
T_DF = 3.73
T_LOC = 0.09
T_SCALE = 0.688


# ===== Trajectory generation =====

def generate_trajectories_t(mu_seq: np.ndarray, sigma_seq: np.ndarray, acf: np.ndarray,
                             n_samples: int = 1000, seed: int = 42) -> np.ndarray:
    """Multivariate Student-t trajectories.

    x_h = μ_h + σ_h * (T_LOC + T_SCALE * t_std_h)
    where t_std ~ MVT(df=T_DF, loc=0, shape=correlation_matrix)

    Args:
        mu_seq: (T,)
        sigma_seq: (T,)
        acf: (T,) autocorrelation function
        n_samples: trajectories per day
    """
    rng = np.random.default_rng(seed)
    T = len(mu_seq)
    rho = np.zeros(T)
    rho[: len(acf)] = acf[:T]
    cor = toeplitz(rho)
    cor = (cor + cor.T) / 2
    eig = np.linalg.eigvalsh(cor)
    if eig.min() < 1e-6:
        cor += np.eye(T) * (1e-6 - eig.min())

    active = sigma_seq > 0
    if active.sum() == 0:
        return np.zeros((n_samples, T))
    cor_eff = cor[np.ix_(active, active)]
    L = cholesky(cor_eff, lower=True)
    n_eff = active.sum()

    v = rng.standard_normal((n_samples, n_eff)) @ L.T
    w = rng.chisquare(T_DF, size=n_samples)
    z_std = v * np.sqrt(T_DF / w)[:, None]
    z_scaled = T_LOC + T_SCALE * z_std

    out = np.zeros((n_samples, T))
    out[:, active] = mu_seq[active][None, :] + sigma_seq[active][None, :] * z_scaled
    out = np.clip(out, 0, None)
    return out


# ===== Reweight =====

def reweight_step(weights: np.ndarray, traj_at_t: np.ndarray,
                   observation: float, noise_std: float) -> np.ndarray:
    """One-step Bayesian update on trajectory pool weights."""
    log_lik = -0.5 * ((traj_at_t - observation) / noise_std) ** 2
    log_lik -= log_lik.max()
    new_weights = weights * np.exp(log_lik)
    s = new_weights.sum()
    if s == 0:
        return weights
    return new_weights / s


def weighted_mean_std(values: np.ndarray, weights: np.ndarray):
    m = float(np.sum(weights * values))
    var = float(np.sum(weights * (values - m) ** 2))
    return m, float(np.sqrt(max(var, 0)))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Particle filter resampling — ESS collapse 방지."""
    n = len(weights)
    positions = (np.arange(n) + rng.uniform()) / n
    indices = np.zeros(n, dtype=np.int64)
    cumsum = np.cumsum(weights)
    i, j = 0, 0
    while i < n and j < n:
        if positions[i] < cumsum[j]:
            indices[i] = j; i += 1
        else:
            j += 1
    if i < n:
        indices[i:] = n - 1
    return indices


# ===== Wrapper interface =====

class ScenarioReweighter:
    """Real-time scenario distribution correction.

    Usage:
        sr = ScenarioReweighter(noise_std=2000)
        sr.init_day(mu_seq, sigma_seq, acf, n_samples=1000)
        for t, obs in enumerate(realtime_observations):
            sr.update(t, obs)
            mu_t, sigma_t, q05_t, q95_t = sr.posterior(future_h)
    """

    def __init__(self, noise_std: float = 2000.0,
                 ess_resample_frac: float = 0.3,
                 enable_pf: bool = True,
                 jitter_ratio: float = 0.05,
                 seed: int = 42):
        self.noise_std = noise_std
        self.ess_resample_frac = ess_resample_frac
        self.enable_pf = enable_pf
        self.jitter_ratio = jitter_ratio
        self.rng = np.random.default_rng(seed)
        self.trajectories = None
        self.weights = None

    def init_day(self, mu_seq: np.ndarray, sigma_seq: np.ndarray,
                 acf: np.ndarray, n_samples: int = 1000):
        self.trajectories = generate_trajectories_t(mu_seq, sigma_seq, acf, n_samples=n_samples)
        n = len(self.trajectories)
        self.weights = np.ones(n) / n
        self.col_std = self.trajectories.std(axis=0, keepdims=True)

    def update(self, t: int, observation: float):
        """Reweight by obs at hour t."""
        self.weights = reweight_step(self.weights, self.trajectories[:, t],
                                     observation, self.noise_std)
        # Particle filter — ESS check & resample
        if self.enable_pf:
            ess = 1.0 / (self.weights ** 2).sum()
            if ess < self.ess_resample_frac * len(self.weights):
                indices = systematic_resample(self.weights, self.rng)
                self.trajectories = self.trajectories[indices].copy()
                if self.jitter_ratio > 0:
                    noise = self.rng.standard_normal(self.trajectories.shape) * self.col_std * self.jitter_ratio
                    self.trajectories = np.clip(self.trajectories + noise, 0, None)
                self.weights = np.ones(len(self.weights)) / len(self.weights)

    def posterior(self, h: int) -> dict:
        """Posterior at future hour h."""
        v = self.trajectories[:, h]
        m, s = weighted_mean_std(v, self.weights)
        q05 = self._quantile(v, 0.05)
        q95 = self._quantile(v, 0.95)
        ess = 1.0 / (self.weights ** 2).sum()
        return {"mean": m, "std": s, "q05": q05, "q95": q95, "ess": ess}

    def _quantile(self, v, q):
        idx = np.argsort(v)
        return float(np.interp(q, np.cumsum(self.weights[idx]), v[idx]))
