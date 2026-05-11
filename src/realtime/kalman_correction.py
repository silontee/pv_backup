"""Bias-state Kalman filter — point correction on LSTM baseline (μ, σ).

Model:
  observation_t  = baseline_t + bias_t + ε_t,    ε_t ~ N(0, R)
  bias_t = bias_{t-1} + η_t,                    η_t ~ N(0, Q)

상태:
  x_t = bias_t (scalar) — slowly varying systematic offset

Update (1차 단순 KF):
  Predict:
    x_hat_t|t-1   = x_hat_{t-1|t-1}
    P_t|t-1       = P_{t-1|t-1} + Q
  Correct (실측 obs_t 들어왔을 때):
    innovation    = obs_t - (baseline_t + x_hat_t|t-1)
    S             = P_t|t-1 + R
    K             = P_t|t-1 / S
    x_hat_t|t     = x_hat_t|t-1 + K * innovation
    P_t|t         = (1 - K) * P_t|t-1
  Output:
    corrected_baseline_τ = baseline_τ + x_hat_t|t   (for τ > t)

확장 가능성 (TODO):
  - tangent-linear: bias 모델을 시간/cloud 등에 conditional하게
  - extended KF: nonlinear observation/state dynamics
  - 사이트별 bias state (multi-dim)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KalmanBiasCorrection:
    """Bias-state Kalman filter, scalar bias state."""

    Q: float = 1e-3        # process noise (bias drift per step) — 작을수록 bias 변화 느림
    R: float = 4.0         # observation noise variance (MWh² scale로) — sensor noise + model mismatch
    init_bias: float = 0.0
    init_variance: float = 1.0

    # internal state
    bias: float = field(init=False, default=0.0)
    variance: float = field(init=False, default=1.0)
    n_updates: int = field(init=False, default=0)

    def __post_init__(self):
        self.bias = self.init_bias
        self.variance = self.init_variance

    def reset(self):
        self.bias = self.init_bias
        self.variance = self.init_variance
        self.n_updates = 0

    def predict(self):
        """Time update — bias state 분산 증가 (random walk)."""
        self.variance += self.Q

    def update(self, observation: float, baseline: float, obs_noise: Optional[float] = None):
        """Measurement update — 새 obs로 bias 추정 갱신.

        Args:
            observation: 실측값
            baseline: 모델 prediction (이미 forecast time에 알려진 값)
            obs_noise: optional override for R (예: σ from model)
        """
        R = obs_noise if obs_noise is not None else self.R
        innovation = observation - (baseline + self.bias)
        S = self.variance + R
        K = self.variance / S
        self.bias += K * innovation
        self.variance = (1 - K) * self.variance
        self.n_updates += 1

    def correct(self, baseline: float) -> float:
        """예측에 현재 bias 적용."""
        return baseline + self.bias

    def correct_array(self, baseline_arr: np.ndarray) -> np.ndarray:
        """Array 입력 시 동일 bias 적용."""
        return baseline_arr + self.bias

    def state(self) -> dict:
        return {
            "bias": float(self.bias),
            "variance": float(self.variance),
            "n_updates": int(self.n_updates),
        }


def simulate_realtime_correction(
    baseline_seq: np.ndarray,
    actual_seq: np.ndarray,
    sigma_seq: Optional[np.ndarray] = None,
    Q: float = 1e-3,
    R: float = 4.0,
    apply_lookahead: int = 1,
):
    """매 시점 t의 obs로 KF 갱신 → t+lookahead 시점부터 보정.

    Args:
        baseline_seq: (T,) 모델 baseline (LSTM μ)
        actual_seq:   (T,) 실측 (cf 또는 kWh 단위, baseline과 동일)
        sigma_seq:    (T,) optional. 있으면 R 대신 σ²를 obs noise로
        Q, R:         KF parameters
        apply_lookahead: t에서 obs 받은 후 t+k 시점부터 보정 (k>=1)

    Returns:
        corrected: (T,) 보정된 prediction (apply_lookahead 만큼 lag된 보정)
        biases:    (T,) 시점별 bias 추정
    """
    n = len(baseline_seq)
    kf = KalmanBiasCorrection(Q=Q, R=R)
    corrected = np.array(baseline_seq, dtype=float).copy()
    biases = np.zeros(n)

    for t in range(n):
        kf.predict()
        if not np.isnan(actual_seq[t]):
            obs_noise = (sigma_seq[t] ** 2) if sigma_seq is not None else None
            kf.update(actual_seq[t], baseline_seq[t], obs_noise=obs_noise)
        biases[t] = kf.bias
        # apply correction to future steps
        for k in range(t + apply_lookahead, n):
            corrected[k] = baseline_seq[k] + kf.bias

    return corrected, biases


# ===== 인터페이스 (추후 LSTM과 결합 시 사용) =====

class RealtimeCorrector:
    """LSTM baseline + KF 결합 wrapper.

    Usage:
        corr = RealtimeCorrector(Q=1e-3, R=4.0)
        for hour in operational_hours:
            μ_baseline, σ_baseline = lstm_predict(hour)
            μ_corrected = corr.predict(μ_baseline)
            if observation_available(hour):
                actual = observe(hour)
                corr.update(actual, μ_baseline)
    """

    def __init__(self, Q: float = 1e-3, R: float = 4.0):
        self.kf = KalmanBiasCorrection(Q=Q, R=R)

    def predict(self, baseline: float) -> float:
        self.kf.predict()
        return self.kf.correct(baseline)

    def update(self, observation: float, baseline: float, sigma: Optional[float] = None):
        obs_noise = (sigma ** 2) if sigma is not None else None
        self.kf.update(observation, baseline, obs_noise=obs_noise)

    def reset(self):
        self.kf.reset()

    def state(self):
        return self.kf.state()
