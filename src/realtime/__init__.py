"""Realtime correction layer.

Two parallel branches:
  1. bias-state Kalman filter (point correction)  → KalmanBiasCorrection
  2. scenario reweight (distribution correction)  → ScenarioReweighter

LSTM baseline의 (μ, σ) 출력 위에 적용.
"""
