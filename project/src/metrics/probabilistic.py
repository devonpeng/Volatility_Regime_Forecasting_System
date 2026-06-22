from __future__ import annotations

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

_EPS = 1e-15


def _clip_proba(y_proba: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_proba, dtype=float), _EPS, 1.0 - _EPS)


def brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    return float(brier_score_loss(y_true, _clip_proba(y_proba)))


def log_loss_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Cross-entropy loss; heavily penalizes overconfident wrong predictions."""
    return float(log_loss(y_true, _clip_proba(y_proba)))


def per_observation_brier(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    """Per-sample Brier loss: (f_t - o_t)^2."""
    y = np.asarray(y_true, dtype=float)
    p = _clip_proba(y_proba)
    return (p - y) ** 2


def per_observation_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> np.ndarray:
    """Per-sample log-loss."""
    y = np.asarray(y_true, dtype=float)
    p = _clip_proba(y_proba)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
