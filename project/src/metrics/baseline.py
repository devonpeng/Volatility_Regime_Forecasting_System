from __future__ import annotations

import numpy as np


def prior_probability_baseline(y: np.ndarray) -> np.ndarray:
    """Expanding historical mean of labels as prior probability (no look-ahead).

    For index t, p_t = mean(y[:t]). The first observation falls back to mean(y).
    """
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if n == 0:
        return np.array([], dtype=float)

    fallback = float(np.mean(y_arr))
    proba = np.empty(n, dtype=float)
    proba[0] = fallback
    for t in range(1, n):
        proba[t] = float(np.mean(y_arr[:t]))
    return proba


def rv_hist_persistence_baseline(
    rv_hist_5d: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Naive persistence baseline using today's rv_hist_5d high/low state.

    For each t, define state_t by comparing rv_hist_5d[t] to the expanding median
    of rv_hist_5d[:t] (no look-ahead). Predict p_t = 1 when today's state is High,
    else p_t = 0. The first sample falls back to unconditional mean(y).
    """
    rv = np.asarray(rv_hist_5d, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if rv.shape[0] != y_arr.shape[0]:
        raise ValueError("rv_hist_5d and y must have the same length")

    n = len(rv)
    if n == 0:
        return np.array([], dtype=float)

    fallback = float(np.mean(y_arr))
    proba = np.empty(n, dtype=float)
    proba[0] = fallback

    for t in range(1, n):
        hist = rv[:t]
        hist = hist[~np.isnan(hist)]
        if hist.size == 0 or np.isnan(rv[t]):
            proba[t] = fallback
            continue
        threshold = float(np.median(hist))
        proba[t] = 1.0 if rv[t] >= threshold else 0.0

    return proba
