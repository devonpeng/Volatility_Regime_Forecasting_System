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
