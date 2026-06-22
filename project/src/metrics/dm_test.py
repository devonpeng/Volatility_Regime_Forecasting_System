from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DMTestResult:
    dm_stat: float
    p_value_two_sided: float
    p_value_one_sided: float
    mean_diff: float
    n: int


def diebold_mariano_test(
    loss_diff: np.ndarray,
    h: int = 1,
) -> DMTestResult:
    """Diebold-Mariano test for equal predictive accuracy.

    ``loss_diff`` is d_t = L_baseline_t - L_model_t. Positive mean_diff means
    the model has lower loss (better) than the baseline.
    """
    d = np.asarray(loss_diff, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return DMTestResult(
            dm_stat=0.0,
            p_value_two_sided=1.0,
            p_value_one_sided=1.0,
            mean_diff=float(np.mean(d)) if n else 0.0,
            n=n,
        )

    d_bar = float(np.mean(d))
    gamma_0 = float(np.var(d, ddof=0))
    h = max(1, int(h))

    if h > 1:
        for k in range(1, h):
            if k >= n:
                break
            gamma_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
            weight = 1.0 - k / h
            gamma_0 += 2.0 * weight * gamma_k

    if gamma_0 <= 0.0:
        return DMTestResult(
            dm_stat=0.0,
            p_value_two_sided=1.0,
            p_value_one_sided=1.0,
            mean_diff=d_bar,
            n=n,
        )

    dm_stat = d_bar / np.sqrt(gamma_0 / n)
    p_two = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    p_one = float(1.0 - stats.norm.cdf(dm_stat))

    return DMTestResult(
        dm_stat=float(dm_stat),
        p_value_two_sided=p_two,
        p_value_one_sided=p_one,
        mean_diff=d_bar,
        n=n,
    )
