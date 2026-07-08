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
    variance_estimator: str
    hac_lag: int


def diebold_mariano_test(
    loss_diff: np.ndarray,
    h: int = 1,
    *,
    use_hac: bool = True,
    hac_lag: int | None = None,
) -> DMTestResult:
    """Diebold-Mariano test for equal predictive accuracy.

    ``loss_diff`` is d_t = L_baseline_t - L_model_t. Positive mean_diff means
    the model has lower loss (better) than the baseline.
    """
    d = np.asarray(loss_diff, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    h = max(1, int(h))
    lag = max(0, int(hac_lag if hac_lag is not None else (h - 1)))
    if n < 2:
        return DMTestResult(
            dm_stat=0.0,
            p_value_two_sided=1.0,
            p_value_one_sided=1.0,
            mean_diff=float(np.mean(d)) if n else 0.0,
            n=n,
            variance_estimator="newey_west" if use_hac else "iid",
            hac_lag=min(lag, max(n - 1, 0)),
        )

    d_bar = float(np.mean(d))
    centered = d - d_bar
    long_run_var = float(np.mean(centered * centered))

    if use_hac and lag > 0:
        max_lag = min(lag, n - 1)
        for k in range(1, max_lag + 1):
            gamma_k = float(np.mean(centered[k:] * centered[:-k]))
            # Newey-West/Bartlett kernel weight.
            weight = 1.0 - (k / (max_lag + 1.0))
            long_run_var += 2.0 * weight * gamma_k
        lag = max_lag
    else:
        lag = 0

    if long_run_var <= 0.0:
        return DMTestResult(
            dm_stat=0.0,
            p_value_two_sided=1.0,
            p_value_one_sided=1.0,
            mean_diff=d_bar,
            n=n,
            variance_estimator="newey_west" if use_hac else "iid",
            hac_lag=lag,
        )

    dm_stat = d_bar / np.sqrt(long_run_var / n)
    p_two = float(2.0 * (1.0 - stats.norm.cdf(abs(dm_stat))))
    p_one = float(1.0 - stats.norm.cdf(dm_stat))

    return DMTestResult(
        dm_stat=float(dm_stat),
        p_value_two_sided=p_two,
        p_value_one_sided=p_one,
        mean_diff=d_bar,
        n=n,
        variance_estimator="newey_west" if use_hac else "iid",
        hac_lag=lag,
    )
