import numpy as np
import pandas as pd


def compute_forward_rv(
    returns: pd.Series,
    horizon: int = 5,
    annualization: float = np.sqrt(252),
) -> pd.Series:
    """Compute forward-looking realized volatility over `horizon` days."""
    sq_returns = returns.pow(2)
    future_mean_sq = (
        sq_returns[::-1]
        .rolling(window=horizon, min_periods=horizon)
        .mean()[::-1]
    )
    return future_mean_sq.pow(0.5).mul(annualization)


def compute_historical_rv(
    returns: pd.Series,
    window: int,
    annualization: float = np.sqrt(252),
) -> pd.Series:
    """Compute backward-looking realized volatility over `window` days."""
    return returns.rolling(window=window, min_periods=window).std().mul(annualization)
