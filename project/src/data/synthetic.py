from datetime import datetime

import numpy as np
import pandas as pd


def generate_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data using a GBM-style random walk."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = pd.bdate_range(start=start, end=end)

    rng = np.random.default_rng(seed)
    n = len(dates)
    if n == 0:
        raise ValueError(f"No business days between {start_date} and {end_date}")

    mu = 0.0002
    sigma = 0.015
    log_returns = rng.normal(mu, sigma, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    intraday_noise = rng.uniform(0.005, 0.02, size=n)
    high = close * (1 + intraday_noise)
    low = close * (1 - intraday_noise)
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, size=n)
    volume = rng.integers(1_000_000, 10_000_000, size=n)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": np.round(open_, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume,
        }
    )
    df.attrs["source"] = "synthetic"
    df.attrs["symbol"] = symbol.upper()
    return df
