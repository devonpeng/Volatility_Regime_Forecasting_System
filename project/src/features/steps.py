"""Individual feature steps. Each function adds one column and returns a new DataFrame."""

import numpy as np
import pandas as pd

from project.src.features.rv import compute_historical_rv


def add_ret_1d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out["Close"].pct_change()
    return out


def add_hl_range(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    return out


def add_rv_hist_5d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rv_hist_5d"] = compute_historical_rv(out["ret_1d"], window=5)
    return out


def add_rv_hist_10d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rv_hist_10d"] = compute_historical_rv(out["ret_1d"], window=10)
    return out


def add_rv_hist_20d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rv_hist_20d"] = compute_historical_rv(out["ret_1d"], window=20)
    return out


def add_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    vol_mean = out["Volume"].rolling(window, min_periods=window).mean()
    vol_std = out["Volume"].rolling(window, min_periods=window).std()
    out["volume_zscore"] = (out["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
    return out
