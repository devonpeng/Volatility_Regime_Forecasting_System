"""Individual feature steps. Each function adds one column and returns a new DataFrame."""

import numpy as np
import pandas as pd

from project.src.features.rv import compute_historical_rv


ANNUALIZATION = np.sqrt(252)


def _series_or_default(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


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


def add_parkinson_rv(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    log_hl = np.log(out["High"] / out["Low"])
    out["parkinson_rv"] = (
        (log_hl ** 2 / (4 * np.log(2)))
        .rolling(window, min_periods=window)
        .mean()
        .pow(0.5)
        * ANNUALIZATION
    )
    return out


def add_garman_klass_rv(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    log_hl_sq = np.log(out["High"] / out["Low"]).pow(2)
    log_co_sq = np.log(out["Close"] / out["Open"]).pow(2)
    gk = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co_sq
    out["gk_rv"] = (
        gk.rolling(window, min_periods=window)
        .mean()
        .clip(lower=0)
        .pow(0.5)
        * ANNUALIZATION
    )
    return out


def add_rogers_satchell_rv(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    log_ho = np.log(out["High"] / out["Open"])
    log_lo = np.log(out["Low"] / out["Open"])
    log_co = np.log(out["Close"] / out["Open"])
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    out["rs_rv"] = (
        rs.rolling(window, min_periods=window).mean().clip(lower=0).pow(0.5) * ANNUALIZATION
    )
    return out


def add_yang_zhang_rv(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    log_oc = np.log(out["Open"] / out["Close"].shift(1))
    log_co = np.log(out["Close"] / out["Open"])
    log_ho = np.log(out["High"] / out["Open"])
    log_lo = np.log(out["Low"] / out["Open"])
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    overnight_var = log_oc.rolling(window, min_periods=window).var()
    open_close_var = log_co.rolling(window, min_periods=window).var()
    rs_var = rs.rolling(window, min_periods=window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yz = overnight_var + k * open_close_var + (1 - k) * rs_var
    out["yz_rv"] = yz.clip(lower=0).pow(0.5) * ANNUALIZATION
    return out


def add_abs_ret_lag1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abs_ret_lag1"] = out["ret_1d"].abs().shift(1)
    return out


def add_rv_ratio_5_20(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rv_ratio_5_20"] = out["rv_hist_5d"] / out["rv_hist_20d"].replace(0, np.nan)
    return out


def add_vix_change_1d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vix = _series_or_default(out, "vix")
    out["vix_change_1d"] = vix.diff().shift(1).fillna(0.0)
    return out


def add_vix_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vix_level"] = _series_or_default(out, "vix")
    return out


def add_vix_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vix = _series_or_default(out, "vix")
    vix3m = _series_or_default(out, "vix3m")
    out["vix_term_structure"] = (vix3m - vix).fillna(0.0)
    return out


def add_vix_rv_spread(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vix = _series_or_default(out, "vix")
    rv = _series_or_default(out, "rv_hist_20d")
    out["vix_rv_spread"] = vix.div(100.0).sub(rv).fillna(0.0)
    return out


def add_vvix_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vvix_level"] = _series_or_default(out, "vvix")
    return out


def add_skew_index_level(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["skew_index_level"] = _series_or_default(out, "skew")
    return out


def add_vol_of_vol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["vol_of_vol"] = _series_or_default(out, "rv_hist_5d").rolling(
        window, min_periods=window
    ).std()
    return out


def add_rv_acceleration(df: pd.DataFrame, lag: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["rv_accel"] = _series_or_default(out, "rv_hist_5d").sub(
        _series_or_default(out, "rv_hist_5d").shift(lag)
    )
    return out


def add_ewma_rv(df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    out = df.copy()
    ret = _series_or_default(out, "ret_1d")
    out["ewma_rv"] = ret.pow(2).ewm(span=span, min_periods=span).mean().pow(0.5) * ANNUALIZATION
    return out


def add_downside_semivar(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    ret = _series_or_default(out, "ret_1d")
    neg_ret = ret.where(ret < 0, 0.0)
    out["downside_rv"] = (
        neg_ret.pow(2).rolling(window, min_periods=window).mean().pow(0.5) * ANNUALIZATION
    )
    return out


def add_ret_skew_20d(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["ret_skew_20d"] = _series_or_default(out, "ret_1d").rolling(
        window, min_periods=window
    ).skew()
    return out


def add_ret_kurt_20d(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["ret_kurt_20d"] = _series_or_default(out, "ret_1d").rolling(
        window, min_periods=window
    ).kurt()
    return out


def add_max_drawdown_20d(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    close = _series_or_default(out, "Close")
    roll_max = close.rolling(window, min_periods=window).max()
    drawdown = close / roll_max - 1.0
    out["max_drawdown_20d"] = drawdown.rolling(window, min_periods=window).min()
    return out


def add_ret_autocorr_1_20d(df: pd.DataFrame, window: int = 20, lag: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["ret_autocorr_1_20d"] = _series_or_default(out, "ret_1d").rolling(
        window, min_periods=window
    ).apply(lambda x: pd.Series(x).autocorr(lag=lag), raw=False)
    return out


def add_yield_change_1d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["yield_change_1d"] = _series_or_default(out, "tnx").diff().shift(1).fillna(0.0)
    return out


def add_dxy_change_1d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dxy_change_1d"] = _series_or_default(out, "dxy").pct_change().shift(1).fillna(0.0)
    return out


def add_gold_ret_1d(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gold_ret_1d"] = _series_or_default(out, "gold").pct_change().shift(1).fillna(0.0)
    return out


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_series = (
        pd.to_datetime(out["Date"]) if "Date" in out.columns else pd.to_datetime(out.index)
    )
    out["day_of_week"] = date_series.dt.dayofweek.astype(float)
    return out


def add_is_month_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_series = (
        pd.to_datetime(out["Date"]) if "Date" in out.columns else pd.to_datetime(out.index)
    )
    out["is_month_end"] = date_series.dt.is_month_end.astype(float)
    return out


def add_is_quarter_end(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    date_series = (
        pd.to_datetime(out["Date"]) if "Date" in out.columns else pd.to_datetime(out.index)
    )
    out["is_quarter_end"] = date_series.dt.is_quarter_end.astype(float)
    return out


def add_dist_from_ma50(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    out = df.copy()
    close = _series_or_default(out, "Close")
    ma = close.rolling(window, min_periods=window).mean()
    out["dist_from_ma50"] = (close - ma) / ma.replace(0, np.nan)
    return out


def add_bb_width(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    close = _series_or_default(out, "Close")
    ma = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    out["bb_width"] = (4.0 * std) / ma.replace(0, np.nan)
    return out


def add_overnight_ret(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["overnight_ret"] = np.log(out["Open"] / out["Close"].shift(1))
    return out


def add_intraday_ret(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["intraday_ret"] = np.log(out["Close"] / out["Open"])
    return out


def add_overnight_vol_5d(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.copy()
    overnight_ret = _series_or_default(out, "overnight_ret")
    out["overnight_vol_5d"] = (
        overnight_ret.rolling(window, min_periods=window).std() * ANNUALIZATION
    )
    return out
