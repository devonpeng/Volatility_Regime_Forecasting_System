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


def _rolling_garch_vol_forecast(
    close: pd.Series,
    *,
    window: int = 750,
    min_obs: int = 252,
    p: int = 1,
    o: int = 0,
    q: int = 1,
    vol: str = "GARCH",
    dist: str = "t",
) -> pd.Series:
    """Rolling one-step-ahead annualized volatility; uses history through t-1 only."""
    from arch import arch_model

    log_close = np.log(pd.to_numeric(close, errors="coerce"))
    ret = log_close.diff() * 100.0
    garch_vol = pd.Series(np.nan, index=close.index, dtype=float)

    for i in range(min_obs, len(close)):
        start = max(0, i - window)
        train_ret = ret.iloc[start:i].dropna()
        if len(train_ret) < min_obs:
            continue
        try:
            model = arch_model(
                train_ret,
                mean="Zero",
                vol=vol,
                p=p,
                o=o,
                q=q,
                dist=dist,
                rescale=False,
            )
            res = model.fit(disp="off", show_warning=False)
            forecast = res.forecast(horizon=1, reindex=False)
            var_1d = float(forecast.variance.iloc[-1, 0])
            garch_vol.iloc[i] = np.sqrt(max(var_1d, 0.0)) / 100.0 * ANNUALIZATION
        except Exception:
            garch_vol.iloc[i] = np.nan

    return garch_vol


def add_garch_vol_1d(
    df: pd.DataFrame,
    window: int = 750,
    min_obs: int = 252,
) -> pd.DataFrame:
    """GARCH(1,1)-t one-step vol forecast, shifted for pre-open availability."""
    out = df.copy()
    vol = _rolling_garch_vol_forecast(
        out["Close"],
        window=window,
        min_obs=min_obs,
        p=1,
        o=0,
        q=1,
        vol="GARCH",
        dist="t",
    )
    out["garch_vol_1d"] = vol.shift(1)
    return out


def add_gjr_garch_vol_1d(
    df: pd.DataFrame,
    window: int = 750,
    min_obs: int = 252,
) -> pd.DataFrame:
    """GJR-GARCH(1,1)-t one-step vol forecast, shifted for pre-open availability."""
    out = df.copy()
    vol = _rolling_garch_vol_forecast(
        out["Close"],
        window=window,
        min_obs=min_obs,
        p=1,
        o=1,
        q=1,
        vol="GARCH",
        dist="t",
    )
    out["gjr_garch_vol_1d"] = vol.shift(1)
    return out


def add_egarch_vol_1d(
    df: pd.DataFrame,
    window: int = 750,
    min_obs: int = 252,
) -> pd.DataFrame:
    """EGARCH(1,1)-t one-step vol forecast, shifted for pre-open availability."""
    out = df.copy()
    vol = _rolling_garch_vol_forecast(
        out["Close"],
        window=window,
        min_obs=min_obs,
        p=1,
        o=1,
        q=1,
        vol="EGARCH",
        dist="t",
    )
    out["egarch_vol_1d"] = vol.shift(1)
    return out


def add_garch_vol_ratio_5_20(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    garch = _series_or_default(out, "garch_vol_1d")
    rv20 = _series_or_default(out, "rv_hist_20d")
    out["garch_vol_ratio_5_20"] = garch / rv20.replace(0, np.nan)
    return out


def add_garch_rv_spread(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    garch = _series_or_default(out, "garch_vol_1d")
    rv20 = _series_or_default(out, "rv_hist_20d")
    out["garch_rv_spread"] = garch - rv20
    return out


def _daily_garman_klass_vol(df: pd.DataFrame) -> pd.Series:
    """Single-day Garman-Klass annualized volatility (no rolling average)."""
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    open_ = pd.to_numeric(df["Open"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    valid = (
        np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(open_)
        & np.isfinite(close)
        & (high > 0)
        & (low > 0)
        & (open_ > 0)
        & (close > 0)
    )
    log_hl_sq = np.log(high / low).pow(2)
    log_co_sq = np.log(close / open_).pow(2)
    gk_var = 0.5 * log_hl_sq - (2.0 * np.log(2.0) - 1.0) * log_co_sq
    daily = np.sqrt(gk_var.clip(lower=0.0)) * ANNUALIZATION
    return daily.where(valid)


def _rolling_har_rv_forecast(
    daily_rv: pd.Series,
    *,
    week_window: int = 5,
    month_window: int = 22,
    fit_window: int = 750,
    min_obs: int = 252,
    alpha: float = 1.0,
) -> pd.Series:
    """Rolling one-step HAR-RV forecast on annualized daily RV (uses info through t)."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rv_d = daily_rv.astype(float)
    rv_w = rv_d.rolling(week_window, min_periods=week_window).mean()
    rv_m = rv_d.rolling(month_window, min_periods=month_window).mean()

    x_all = pd.DataFrame({"d": rv_d, "w": rv_w, "m": rv_m})
    y_all = rv_d.shift(-1)
    forecast = pd.Series(np.nan, index=daily_rv.index, dtype=float)

    for i in range(len(daily_rv)):
        start = max(0, i - fit_window)
        # Train on pairs available through t=i: features at s, target RV_d[s+1] with s+1 <= i.
        train = pd.concat(
            [x_all.iloc[start:i], y_all.iloc[start:i].rename("y")],
            axis=1,
        ).dropna()
        if len(train) < min_obs:
            continue
        x_now = x_all.iloc[i]
        if x_now.isna().any():
            continue
        try:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
                ]
            )
            model.fit(train[["d", "w", "m"]].to_numpy(dtype=float), train["y"].to_numpy(dtype=float))
            pred = float(model.predict(x_now.to_numpy(dtype=float).reshape(1, -1))[0])
            forecast.iloc[i] = max(pred, 0.0)
        except Exception:
            forecast.iloc[i] = np.nan

    return forecast


def add_har_rv_forecast_1d(
    df: pd.DataFrame,
    week_window: int = 5,
    month_window: int = 22,
    fit_window: int = 750,
    min_obs: int = 252,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """HAR-RV one-step forecast from daily Garman-Klass vol; shifted for pre-open use."""
    out = df.copy()
    daily_rv = _daily_garman_klass_vol(out)
    forecast = _rolling_har_rv_forecast(
        daily_rv,
        week_window=week_window,
        month_window=month_window,
        fit_window=fit_window,
        min_obs=min_obs,
        alpha=alpha,
    )
    out["har_rv_forecast_1d"] = forecast.shift(1)
    return out


def _kalman_medium_filter(
    close: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Local-linear-trend Kalman filter on log price (filtering only, no smoothing).

    Filters ``log(Close)`` so Q/R stay scale-invariant across assets. Returns
    (price, log_level, trend, state_var) where ``price = exp(log_level)``,
    ``trend`` is the log-space velocity, and ``state_var`` is the log-level
    state variance. Non-positive / non-finite closes are treated as missing.
    """
    from pykalman import KalmanFilter

    close_num = pd.to_numeric(close, errors="coerce").astype(float)
    values = close_num.to_numpy(dtype=float)
    price = pd.Series(np.nan, index=close.index, dtype=float)
    log_level = pd.Series(np.nan, index=close.index, dtype=float)
    trend = pd.Series(np.nan, index=close.index, dtype=float)
    state_var = pd.Series(np.nan, index=close.index, dtype=float)

    valid_mask = np.isfinite(values) & (values > 0)
    if int(valid_mask.sum()) < 2:
        return price, log_level, trend, state_var

    log_obs = np.full(len(values), np.nan, dtype=float)
    log_obs[valid_mask] = np.log(values[valid_mask])

    first_valid = int(np.flatnonzero(valid_mask)[0])
    obs = log_obs.copy()
    # Forward-fill only for filter stability on isolated NaNs; outputs stay NaN where input invalid.
    if not valid_mask.all():
        obs = pd.Series(obs).ffill().bfill().to_numpy(dtype=float)

    process_var = 1e-5
    obs_var = 1e-3
    kf = KalmanFilter(
        transition_matrices=np.array([[1.0, 1.0], [0.0, 1.0]]),
        observation_matrices=np.array([[1.0, 0.0]]),
        initial_state_mean=np.array([obs[first_valid], 0.0]),
        initial_state_covariance=np.eye(2),
        transition_covariance=process_var * np.eye(2),
        observation_covariance=np.array([[obs_var]]),
    )
    state_means, state_covs = kf.filter(obs)
    filtered_log = state_means[:, 0]
    log_level.iloc[:] = np.where(valid_mask, filtered_log, np.nan)
    trend.iloc[:] = np.where(valid_mask, state_means[:, 1], np.nan)
    state_var.iloc[:] = np.where(valid_mask, state_covs[:, 0, 0], np.nan)
    price.iloc[:] = np.where(valid_mask, np.exp(filtered_log), np.nan)
    return price, log_level, trend, state_var


_KALMAN_CACHE_COLS = (
    "_kalman_price_raw",
    "_kalman_log_raw",
    "_kalman_trend_raw",
    "_kalman_state_var_raw",
)


def _get_kalman_cache(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Reuse pipeline cache when present; otherwise run the filter once."""
    if all(c in df.columns for c in _KALMAN_CACHE_COLS):
        return (
            df["_kalman_price_raw"],
            df["_kalman_log_raw"],
            df["_kalman_trend_raw"],
            df["_kalman_state_var_raw"],
        )
    return _kalman_medium_filter(df["Close"])


def add_kalman_price_medium(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price, log_level, trend, state_var = _kalman_medium_filter(out["Close"])
    # Cache intermediate filter outputs for sibling Kalman steps in the same pipeline run.
    out["_kalman_price_raw"] = price
    out["_kalman_log_raw"] = log_level
    out["_kalman_trend_raw"] = trend
    out["_kalman_state_var_raw"] = state_var
    out["kalman_price_medium"] = price.shift(1)
    return out


def add_kalman_ret_1d_medium(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price, _, _, _ = _get_kalman_cache(out)
    out["kalman_ret_1d_medium"] = price.pct_change().shift(1)
    return out


def add_kalman_trend_medium(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, _, trend, _ = _get_kalman_cache(out)
    out["kalman_trend_medium"] = trend.shift(1)
    return out


def add_kalman_resid_medium(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, log_level, _, _ = _get_kalman_cache(out)
    close = pd.to_numeric(out["Close"], errors="coerce").astype(float)
    log_close = np.log(close.where(close > 0))
    out["kalman_resid_medium"] = (log_close - log_level).shift(1)
    return out


def add_kalman_resid_zscore_medium(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    out = df.copy()
    _, log_level, _, _ = _get_kalman_cache(out)
    close = pd.to_numeric(out["Close"], errors="coerce").astype(float)
    log_close = np.log(close.where(close > 0))
    resid = log_close - log_level
    mu = resid.rolling(window, min_periods=window).mean()
    sigma = resid.rolling(window, min_periods=window).std()
    out["kalman_resid_zscore_medium"] = ((resid - mu) / sigma.replace(0, np.nan)).shift(1)
    return out


def add_kalman_state_var_medium(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    _, _, _, state_var = _get_kalman_cache(out)
    out["kalman_state_var_medium"] = state_var.shift(1)
    return out


def add_close_to_kalman_medium(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    price, _, _, _ = _get_kalman_cache(out)
    close = pd.to_numeric(out["Close"], errors="coerce")
    out["close_to_kalman_medium"] = ((close - price) / price.replace(0, np.nan)).shift(1)
    # Drop cache columns after the last Kalman step.
    drop_cols = [c for c in _KALMAN_CACHE_COLS if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out
