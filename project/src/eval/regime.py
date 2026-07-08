from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import yfinance as yf


def fetch_vix_ma_series(
    start_date: str,
    end_date: str,
    *,
    ma_window: int = 30,
) -> pd.Series:
    start_ts = pd.Timestamp(start_date) - pd.Timedelta(days=90)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    vix_df = yf.download(
        "^VIX",
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_ts.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if vix_df is None or vix_df.empty:
        raise ValueError("Failed to fetch ^VIX data from yfinance (empty result)")
    if "Close" not in vix_df.columns:
        raise ValueError("^VIX data does not contain Close column")
    vix_close_raw = vix_df["Close"]
    if isinstance(vix_close_raw, pd.DataFrame):
        if vix_close_raw.shape[1] != 1:
            raise ValueError(
                f"^VIX Close has unexpected shape {vix_close_raw.shape}; expected single column"
            )
        vix_close = vix_close_raw.iloc[:, 0].astype(float)
    else:
        vix_close = vix_close_raw.astype(float)
    vix_ma = vix_close.rolling(window=ma_window, min_periods=10).mean().bfill().ffill()
    return pd.Series(vix_ma, name="vix_ma30")


def build_macro_regime_labels(
    oof_valid_df: pd.DataFrame,
    pipeline_config: dict[str, Any],
    *,
    ma_window: int = 30,
) -> tuple[pd.Series, float, str, dict[str, Any]]:
    if "Date" not in oof_valid_df.columns:
        raise ValueError("oof_predictions.csv must contain Date for VIX regime split")
    start_date = str(pipeline_config.get("start_date", ""))
    end_date = str(pipeline_config.get("end_date", ""))
    if not start_date or not end_date:
        raise ValueError("pipeline.json must contain start_date and end_date for VIX split")

    dates = pd.to_datetime(oof_valid_df["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Date column contains invalid values; cannot align VIX regimes")

    try:
        vix_ma = fetch_vix_ma_series(start_date, end_date, ma_window=ma_window)
        vix_ma.index = pd.to_datetime(vix_ma.index)
        vix_for_oof = vix_ma.reindex(dates).bfill().ffill()
        if vix_for_oof.isna().all():
            raise ValueError("Unable to align fetched VIX series to OOF validation dates")
        macro_indicator = pd.Series(vix_for_oof.to_numpy(), index=oof_valid_df.index)
        indicator_name = "vix_ma30_from_yfinance"
        indicator_meta: dict[str, Any] = {
            "used_fallback": False,
            "reason": None,
            "source_symbol": "^VIX",
            "construction": "close_rolling_mean",
            "ma_window": ma_window,
        }
    except Exception as exc:
        if "Close" not in oof_valid_df.columns:
            raise ValueError(
                "VIX fetch failed and no Close column available for fallback regime split"
            )
        close = oof_valid_df["Close"].astype(float)
        returns = close.pct_change()
        hist_vol_30 = returns.rolling(window=30, min_periods=10).std() * np.sqrt(252.0)
        macro_indicator = hist_vol_30.rolling(window=ma_window, min_periods=10).mean()
        macro_indicator = macro_indicator.bfill().ffill()
        indicator_name = "fallback_rolling_volatility_ma30_from_close"
        indicator_meta = {
            "used_fallback": True,
            "reason": str(exc),
            "source_symbol": "oof_valid_df.Close",
            "construction": "annualized_close_return_volatility_then_ma",
            "annualization_factor": 252.0,
            "vol_window": 30,
            "ma_window": ma_window,
        }

    threshold = float(macro_indicator.median())
    regimes = pd.Series(
        np.where(macro_indicator >= threshold, "High_VIX", "Low_VIX"),
        index=oof_valid_df.index,
        name="Regime",
    )
    return regimes, threshold, indicator_name, indicator_meta, macro_indicator


def group_confusion_matrices(
    df: pd.DataFrame,
    *,
    group_col: str,
    y_col: str = "y",
    proba_col: str = "model_proba",
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group_value, part in df.groupby(group_col):
        y_true = part[y_col].astype(int).to_numpy()
        y_pred = (part[proba_col].astype(float).to_numpy() >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        out[str(group_value)] = {
            "n_samples": int(len(part)),
            "threshold": threshold,
            "confusion_matrix": cm.tolist(),
        }
    return out
