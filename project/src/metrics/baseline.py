from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


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


def garch_calibrated_baseline_oof(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cv,
    feature_name: str = "garch_vol_1d",
) -> pd.Series:
    """Calibrate a single GARCH vol forecast into binary probabilities via logistic OOF."""
    if feature_name not in X.columns:
        raise ValueError(f"{feature_name} not found in X columns")

    y_series = y if isinstance(y, pd.Series) else pd.Series(y, index=X.index)
    oof = pd.Series(np.nan, index=X.index, dtype=float)

    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx][[feature_name]]
        y_train = y_series.iloc[train_idx]
        X_test = X.iloc[test_idx][[feature_name]]

        train_mask = X_train[feature_name].notna() & y_train.notna()
        X_train = X_train.loc[train_mask]
        y_train = y_train.loc[train_mask]

        if len(X_train) < 50 or y_train.nunique() < 2:
            fallback = float(y_train.mean()) if len(y_train) else 0.5
            oof.iloc[test_idx] = fallback
            continue

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
        model.fit(X_train, y_train)

        test_feat = X_test[feature_name]
        proba = np.full(len(X_test), float(y_train.mean()), dtype=float)
        valid_test = test_feat.notna()
        if valid_test.any():
            proba[valid_test.to_numpy()] = model.predict_proba(
                X_test.loc[valid_test]
            )[:, 1]
        oof.iloc[test_idx] = proba

    return oof
