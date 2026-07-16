from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


_EPS = 1e-15


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
    # Ensure no exact 0/1 from edge cases
    proba = np.clip(proba, _EPS, 1 - _EPS)
    return proba


def rv_hist_persistence_baseline(
    rv_hist_5d: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Soft persistence baseline using today's rv_hist_5d high/low state.

    For each t, classify the rv state (above/below expanding median), then estimate
    P(y=1 | that state) from a rolling window of historical observations where the
    rv state matched. This produces smooth probabilities instead of hard 0/1.

    Falls back to the expanding mean of y when insufficient history is available.
    Predictions are clipped to [_EPS, 1-_EPS] to avoid infinite log-loss.
    """
    rv = np.asarray(rv_hist_5d, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if rv.shape[0] != y_arr.shape[0]:
        raise ValueError("rv_hist_5d and y must have the same length")

    n = len(rv)
    if n == 0:
        return np.array([], dtype=float)

    fallback = float(np.mean(y_arr))
    proba = np.full(n, fallback, dtype=float)

    LOOKBACK = 120
    MIN_SAMPLES = 20

    for t in range(1, n):
        # Compute expanding median of past rv values
        hist_rv = rv[:t]
        mask = ~np.isnan(hist_rv)
        if not np.any(mask):
            proba[t] = fallback
            continue

        threshold = float(np.median(hist_rv[mask]))

        if np.isnan(rv[t]):
            proba[t] = fallback
            continue

        state_is_high = bool(rv[t] >= threshold)

        # Gather rolling history of matching states
        start = max(0, t - LOOKBACK)
        above_y_sum = 0.0; above_n = 0
        below_y_sum = 0.0; below_n = 0

        for j in range(start, t):
            if np.isnan(rv[j]) or np.isnan(y_arr[j]):
                continue
            hist_j = rv[:j]
            mask_j = ~np.isnan(hist_j)
            if not np.any(mask_j):
                continue
            med_j = float(np.median(hist_j[mask_j]))
            was_high = bool(rv[j] >= med_j)

            if was_high:
                above_y_sum += y_arr[j]
                above_n += 1
            else:
                below_y_sum += y_arr[j]
                below_n += 1

        total_n = above_n + below_n
        if total_n >= MIN_SAMPLES:
            p_above = float(above_y_sum / above_n) if above_n > 0 else 0.5
            p_below = float(below_y_sum / below_n) if below_n > 0 else 0.5
            proba[t] = p_above if state_is_high else p_below
        else:
            valid_y = y_arr[:t][~np.isnan(y_arr[:t])]
            proba[t] = float(np.mean(valid_y)) if valid_y.size > 0 else fallback

    # Clip to prevent exact 0/1 from edge cases (all-zero/y history)
    proba = np.clip(proba, _EPS, 1 - _EPS)
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
