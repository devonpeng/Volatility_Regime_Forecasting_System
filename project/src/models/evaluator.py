from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.features.selection import select_features_on_fold
from project.src.metrics.probabilistic import brier_score, log_loss_score
from project.src.models.registry import build_model

OOF_METRIC_NAMES = ("brier_score", "log_loss")


def compute_oof_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Compute probabilistic metrics on aligned OOF predictions."""
    return {
        "brier_score": brier_score(y_true, y_proba),
        "log_loss": log_loss_score(y_true, y_proba),
    }


def evaluate_oof(
    model_name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplitWithPurge,
    random_state: int = 42,
    feature_selection: str | None = None,
    top_k_features: int = 5,
    nested_feature_selection: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full-length OOF probabilities, valid OOF proba, and valid true labels."""
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    feature_names = list(X.columns)

    oof_proba = np.full(len(y_arr), np.nan)
    for _, (train_idx, val_idx) in enumerate(cv.split(X_arr)):
        X_train_df = pd.DataFrame(X_arr[train_idx], columns=feature_names)
        X_val_df = pd.DataFrame(X_arr[val_idx], columns=feature_names)
        y_train = y_arr[train_idx]

        if nested_feature_selection and feature_selection is not None:
            X_train_df, X_val_df, _ = select_features_on_fold(
                X_train_df,
                pd.Series(y_train),
                X_val_df,
                method=feature_selection,
                top_k=top_k_features,
                random_state=random_state,
            )

        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train_df.to_numpy())
        X_val = scaler.transform(X_val_df.to_numpy())

        if model_name == "lgbm":
            fold_feature_names = (
                list(X_train_df.columns)
                if nested_feature_selection and feature_selection is not None
                else feature_names
            )
            X_train_fit = pd.DataFrame(X_train, columns=fold_feature_names)
            X_val_fit = pd.DataFrame(X_val, columns=fold_feature_names)
        else:
            X_train_fit = X_train
            X_val_fit = X_val

        model = build_model(model_name, params, random_state=random_state)
        model.fit(X_train_fit, y_train)
        oof_proba[val_idx] = model.predict_proba(X_val_fit)[:, 1]

    valid = ~np.isnan(oof_proba)
    return oof_proba, oof_proba[valid], y_arr[valid]
