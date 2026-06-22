from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from project.src.models.registry import build_model


def fit_best_model(
    model_name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[Any, RobustScaler, np.ndarray | pd.DataFrame, list[str]]:
    """Fit the best model on all samples for persistence."""
    feature_names = list(X.columns)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X.to_numpy())

    if model_name == "lgbm":
        X_fit: np.ndarray | pd.DataFrame = pd.DataFrame(X_scaled, columns=feature_names)
    else:
        X_fit = X_scaled

    model = build_model(model_name, params, random_state=random_state)
    model.fit(X_fit, y.to_numpy())
    return model, scaler, X_fit, feature_names
