import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.features.selection import select_features_on_fold


def objective_logistic_classifier(
    trial: optuna.Trial,
    X: np.ndarray,
    y_clf: np.ndarray,
    cv: TimeSeriesSplitWithPurge,
    *,
    feature_names: list[str] | None = None,
    feature_selection: str | None = None,
    top_k_features: int = 5,
    random_state: int = 42,
) -> float:
    logistic_params = {
        "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
        "random_state": random_state,
        "max_iter": 1000,
        "class_weight": "balanced",
    }

    auc_scores = []
    cols = feature_names or [f"f{i}" for i in range(X.shape[1])]

    for _, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train_df = pd.DataFrame(X[train_idx], columns=cols)
        X_val_df = pd.DataFrame(X[val_idx], columns=cols)
        y_train, y_val = y_clf[train_idx], y_clf[val_idx]
        if np.unique(y_train).size < 2:
            continue

        if feature_selection is not None:
            X_train_df, X_val_df, _ = select_features_on_fold(
                X_train_df,
                pd.Series(y_train),
                X_val_df,
                method=feature_selection,
                top_k=top_k_features,
                random_state=random_state,
            )

        scaler = RobustScaler()
        X_train = X_train_df.to_numpy()
        X_val = X_val_df.to_numpy()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(**logistic_params)
        model.fit(X_train_scaled, y_train)

        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        if np.unique(y_val).size < 2:
            continue

        fold_auc = roc_auc_score(y_val, y_pred_proba)
        if np.isfinite(fold_auc):
            auc_scores.append(float(fold_auc))

    if not auc_scores:
        return 0.5

    return float(np.mean(auc_scores))
