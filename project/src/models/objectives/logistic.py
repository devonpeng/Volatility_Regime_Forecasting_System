import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from project.src.data.spliter import TimeSeriesSplitWithPurge


def objective_logistic_classifier(
    trial: optuna.Trial,
    X: np.ndarray,
    y_clf: np.ndarray,
    cv: TimeSeriesSplitWithPurge,
) -> float:
    logistic_params = {
        "C": trial.suggest_float("C", 1e-2, 10.0, log=True),
        "random_state": 42,
        "max_iter": 1000,
        "class_weight": "balanced",
    }

    auc_scores = []
    scaler = RobustScaler()

    for _, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_clf[train_idx], y_clf[val_idx]
        if np.unique(y_train).size < 2:
            continue

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
