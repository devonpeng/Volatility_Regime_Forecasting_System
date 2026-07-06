import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from project.src.data.spliter import TimeSeriesSplitWithPurge


def objective_lgbm_classifier(
    trial: optuna.Trial,
    X: np.ndarray,
    y_clf: np.ndarray,
    cv: TimeSeriesSplitWithPurge,
) -> float:
    lgbm_params = {
        "objective": "binary",
        "metric": "auc",
        "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 60),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
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

        model = lgb.LGBMClassifier(**lgbm_params)
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
