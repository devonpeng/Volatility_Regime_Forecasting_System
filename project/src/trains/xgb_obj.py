import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from project.src.data.spliter import TimeSeriesSplitWithPurge


def objective_xgb_classifier(
    trial: optuna.Trial,
    X: np.ndarray,
    y_clf: np.ndarray,
    cv: TimeSeriesSplitWithPurge,
) -> float:
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",
    }

    auc_scores = []
    scaler = RobustScaler()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_clf[train_idx], y_clf[val_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        callbacks = [xgb.callback.EarlyStopping(
            rounds=50,
            checkpoint=True,
            save_best=True,
        )]
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=callbacks,
            verbose=False,
        )

        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        fold_auc = roc_auc_score(y_val, y_pred_proba)
        if not np.isfinite(fold_auc):
            continue

        auc_scores.append(float(fold_auc))
        trial.report(fold_auc, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    if not auc_scores:
        return 0.0

    return float(np.mean(auc_scores))
