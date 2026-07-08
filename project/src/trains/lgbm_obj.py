import lightgbm as lgb
import numpy as np
import optuna
from io import StringIO
from contextlib import redirect_stderr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from project.src.data.spliter import TimeSeriesSplitWithPurge


def _fit_model_with_warn_once(model, X, y=None, eval_set=None, callbacks=None):
    """Fit LightGBM model; suppress repeated split warnings during training,
    then print any collected warnings exactly once after fitting."""
    log = StringIO()
    with redirect_stderr(log):
        model.fit(X, y, eval_set=eval_set, callbacks=callbacks)

    text = log.getvalue()
    if "No further splits with positive gain" in text:
        lines = [line.strip() for line in text.splitlines()
                 if "No further splits with positive gain" in line]
        unique_lines = list(dict.fromkeys(lines))
        if unique_lines:
            print(f"[LightGBM] No further splits warning (suppressed duplicates):\n" + "\n".join(unique_lines))
    return model


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

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_clf[train_idx], y_clf[val_idx]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        callbacks = [lgb.callback.early_stopping(stopping_rounds=50)]
        model = lgb.LGBMClassifier(**lgbm_params)
        model = _fit_model_with_warn_once(
            model=model,
            X=X_train_scaled,
            y=y_train,
            eval_set=[(X_val_scaled, y_val)],
            callbacks=callbacks,
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
