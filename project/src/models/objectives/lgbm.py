import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from io import StringIO
from contextlib import redirect_stderr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.features.selection import select_features_on_fold

_ES_THRESHOLD = 5000


def _train_with_warn_once(params, dtrain, valid_sets=None, num_boost_round=100, callbacks=None):
    """Train LightGBM model; suppress repeated split warnings during training,
    then print any collected warnings exactly once after training."""
    effective_params = dict(params)
    effective_params["verbose"] = -1

    log = StringIO()
    with redirect_stderr(log):
        model = lgb.train(
            params=effective_params,
            train_set=dtrain,
            valid_sets=valid_sets,
            num_boost_round=num_boost_round,
            callbacks=callbacks,
        )

    text = log.getvalue()
    if "No further splits with positive gain" in text:
        lines = [line.strip() for line in text.splitlines()
                 if "No further splits with positive gain" in line]
        unique_lines = list(dict.fromkeys(lines))  # preserve order, deduplicate
        if unique_lines:
            print(f"[LightGBM] No further splits warning (suppressed duplicates):\n" + "\n".join(unique_lines))
    return model


def objective_lgbm_classifier(
    trial: optuna.Trial,
    X: np.ndarray,
    y_clf: np.ndarray,
    cv: TimeSeriesSplitWithPurge,
    *,
    feature_names: list[str] | None = None,
    feature_selection: str | None = None,
    top_k_features: int = 5,
    random_state: int = 42,
    **kwargs,
) -> float:
    # 限制 n_estimators 上限，高于 300 的树收益递减且训练耗时大
    n_estimators = trial.suggest_int("n_estimators", 150, 300, step=50)
    use_es = X.shape[0] > _ES_THRESHOLD

    lgbm_params = {
        "objective": "binary",
        "metric": "auc",
        # lr 上限从 0.5 降到 0.3，减少方差并加快收敛
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 60),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": random_state,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }

    auc_scores = []
    cols = feature_names or [f"f{i}" for i in range(X.shape[1])]

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train_df = pd.DataFrame(X[train_idx], columns=cols)
        X_val_df = pd.DataFrame(X[val_idx], columns=cols)
        y_train, y_val = y_clf[train_idx], y_clf[val_idx]
        if np.unique(y_train).size < 2 or np.unique(y_val).size < 2:
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
        X_train_scaled = scaler.fit_transform(X_train_df.to_numpy())
        X_val_scaled = scaler.transform(X_val_df.to_numpy())

        dtrain = lgb.Dataset(X_train_scaled, label=y_train)
        dval = lgb.Dataset(X_val_scaled, label=y_val, reference=dtrain)

        callbacks = []
        if use_es:
            # es 轮数从 50 降到 30，更快终止过拟合的训练
            callbacks.append(lgb.callback.early_stopping(stopping_rounds=30))

        gbm = _train_with_warn_once(
            params=lgbm_params,
            dtrain=dtrain,
            valid_sets=[dval],
            num_boost_round=n_estimators,
            callbacks=callbacks,
        )

        y_pred_proba = gbm.predict(X_val_scaled)

        fold_auc = roc_auc_score(y_val, y_pred_proba)
        if not np.isfinite(fold_auc):
            continue

        auc_scores.append(float(fold_auc))

        # ── 启用 Optuna Pruning ──────────────────────────────
        # HyperbandPruner 需要每个 iteration 的 score 才能判断是否剪枝
        trial.report(float(fold_auc), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
        # ─────────────────────────────────────────────────────

    if not auc_scores:
        return 0.5

    return float(np.mean(auc_scores))
