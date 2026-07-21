import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.features.selection import select_features_on_fold

_ES_THRESHOLD = 5000


def objective_xgb_classifier(
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

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        # lr 上限从 0.5 降到 0.3，减少方差并加快收敛
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "random_state": random_state,
        "tree_method": "hist",
    }

    auc_scores = []
    cols = feature_names or [f"f{i}" for i in range(X.shape[1])]

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train_df = pd.DataFrame(X[train_idx], columns=cols)
        X_val_df = pd.DataFrame(X[val_idx], columns=cols)
        y_train, y_val = y_clf[train_idx], y_clf[val_idx]
        # AUC is undefined for single-class folds; skip before xgb.train
        # (eval_metric=auc on a one-class eval set floods stderr warnings).
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

        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)

        callbacks = []
        if use_es:
            # es 轮数从 50 降到 30，更快终止过拟合的训练
            callbacks.append(xgb.callback.EarlyStopping(rounds=30))

        bst = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=[(dval, "validation")],
            callbacks=callbacks,
            verbose_eval=False,
        )

        y_pred_proba = bst.predict(dval)

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
