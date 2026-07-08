from __future__ import annotations

from typing import Any, Callable

import lightgbm as lgb
import optuna
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.models.objectives.lgbm import objective_lgbm_classifier
from project.src.models.objectives.logistic import objective_logistic_classifier
from project.src.models.objectives.xgb import objective_xgb_classifier

ObjectiveFn = Callable[
    [optuna.Trial, Any, Any, TimeSeriesSplitWithPurge],
    float,
]

MODEL_REGISTRY: dict[str, ObjectiveFn] = {
    "logistic": objective_logistic_classifier,
    "xgb": objective_xgb_classifier,
    "lgbm": objective_lgbm_classifier,
}


def register_model(name: str, objective: ObjectiveFn) -> None:
    """Register a custom model objective for pipeline extension."""
    MODEL_REGISTRY[name] = objective


def build_model(name: str, params: dict[str, Any], random_state: int = 42):
    if name == "logistic":
        return LogisticRegression(
            C=params.get("C", 1.0),
            random_state=random_state,
            max_iter=1000,
            class_weight="balanced",
        )
    if name == "xgb":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
            **params,
        )
    if name == "lgbm":
        return lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
