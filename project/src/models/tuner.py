from __future__ import annotations

import logging
import warnings
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.models.registry import MODEL_REGISTRY
from project.src.models.types import ModelResult

logger = logging.getLogger(__name__)


class _PatiencePruner(optuna.pruners.BasePruner):
    """Trial-level early stopping: prune if best score doesn't improve
    for ``patience`` consecutive completed folds."""

    def __init__(self, patience: int = 5):
        self._patience = patience

    def prune(self, study: optuna.study.Study, trial: optuna.trial.Trial) -> bool:
        values = [v for v in trial.intermediate_values.values() if v is not None]
        if len(values) < 2:
            return False

        for i in range(len(values)):
            running_max = max(values[: i + 1])
            if values[i] != running_max:
                continue
            counter = 0
            for j in range(i + 1, len(values)):
                if values[j] > running_max:
                    counter = 0
                    running_max = values[j]
                else:
                    counter += 1
            if counter >= self._patience:
                return True
        return False


def create_study(*, early_stop_patience: int = 5) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna.create_study(
        direction="maximize",
        pruner=_PatiencePruner(patience=early_stop_patience),
    )


def _objective_fn(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplitWithPurge,
    *,
    feature_selection: str | None = None,
    top_k_features: int = 5,
    nested_feature_selection: bool = False,
    random_state: int = 42,
    early_stopping: bool = True,
) -> Callable[[optuna.Trial], float]:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    objective = MODEL_REGISTRY[name]
    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    feature_names = list(X.columns)

    def _fn(trial: optuna.Trial) -> float:
        return objective(
            trial,
            X_arr,
            y_arr,
            cv,
            feature_names=feature_names,
            feature_selection=feature_selection if nested_feature_selection else None,
            top_k_features=top_k_features,
            random_state=random_state,
            early_stopping=early_stopping,
        )

    return _fn


def study_to_result(name: str, study: optuna.Study) -> ModelResult:
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(t.value)
    ]
    if not completed:
        return ModelResult(
            name=name,
            best_auc=0.5,
            best_params={},
            study=study,
        )
    return ModelResult(
        name=name,
        best_auc=float(study.best_value),
        best_params=study.best_params,
        study=study,
    )


def run_one_trial(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplitWithPurge,
    study: optuna.Study,
    *,
    feature_selection: str | None = None,
    top_k_features: int = 5,
    nested_feature_selection: bool = False,
    random_state: int = 42,
    early_stopping: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> optuna.Study:
    """Run a single Optuna trial on *study* (mutates and returns study)."""
    obj = _objective_fn(
        name,
        X,
        y,
        cv,
        feature_selection=feature_selection,
        top_k_features=top_k_features,
        nested_feature_selection=nested_feature_selection,
        random_state=random_state,
        early_stopping=early_stopping,
    )

    def _callback(study_: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if should_stop and should_stop():
            study_.stop()

    study.optimize(
        obj,
        n_trials=1,
        show_progress_bar=False,
        callbacks=[_callback],
    )
    return study


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplitWithPurge,
    n_trials: int = 30,
    feature_selection: str | None = None,
    top_k_features: int = 5,
    nested_feature_selection: bool = False,
    random_state: int = 42,
    early_stopping: bool = True,
    early_stop_patience: int = 5,
    study: optuna.Study | None = None,
) -> ModelResult:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")

    X_arr = X.to_numpy()
    _ = list(cv.split(X_arr))
    n_folds = cv.get_n_splits(X_arr)

    if n_folds > 10:
        warnings.warn(
            f"n_splits={n_folds} (>10). This will significantly increase "
            f"training time. Consider reducing to 5–10 splits.",
            stacklevel=2,
        )

    if study is None:
        study = create_study(early_stop_patience=early_stop_patience)

    obj = _objective_fn(
        name,
        X,
        y,
        cv,
        feature_selection=feature_selection,
        top_k_features=top_k_features,
        nested_feature_selection=nested_feature_selection,
        random_state=random_state,
        early_stopping=early_stopping,
    )
    study.optimize(obj, n_trials=n_trials, show_progress_bar=True)
    return study_to_result(name, study)


def tune_models(
    names: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplitWithPurge,
    n_trials: int = 30,
    feature_selection: str | None = None,
    top_k_features: int = 5,
    nested_feature_selection: bool = False,
    random_state: int = 42,
    early_stopping: bool = True,
    early_stop_patience: int = 5,
) -> list[ModelResult]:
    results: list[ModelResult] = []
    for name in names:
        print(f"Tuning model: {name} ({n_trials} trials)...")
        results.append(
            tune_model(
                name,
                X,
                y,
                cv,
                n_trials=n_trials,
                feature_selection=feature_selection,
                top_k_features=top_k_features,
                nested_feature_selection=nested_feature_selection,
                random_state=random_state,
                early_stopping=early_stopping,
                early_stop_patience=early_stop_patience,
            )
        )
    return results
