from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.models.registry import MODEL_REGISTRY
from project.src.models.types import ModelResult


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplitWithPurge,
    n_trials: int = 30,
) -> ModelResult:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")

    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    objective = MODEL_REGISTRY[name]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_arr, y_arr, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

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


def tune_models(
    names: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    cv: TimeSeriesSplitWithPurge,
    n_trials: int = 30,
) -> list[ModelResult]:
    results: list[ModelResult] = []
    for name in names:
        print(f"Tuning model: {name} ({n_trials} trials)...")
        results.append(tune_model(name, X, y, cv, n_trials=n_trials))
    return results
