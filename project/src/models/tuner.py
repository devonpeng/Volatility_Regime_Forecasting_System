from __future__ import annotations

import logging
import warnings

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
        # Only act after we have enough reports
        values = [v for v in trial.intermediate_values.values() if v is not None]
        if len(values) < 2:
            return False

        # Check whether any value improved over everything seen before it
        for i in range(len(values)):
            running_max = max(values[:i + 1])
            if values[i] != running_max:
                continue
            # values[i] set a new high-water mark – reset counter from here
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
) -> ModelResult:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")

    X_arr = X.to_numpy()
    y_arr = y.to_numpy()
    objective = MODEL_REGISTRY[name]

    # Determine actual number of folds (respects auto_adjust)
    _ = list(cv.split(X_arr))
    n_folds = cv.get_n_splits(X_arr)

    # ── Warning: too many CV splits make tuning very slow ──────────────
    if n_folds > 10:
        warnings.warn(
            f"n_splits={n_folds} (>10). This will significantly increase "
            f"training time. Consider reducing to 5–10 splits.",
            stacklevel=2,
        )
    # ──────────────────────────────────────────────────────────────────

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        pruner=_PatiencePruner(patience=early_stop_patience),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            X_arr,
            y_arr,
            cv,
            feature_names=list(X.columns),
            feature_selection=feature_selection if nested_feature_selection else None,
            top_k_features=top_k_features,
            random_state=random_state,
            early_stopping=early_stopping,
        ),
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
