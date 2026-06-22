from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project.src.features.selection import FeatureSelectionResult
from project.src.models.fit import fit_best_model
from project.src.models.types import ModelResult
from project.src.utils.cache import RunCache


def save_run_artifacts(
    run_cache: RunCache,
    *,
    pipeline_config: dict[str, Any],
    cv_config: dict[str, Any],
    data_source: str,
    raw_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
    model_results: list[ModelResult],
    best_model_name: str,
    best_params: dict[str, Any],
    best_auc: float,
    oof_metrics: dict[str, float],
    oof_proba_full: np.ndarray,
    feature_selection_result: FeatureSelectionResult | None,
    selected_features: list[str],
    random_state: int,
) -> Path:
    """Persist config, data, best model weights, and OOF predictions for one run."""
    run_cache.save_json(
        "pipeline.json",
        {
            **pipeline_config,
            "data_source": data_source,
            "best_model_name": best_model_name,
            "best_auc": best_auc,
        },
    )
    run_cache.save_json("cv.json", cv_config)
    run_cache.save_json("oof_metrics.json", oof_metrics)
    run_cache.save_json("best_params.json", {"model": best_model_name, "params": best_params})
    run_cache.save_json(
        "model_comparison.json",
        {
            "models": [
                {
                    "name": result.name,
                    "best_auc": result.best_auc,
                    "best_params": result.best_params,
                }
                for result in model_results
            ]
        },
    )

    if feature_selection_result is not None:
        run_cache.save_json(
            "feature_selection.json",
            {
                "method": feature_selection_result.method,
                "selected_features": feature_selection_result.selected_features,
                "scores": feature_selection_result.scores.to_dict(),
            },
        )
    else:
        run_cache.save_json(
            "feature_selection.json",
            {"method": None, "selected_features": selected_features},
        )

    run_cache.save_csv("raw.csv", raw_df)
    run_cache.save_csv("X.csv", X.reset_index(drop=True))
    run_cache.save_csv("y.csv", pd.DataFrame({"y": y.reset_index(drop=True)}))
    run_cache.save_csv("meta.csv", meta_df.reset_index(drop=True))

    oof_df = meta_df.copy().reset_index(drop=True)
    oof_df["oof_proba"] = oof_proba_full
    run_cache.save_csv("oof_predictions.csv", oof_df)

    model, scaler, _, feature_names = fit_best_model(
        best_model_name,
        best_params,
        X,
        y,
        random_state=random_state,
    )
    run_cache.save_model_bundle(model, scaler, feature_names, best_model_name)
    return run_cache.root
