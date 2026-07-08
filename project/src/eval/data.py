from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_run_data(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, int, pd.DataFrame, str, dict[str, Any]]:
    y_path = run_dir / "data" / "y.csv"
    oof_path = run_dir / "data" / "oof_predictions.csv"
    x_path = run_dir / "data" / "X.csv"
    pipeline_path = run_dir / "config" / "pipeline.json"
    cv_path = run_dir / "config" / "cv.json"

    if not y_path.exists() or not oof_path.exists():
        raise FileNotFoundError(
            f"Missing y.csv or oof_predictions.csv under {run_dir / 'data'}"
        )

    y_df = pd.read_csv(y_path)
    y = y_df["y"].to_numpy()
    oof_df = pd.read_csv(oof_path)
    if "oof_proba" not in oof_df.columns:
        raise ValueError("oof_predictions.csv must contain column 'oof_proba'")

    valid = ~oof_df["oof_proba"].isna()
    y_valid = y[valid.to_numpy()]
    model_proba = oof_df.loc[valid, "oof_proba"].to_numpy(dtype=float)

    horizon = 5
    best_model_name = "unknown"
    pipeline_config: dict[str, Any] = {}
    if pipeline_path.exists():
        with pipeline_path.open(encoding="utf-8") as f:
            pipeline_config = json.load(f)
        horizon = int(pipeline_config.get("horizon", 5))
        best_model_name = str(pipeline_config.get("best_model_name", "unknown"))

    cv_config: dict[str, Any] = {}
    if cv_path.exists():
        with cv_path.open(encoding="utf-8") as f:
            cv_config = json.load(f)
    pipeline_config["_cv_config"] = cv_config

    valid_df = oof_df.loc[valid].reset_index(drop=True)
    valid_df["y"] = y_valid
    valid_df["model_proba"] = model_proba

    # Try loading full X (pre-selection) first; fall back to selected X
    x_full_df = None
    x_all_path = run_dir / "data" / "X_all.csv"
    x_selected_df = None
    if x_all_path.exists():
        x_full_df = pd.read_csv(x_all_path)
    elif x_path.exists():
        x_selected_df = pd.read_csv(x_path)

    if x_full_df is not None or x_selected_df is not None:
        x_df = x_full_df if x_full_df is not None else x_selected_df
        if "rv_hist_5d" in x_df.columns:
            valid_df["rv_hist_5d"] = x_df.loc[valid, "rv_hist_5d"].to_numpy(dtype=float)
        # Keep full matrices for fold-wise GARCH logistic calibration etc.
        pipeline_config["_x_full"] = x_df
        pipeline_config["_y_full"] = y_df["y"]
        pipeline_config["_oof_valid_mask"] = valid.to_numpy()
    return y_valid, model_proba, horizon, valid_df, best_model_name, pipeline_config
