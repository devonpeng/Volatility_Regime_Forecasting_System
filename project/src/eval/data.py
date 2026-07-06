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

    if not y_path.exists() or not oof_path.exists():
        raise FileNotFoundError(
            f"Missing y.csv or oof_predictions.csv under {run_dir / 'data'}"
        )

    y = pd.read_csv(y_path)["y"].to_numpy()
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

    valid_df = oof_df.loc[valid].reset_index(drop=True)
    valid_df["y"] = y_valid
    valid_df["model_proba"] = model_proba
    if x_path.exists():
        x_df = pd.read_csv(x_path)
        if "rv_hist_5d" in x_df.columns:
            valid_df["rv_hist_5d"] = x_df.loc[valid, "rv_hist_5d"].to_numpy(dtype=float)
    return y_valid, model_proba, horizon, valid_df, best_model_name, pipeline_config
