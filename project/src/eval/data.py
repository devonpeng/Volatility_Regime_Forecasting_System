from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project.src.utils.cache import get_cache_root, load_metadata, lookup_dataset_dir


# Columns needed by baselines that survive feature selection → no lookup needed
_DIRECTLY_AVAILABLE = {"rv_hist_5d", "garch_vol_1d"}


def _build_cache_key(*, symbol: str, start_date: str, end_date: str,
                     horizon: int, label_method: str,
                     use_synthetic: bool,
                     feature_names: list[str]) -> str | None:
    """Reconstruct the SHA-256 hash used during build_features."""
    try:
        import hashlib
        payload = {
            "symbol": symbol.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "horizon": int(horizon),
            "label_method": label_method,
            "use_synthetic": bool(use_synthetic),
            "feature_names": sorted(feature_names),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    except Exception:
        return None


def _load_feature_from_dataset_cache(
    needed_cols: list[str],
    *,
    valid_n: int,
    x_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Search metadata.json for a dataset whose X.csv contains all needed_cols
    AND has enough overlapping rows with the run's sample space.

    Returns dict[col_name -> numpy_array_for_valid_rows].
    """
    root = get_cache_root()
    ds_root = root / "datasets"
    if not ds_root.is_dir():
        return {}

    meta = load_metadata(root)
    candidates: list[tuple[str, dict[str, np.ndarray]]] = []

    for cache_key, info in meta.items():
        feat_list = info.get("feature_names", [])
        if not all(c in feat_list for c in needed_cols):
            continue

        ds_path = root / info.get("dataset_dir", "")
        if not ds_path.is_dir():
            continue

        x_path = ds_path / "X.csv"
        if not x_path.exists():
            continue

        try:
            candidate_x = pd.read_csv(x_path)
        except Exception:
            continue

        n_ds = len(candidate_x)
        n_run = len(x_df)
        # The dataset must cover roughly the same sample space
        if abs(n_ds - n_run) > max(50, int(0.05 * n_run)):
            continue

        # Extract needed columns
        cols_found: dict[str, np.ndarray] = {}
        for col in needed_cols:
            if col in candidate_x.columns:
                cols_found[col] = candidate_x[col].to_numpy(dtype=float)
        
        if cols_found:
            candidates.append((cache_key, cols_found))

    if not candidates:
        return {}

    # Use candidate with most extracted columns
    best_key, result = max(candidates, key=lambda c: len(c[1]))
    
    # Trim to match valid sample size (both filtered to same NaN-free subset)
    return {col: arr[:valid_n] for col, arr in result.items()}


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

    # ── Try loading full X (pre-selection) first; fall back to selected X ──
    x_full_df = None
    x_all_path = run_dir / "data" / "X_all.csv"
    x_selected_df = None
    if x_all_path.exists():
        x_full_df = pd.read_csv(x_all_path)
    elif x_path.exists():
        x_selected_df = pd.read_csv(x_path)

    if x_full_df is not None or x_selected_df is not None:
        x_df = x_full_df if x_full_df is not None else x_selected_df
        valid_n = int(valid.sum())

        # Step 1: Load features directly available in x_df
        for col in _DIRECTLY_AVAILABLE:
            if col in x_df.columns:
                valid_df[col] = x_df.loc[valid, col].to_numpy(dtype=float)

        # Step 2: Look up missing features from dataset cache via metadata.json
        symbols_needed = [c for c in _DIRECTLY_AVAILABLE if c not in x_df.columns]
        if symbols_needed:
            loaded = _load_feature_from_dataset_cache(
                symbols_needed, valid_n=valid_n, x_df=x_df
            )
            for col in symbols_needed:
                if col in loaded:
                    arr = loaded[col]
                    if len(arr) >= valid_n:
                        valid_df[col] = arr[:valid_n]

        # Keep full matrices for fold-wise GARCH logistic calibration etc.
        pipeline_config["_x_full"] = x_df
        pipeline_config["_y_full"] = y_df["y"]
        pipeline_config["_oof_valid_mask"] = valid.to_numpy()

    return y_valid, model_proba, horizon, valid_df, best_model_name, pipeline_config
