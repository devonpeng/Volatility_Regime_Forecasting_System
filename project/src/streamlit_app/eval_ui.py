"""Shared helpers for Station 4/5 eval deep-dive and holdout predictions."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from project.src.eval.reporting import run_dm_evaluation
from project.src.models.registry import build_model
from project.src.pipline import RVPipeline
from project.src.streamlit_app.precomputed import (
    date_range_presets,
    extract_xy_from_wide,
    load_grid_recipe,
    relative_run_dir,
    resolve_run_dir,
    run_dir_for_experiment,
    slice_by_dates,
)


def _fs(method: str | None) -> str | None:
    if method in (None, "", "none", "None"):
        return None
    return method


def _cv_params_for_row(row: dict) -> dict[str, int]:
    recipe = load_grid_recipe()
    profiles = recipe.get("cv_profiles") or {}
    key = str(row.get("cv_profile") or "default")
    cv = profiles.get(key) or profiles.get("default") or {
        "n_splits": 12,
        "train_window": 252,
        "test_size": 21,
        "gap": 5,
    }
    return {
        "n_splits": int(row.get("n_splits") or cv.get("n_splits", 12)),
        "train_window_size": int(row.get("train_window") or cv.get("train_window", 252)),
        "test_size": int(row.get("test_size") or cv.get("test_size", 21)),
        "gap_size": int(row.get("gap") or cv.get("gap", 5)),
    }


def resolve_dates_for_row(row: dict) -> tuple[str, str, str]:
    """Return (date_range_key, start, end)."""
    presets = date_range_presets()
    key = str(row.get("date_range") or "custom")
    if key in presets:
        return key, presets[key]["start"], presets[key]["end"]
    start = str(row.get("start") or presets.get("full", {}).get("start", "2005-01-01"))
    end = str(row.get("end") or presets.get("full", {}).get("end", "2026-03-01"))
    return key, start, end


def rebuild_experiment_run(
    row: dict,
    full_df: pd.DataFrame,
    *,
    label_method: str | None = None,
) -> Path:
    """Re-run RVPipeline for *row* and persist under precomputed runs/."""
    exp_id = str(row.get("experiment_id") or f"custom_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}")
    date_key, start, end = resolve_dates_for_row(row)
    horizon = int(row.get("horizon") or 5)
    symbol = str(row.get("symbol") or "SPY")
    fs_raw = row.get("feature_selection")
    fs = _fs(None if pd.isna(fs_raw) else str(fs_raw))
    top_k = int(row.get("top_k") or 12)
    model = str(row.get("model") or "lgbm")
    lm = label_method or str(row.get("label_method") or "expanding")
    n_trials = int(row.get("n_trials") or 15)
    cv = _cv_params_for_row(row)

    sliced = slice_by_dates(full_df, start, end)
    X, y, meta_df, _ = extract_xy_from_wide(
        sliced,
        horizon=horizon,
        label_method=lm,
    )
    pipe = RVPipeline(
        symbol=symbol,
        start_date=start,
        end_date=end,
        horizon=horizon,
        models=[model],
        n_trials=n_trials,
        label_method=lm,
        feature_selection=fs,
        nested_feature_selection=False,
        top_k_features=top_k,
        n_splits=cv["n_splits"],
        train_window_size=cv["train_window_size"],
        test_size=cv["test_size"],
        gap_size=cv["gap_size"],
    )
    pipe.persist_artifacts = False
    pipe.run_id = exp_id
    pipe.inject_dataset(X, y, meta_df, raw_df=sliced)
    pipe.run(persist=False)

    run_path = run_dir_for_experiment(exp_id)
    if run_path.exists():
        shutil.rmtree(run_path)
    pipe.persist_run_artifacts(run_path)
    row["run_dir"] = relative_run_dir(run_path)
    row["date_range"] = date_key
    row["start"] = start
    row["end"] = end
    return run_path


def ensure_run_dir(row: dict, full_df: pd.DataFrame | None = None) -> Path:
    """Hybrid: prefer disk artifacts, else rebuild (requires *full_df*)."""
    existing = resolve_run_dir(row=row)
    if existing is not None:
        return existing
    if full_df is None:
        raise FileNotFoundError(
            f"No run artifacts for {row.get('experiment_id')}; load teaching pack to rebuild."
        )
    return rebuild_experiment_run(row, full_df)


def load_eval_report(run_dir: Path, *, regime_method: str = "kmeans", n_regimes: int = 2) -> dict[str, Any]:
    return run_dm_evaluation(
        Path(run_dir),
        save_report=False,
        regime_method=regime_method,
        n_regimes=n_regimes,
        shap_max_samples=50,
    )


def holdout_predictions_from_run(
    run_dir: Path,
    *,
    train_ratio: float = 0.8,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Fit on train slice, return (test_dates, y_test, test_proba)."""
    from sklearn.preprocessing import RobustScaler

    run_dir = Path(run_dir)
    x_path = run_dir / "data" / "X.csv"
    y_path = run_dir / "data" / "y.csv"
    meta_path = run_dir / "data" / "meta.csv"
    params_path = run_dir / "config" / "best_params.json"
    if not x_path.is_file() or not y_path.is_file():
        raise FileNotFoundError(f"Missing X/y under {run_dir / 'data'}")

    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = pd.Series(y).astype(int)

    if meta_path.is_file():
        meta = pd.read_csv(meta_path)
        dates = pd.to_datetime(meta["Date"]) if "Date" in meta.columns else pd.RangeIndex(len(X))
    else:
        dates = pd.RangeIndex(len(X))

    params_payload = {}
    if params_path.is_file():
        params_payload = json.loads(params_path.read_text(encoding="utf-8"))
    model_name = str(params_payload.get("model") or "lgbm")
    best_params = params_payload.get("params") or {}

    split_idx = max(1, min(len(X) - 1, int(len(X) * float(train_ratio))))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    test_dates = dates.iloc[split_idx:] if hasattr(dates, "iloc") else dates[split_idx:]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train.to_numpy())
    X_test_s = scaler.transform(X_test.to_numpy())
    model = build_model(model_name, best_params)
    if model_name == "lgbm":
        model.fit(pd.DataFrame(X_train_s, columns=X_train.columns), y_train.to_numpy())
        proba = model.predict_proba(pd.DataFrame(X_test_s, columns=X_test.columns))[:, 1]
    else:
        model.fit(X_train_s, y_train.to_numpy())
        proba = model.predict_proba(X_test_s)[:, 1]
    return pd.DatetimeIndex(pd.to_datetime(test_dates)), y_test.to_numpy(), np.asarray(proba)


def oof_series_from_report(report: dict) -> tuple[list, np.ndarray, np.ndarray]:
    dates = report.get("_dates") or list(range(len(report.get("_y_valid") or [])))
    y = np.asarray(report.get("_y_valid") or [], dtype=int)
    proba = np.asarray(report.get("_model_proba") or [], dtype=float)
    return dates, y, proba


def actual_rv_aligned_to_dates(
    dates,
    *,
    horizon: int,
    full_df: pd.DataFrame | None,
) -> np.ndarray | None:
    """Align ``rv_fwd_{horizon}d`` from *full_df* to *dates* (same length)."""
    if full_df is None or "Date" not in full_df.columns:
        return None
    col = f"rv_fwd_{int(horizon)}d"
    if col not in full_df.columns:
        return None
    src = full_df[["Date", col]].copy()
    src["Date"] = pd.to_datetime(src["Date"])
    src = src.drop_duplicates(subset=["Date"], keep="last").set_index("Date")
    idx = pd.to_datetime(pd.Series(dates))
    aligned = src.reindex(idx)[col]
    return aligned.to_numpy(dtype=float)


def regime_bands_aligned_to_dates(
    dates,
    *,
    full_df: pd.DataFrame | None,
    start_date: str | None,
    end_date: str | None,
    n_regimes: int = 2,
    method: str = "kmeans",
) -> tuple[np.ndarray | None, list | None]:
    """Cluster macro regimes for *dates*; return (labels, cluster_order) or (None, None)."""
    try:
        from project.src.eval.regime import build_macro_regime_labels

        idx = pd.to_datetime(pd.Series(dates))
        if len(idx) == 0:
            return None, None

        frame = pd.DataFrame({"Date": idx})
        if full_df is not None and "Date" in full_df.columns and "Close" in full_df.columns:
            src = full_df[["Date", "Close"]].copy()
            src["Date"] = pd.to_datetime(src["Date"])
            src = src.drop_duplicates(subset=["Date"], keep="last")
            frame = frame.merge(src, on="Date", how="left")

        start = str(start_date or "").strip()
        end = str(end_date or "").strip()
        if not start:
            start = str(pd.Timestamp(idx.min()).date())
        if not end:
            end = str(pd.Timestamp(idx.max()).date())

        regimes, _name, _meta, _macro, cluster_meta = build_macro_regime_labels(
            frame,
            {"start_date": start, "end_date": end},
            ma_window=30,
            method=method,
            n_regimes=int(n_regimes),
        )
        labels = np.asarray(regimes, dtype=object)
        if len(labels) != len(idx):
            labels = labels[: len(idx)]
        order = list(cluster_meta.get("cluster_order") or [])
        return labels, order
    except Exception:
        return None, None


def holdout_predictions_from_session(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
    model_name: str,
    best_params: dict,
    train_ratio: float = 0.8,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Station 4 helper: train_ratio holdout from in-memory session arrays."""
    from sklearn.preprocessing import RobustScaler

    split_idx = max(1, min(len(X) - 1, int(len(X) * float(train_ratio))))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates = pd.to_datetime(meta_df["Date"]) if meta_df is not None and "Date" in meta_df.columns else pd.RangeIndex(len(X))
    test_dates = dates.iloc[split_idx:] if hasattr(dates, "iloc") else dates[split_idx:]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train.to_numpy())
    X_test_s = scaler.transform(X_test.to_numpy())
    model = build_model(model_name, best_params)
    if model_name == "lgbm":
        model.fit(pd.DataFrame(X_train_s, columns=X_train.columns), y_train.to_numpy())
        proba = model.predict_proba(pd.DataFrame(X_test_s, columns=X_test.columns))[:, 1]
    else:
        model.fit(X_train_s, y_train.to_numpy())
        proba = model.predict_proba(X_test_s)[:, 1]
    return pd.DatetimeIndex(pd.to_datetime(test_dates)), y_test.to_numpy(), np.asarray(proba)
