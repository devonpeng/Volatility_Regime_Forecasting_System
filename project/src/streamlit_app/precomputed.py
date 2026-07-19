"""Helpers for teaching-pack Parquet load / slice / dataset extraction."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from project.src.features.core import DEFAULT_FEATURE_STEPS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRECOMPUTED_DIR = PROJECT_ROOT / "cache" / "precomputed"
MANIFEST_PATH = PRECOMPUTED_DIR / "manifest.json"

TEACHING_SYMBOLS = ("SPY", "QQQ")
TEACHING_HORIZONS = (5, 10)
DEFAULT_PACK_START = "2005-01-01"
DEFAULT_PACK_END = "2026-03-01"

RAW_OHLCV = ("Date", "Open", "High", "Low", "Close", "Volume")
RAW_MACRO = ("vix", "vix3m", "vvix", "skew", "tnx", "dxy", "gold")


def feature_column_names() -> list[str]:
    return [step.name for step in DEFAULT_FEATURE_STEPS]


def parquet_path(symbol: str, horizon: int) -> Path:
    return PRECOMPUTED_DIR / f"{symbol.upper()}_{int(horizon)}d.parquet"


def grid_results_path() -> Path:
    return PRECOMPUTED_DIR / "grid_results.parquet"


def precomputed_runs_root() -> Path:
    return PRECOMPUTED_DIR / "runs"


def run_dir_for_experiment(experiment_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(experiment_id))
    return precomputed_runs_root() / (safe or "unknown")


def run_dir_is_complete(run_dir: Path) -> bool:
    """True when artifacts needed by ``load_run_data`` exist."""
    return (
        run_dir.is_dir()
        and (run_dir / "data" / "y.csv").is_file()
        and (run_dir / "data" / "oof_predictions.csv").is_file()
        and (run_dir / "config" / "pipeline.json").is_file()
    )


def resolve_run_dir(experiment_id: str | None = None, *, row: dict | None = None) -> Path | None:
    """Resolve a precomputed run directory from a row or experiment id."""
    if row:
        raw = row.get("run_dir")
        if raw:
            path = Path(str(raw))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if run_dir_is_complete(path):
                return path
        eid = row.get("experiment_id") or experiment_id
        if eid:
            path = run_dir_for_experiment(str(eid))
            if run_dir_is_complete(path):
                return path
            return None
    if experiment_id:
        path = run_dir_for_experiment(str(experiment_id))
        if run_dir_is_complete(path):
            return path
    return None


def load_grid_recipe() -> dict:
    path = PROJECT_ROOT / "configs" / "grid_recipe.yaml"
    if not path.is_file():
        return {}
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def date_range_presets() -> dict[str, dict[str, str]]:
    """Named date ranges from ``grid_recipe.yaml`` (full / pre_2020 / post_2020)."""
    recipe = load_grid_recipe()
    ranges = recipe.get("date_ranges") or {}
    if ranges:
        return {
            str(k): {"start": str(v["start"]), "end": str(v["end"])}
            for k, v in ranges.items()
            if isinstance(v, dict) and "start" in v and "end" in v
        }
    return {
        "full": {"start": DEFAULT_PACK_START, "end": DEFAULT_PACK_END},
        "pre_2020": {"start": "2005-01-01", "end": "2019-12-31"},
        "post_2020": {"start": "2020-01-01", "end": "2025-12-31"},
    }


def relative_run_dir(run_dir: Path) -> str:
    """Store paths relative to project root when possible."""
    try:
        return str(run_dir.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(run_dir.resolve())


def parquet_available(symbol: str, horizon: int) -> bool:
    return parquet_path(symbol, horizon).is_file()


def load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_precomputed_df(symbol: str, horizon: int) -> pd.DataFrame:
    path = parquet_path(symbol, horizon)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing teaching pack: {path}. "
            "Run: uv run python scripts/precompute_data.py"
        )
    df = pd.read_parquet(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df.attrs["source"] = "precomputed"
    df.attrs["symbol"] = symbol.upper()
    return df


def cached_load_precomputed(symbol: str, horizon: int) -> pd.DataFrame:
    """Streamlit-friendly cached loader (no-op cache outside Streamlit)."""
    try:
        import streamlit as st

        @st.cache_resource
        def _load(sym: str, hrz: int) -> pd.DataFrame:
            return load_precomputed_df(sym, hrz)

        return _load(symbol, horizon)
    except Exception:
        return load_precomputed_df(symbol, horizon)

def date_bounds(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(df["Date"])
    return dates.min(), dates.max()


def slice_by_dates(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out["Date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    mask = (dates >= start_ts) & (dates <= end_ts)
    return out.loc[mask].reset_index(drop=True)


def label_column(label_method: str) -> str:
    if label_method == "expanding":
        return "y_expanding"
    if label_method in ("median", "rolling"):
        return "y_median"
    raise ValueError(f"Unknown label_method: {label_method}")


def extract_xy_from_wide(
    df: pd.DataFrame,
    *,
    horizon: int,
    label_method: str,
    selected_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build X, y, meta_df, full_valid from a wide precomputed frame.

    Returns (X, y, meta_df, full_valid_df) where full_valid_df retains all
    columns on valid rows (for importance / SHAP).
    """
    feats = selected_features or [
        c for c in feature_column_names() if c in df.columns
    ]
    target_col = f"rv_fwd_{horizon}d"
    y_col = label_column(label_method)
    if target_col not in df.columns:
        raise KeyError(f"Missing target column {target_col}")
    if y_col not in df.columns:
        raise KeyError(f"Missing label column {y_col}")

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:8]}")

    work = df.copy()
    work["y"] = work[y_col].astype(int)
    required = feats + [target_col, "y"]
    valid = work[required].notna().all(axis=1)
    data = work.loc[valid].reset_index(drop=True)

    X = data[feats].copy()
    y = data["y"].copy()
    meta_cols = [c for c in ("Date", "Close", target_col, "y") if c in data.columns]
    meta_df = data[meta_cols].copy()
    return X, y, meta_df, data
