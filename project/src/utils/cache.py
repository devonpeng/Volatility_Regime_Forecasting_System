from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv

import functools


@functools.lru_cache(maxsize=1)
def get_cache_root(path: str = "./cache") -> Path:
    """Root cache directory (shared data + per-run artifacts)."""
    load_dotenv()
    cache_path = os.getenv("CACHE_PATH", path)
    root = Path(cache_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    get_data_dir(root)
    get_runs_dir(root)
    return root


def get_cache_path(path: str = "./cache") -> Path:
    """Backward-compatible alias for cache root."""
    return get_cache_root(path)


def get_data_dir(root: Path | None = None) -> Path:
    """Shared market-data cache (yfinance CSV)."""
    data_dir = (root or get_cache_root()) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_runs_dir(root: Path | None = None) -> Path:
    """Parent directory for per-run artifact folders."""
    runs_dir = (root or get_cache_root()) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def get_model_dir(path: str = "./cache") -> Path:
    """Backward-compatible alias; points to shared runs dir."""
    return get_runs_dir(get_cache_root(path))


def make_run_dir_name(model_name: str, feature_selection: str | None) -> str:
    """Format: {model}-{feature_selection}-{timestamp}."""
    fs = feature_selection or "none"
    safe_model = _sanitize_name(model_name)
    safe_fs = _sanitize_name(fs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_model}-{safe_fs}-{timestamp}"


def _sanitize_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-") or "unknown"


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class RunCache:
    """
    Per-run artifact store under cache/runs/{model}-{feature_selection}-{time}/.

    Layout:
        config/   pipeline, cv, params, feature selection, model comparison
        data/     raw, features, labels, oof predictions
        model/    best model weights + scaler
        plots/    diagnostic figures
    """

    def __init__(self, run_dir: Path):
        self.root = run_dir
        self.config_dir = run_dir / "config"
        self.data_dir = run_dir / "data"
        self.model_dir = run_dir / "model"
        self.plots_dir = run_dir / "plots"
        for directory in (self.config_dir, self.data_dir, self.model_dir, self.plots_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(
        cls,
        model_name: str,
        feature_selection: str | None,
        parent: Path | None = None,
    ) -> RunCache:
        parent = parent or get_runs_dir()
        run_dir = parent / make_run_dir_name(model_name, feature_selection)
        return cls(run_dir)

    def save_json(self, filename: str, payload: dict[str, Any], *, subdir: str = "config") -> Path:
        target_dir = self.root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        return path

    def save_csv(self, filename: str, df: pd.DataFrame, *, subdir: str = "data") -> Path:
        target_dir = self.root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        df.to_csv(path, index=False)
        return path

    def save_model_bundle(
        self,
        model: Any,
        scaler: Any,
        feature_names: list[str],
        model_name: str,
    ) -> dict[str, Path]:
        bundle = {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "model_name": model_name,
        }
        model_path = self.model_dir / "best_model.joblib"
        joblib.dump(bundle, model_path)
        return {"bundle": model_path}

    @classmethod
    def load_model_bundle(cls, run_dir: Path | str) -> dict[str, Any]:
        path = Path(run_dir) / "model" / "best_model.joblib"
        return joblib.load(path)
