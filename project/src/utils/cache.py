from __future__ import annotations

import functools
import hashlib
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


@functools.lru_cache(maxsize=1)
def get_cache_root(path: str = "./cache") -> Path:
    """Root cache directory (shared data + per-run artifacts)."""
    load_dotenv()
    cache_path = os.getenv("CACHE_PATH", path)
    root = Path(cache_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    get_data_dir(root)
    get_runs_dir(root)
    get_datasets_dir(root)
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


def get_datasets_dir(root: Path | None = None) -> Path:
    """Shared pre-selection feature datasets keyed by content hash."""
    datasets_dir = (root or get_cache_root()) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    return datasets_dir


def get_metadata_path(root: Path | None = None) -> Path:
    """Index of shared dataset caches (outside runs/)."""
    return (root or get_cache_root()) / "metadata.json"


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


def dataset_cache_key(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int,
    label_method: str,
    use_synthetic: bool,
    feature_names: list[str],
) -> str:
    """Stable sha256 for build_features inputs (pre feature-selection)."""
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


def load_metadata(root: Path | None = None) -> dict[str, Any]:
    path = get_metadata_path(root)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_metadata(metadata: dict[str, Any], root: Path | None = None) -> Path:
    path = get_metadata_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=_json_default)
    return path


def lookup_dataset_dir(cache_key: str, root: Path | None = None) -> Path | None:
    """Return datasets/{hash} if metadata points to a complete cache entry."""
    root = root or get_cache_root()
    entry = load_metadata(root).get(cache_key)
    if not isinstance(entry, dict):
        return None
    dataset_dir = entry.get("dataset_dir")
    if not dataset_dir:
        return None
    path = Path(dataset_dir)
    if not path.is_absolute():
        path = root / path
    required = ("raw.csv", "X.csv", "y.csv", "meta.csv")
    if not all((path / name).exists() for name in required):
        return None
    return path


def load_cached_dataset(
    cache_key: str,
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame] | None:
    """Load raw/X/y/meta for a hashed build_features cache hit."""
    root = root or get_cache_root()
    dataset_dir = lookup_dataset_dir(cache_key, root=root)
    if dataset_dir is None:
        return None

    entry = load_metadata(root).get(cache_key, {})
    raw_df = pd.read_csv(dataset_dir / "raw.csv")
    if "Date" in raw_df.columns:
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    raw_df.attrs["source"] = entry.get("data_source", "dataset_cache")
    raw_df.attrs["symbol"] = entry.get("symbol", "")

    X = pd.read_csv(dataset_dir / "X.csv")
    y = pd.read_csv(dataset_dir / "y.csv")["y"]
    meta_df = pd.read_csv(dataset_dir / "meta.csv")
    if "Date" in meta_df.columns:
        meta_df["Date"] = pd.to_datetime(meta_df["Date"])
    return raw_df, X, y, meta_df


def save_cached_dataset(
    cache_key: str,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    horizon: int,
    label_method: str,
    use_synthetic: bool,
    feature_names: list[str],
    raw_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    meta_df: pd.DataFrame,
    root: Path | None = None,
) -> Path:
    """Persist full pre-selection features and update metadata.json."""
    root = root or get_cache_root()
    dataset_dir = get_datasets_dir(root) / cache_key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    raw_df.to_csv(dataset_dir / "raw.csv", index=False)
    X.reset_index(drop=True).to_csv(dataset_dir / "X.csv", index=False)
    pd.DataFrame({"y": y.reset_index(drop=True)}).to_csv(dataset_dir / "y.csv", index=False)
    meta_df.reset_index(drop=True).to_csv(dataset_dir / "meta.csv", index=False)

    metadata = load_metadata(root)
    metadata[cache_key] = {
        "symbol": symbol.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "horizon": int(horizon),
        "label_method": label_method,
        "use_synthetic": bool(use_synthetic),
        "feature_names": sorted(feature_names),
        "dataset_dir": str(Path("datasets") / cache_key),
        "data_source": raw_df.attrs.get("source", "unknown"),
    }
    save_metadata(metadata, root=root)
    return dataset_dir


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
