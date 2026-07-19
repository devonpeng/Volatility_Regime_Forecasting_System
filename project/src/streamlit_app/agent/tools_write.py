"""Propose-write tools — validate and return pending actions (no session mutation)."""
from __future__ import annotations

import json
from typing import Any

import streamlit as st
from langchain_core.tools import tool

from project.src.models.registry import MODEL_REGISTRY
from project.src.streamlit_app.agent.apply import make_pending
from project.src.streamlit_app.precomputed import (
    TEACHING_HORIZONS,
    TEACHING_SYMBOLS,
    cached_load_precomputed,
    date_bounds,
    feature_column_names,
    parquet_available,
)
from project.src.streamlit_app.session_manager import SS_RAW_DF


def _ok(pending: dict[str, Any]) -> str:
    return json.dumps(pending, default=str)


def _err(msg: str) -> str:
    return json.dumps({"status": "error", "error": msg})


@tool
def propose_set_globals(
    symbol: str,
    horizon: int,
    date_start: str,
    date_end: str,
) -> str:
    """Propose loading a teaching pack (symbol/horizon/date range). Requires human Apply."""
    symbol = symbol.upper().strip()
    horizon = int(horizon)
    date_start = str(date_start)[:10]
    date_end = str(date_end)[:10]
    if symbol not in TEACHING_SYMBOLS:
        return _err(f"symbol must be one of {list(TEACHING_SYMBOLS)}")
    if horizon not in TEACHING_HORIZONS:
        return _err(f"horizon must be one of {list(TEACHING_HORIZONS)}")
    if not parquet_available(symbol, horizon):
        return _err(f"pack missing for {symbol}_{horizon}d")
    try:
        full = cached_load_precomputed(symbol, horizon)
        dmin, dmax = date_bounds(full)
        if date_start < str(dmin.date()) or date_end > str(dmax.date()) or date_start > date_end:
            return _err(
                f"dates must satisfy {dmin.date()} ≤ start ≤ end ≤ {dmax.date()}"
            )
    except Exception as exc:
        return _err(str(exc))

    pending = make_pending(
        "set_globals",
        {
            "symbol": symbol,
            "horizon": horizon,
            "date_start": date_start,
            "date_end": date_end,
        },
        f"Load {symbol} · {horizon}d · {date_start} → {date_end}",
    )
    return _ok(pending)


@tool
def propose_set_features(
    selected_features: list[str],
    label_method: str = "expanding",
) -> str:
    """Propose Station 2 feature selection + label method. Requires human Apply."""
    if label_method not in ("expanding", "median"):
        return _err("label_method must be 'expanding' or 'median'")
    raw = st.session_state.get(SS_RAW_DF)
    registry = set(feature_column_names())
    if raw is not None:
        valid = [f for f in selected_features if f in raw.columns]
    else:
        valid = [f for f in selected_features if f in registry]
    if not valid:
        return _err("No valid features in proposal (load data / check names).")
    pending = make_pending(
        "set_features",
        {"selected_features": valid, "label_method": label_method},
        f"Set {len(valid)} features · label={label_method}",
    )
    return _ok(pending)


@tool
def propose_set_model_config(
    cv_n_splits: int | None = None,
    cv_train_window: int | None = None,
    cv_test_size: int | None = None,
    cv_gap_size: int | None = None,
    train_ratio: float | None = None,
    n_trials: int | None = None,
    selected_models: list[str] | None = None,
) -> str:
    """Propose Station 3 CV / training settings. Requires human Apply."""
    payload: dict[str, Any] = {}
    if cv_n_splits is not None:
        payload["cv_n_splits"] = int(cv_n_splits)
    if cv_train_window is not None:
        payload["cv_train_window"] = int(cv_train_window)
    if cv_test_size is not None:
        payload["cv_test_size"] = int(cv_test_size)
    if cv_gap_size is not None:
        payload["cv_gap_size"] = int(cv_gap_size)
    if train_ratio is not None:
        payload["train_ratio"] = float(train_ratio)
    if n_trials is not None:
        payload["n_trials"] = int(n_trials)
    if selected_models is not None:
        allowed = set(MODEL_REGISTRY.keys())
        models = [m for m in selected_models if m in allowed]
        if not models:
            return _err(f"selected_models must include one of {sorted(allowed)}")
        payload["selected_models"] = models
    if not payload:
        return _err("Provide at least one model config field.")
    pending = make_pending(
        "set_model_config",
        payload,
        f"Update model/CV config: {', '.join(payload.keys())}",
    )
    return _ok(pending)


@tool
def propose_set_comparison_browse(
    symbol: str | None = None,
    horizon: str | None = None,
    feature_selection: str | None = None,
    model: str | None = None,
    date_range: str | None = None,
    top_k: str | None = None,
    selected_ids: list[str] | None = None,
) -> str:
    """Propose Quick Browse filters / selected experiment ids. Requires human Apply."""
    filters: dict[str, Any] = {}
    for key, val in [
        ("symbol", symbol),
        ("horizon", horizon),
        ("feature_selection", feature_selection),
        ("model", model),
        ("date_range", date_range),
        ("top_k", top_k),
    ]:
        if val is not None:
            filters[key] = str(val)
    payload: dict[str, Any] = {"filters": filters}
    if selected_ids is not None:
        payload["selected_ids"] = [str(x) for x in selected_ids]
    if not filters and selected_ids is None:
        return _err("Provide at least one filter or selected_ids.")
    pending = make_pending(
        "set_comparison_browse",
        payload,
        f"Comparison browse: filters={filters}, ids={selected_ids}",
    )
    return _ok(pending)


@tool
def propose_set_comparison_custom(
    start: str | None = None,
    end: str | None = None,
    fs_method: str | None = None,
    top_k: int | None = None,
    nested: bool | None = None,
    models: list[str] | None = None,
    n_trials: int | None = None,
    early_stop: int | None = None,
    n_splits: int | None = None,
    train_window: int | None = None,
    test_size: int | None = None,
    gap: int | None = None,
) -> str:
    """Propose Custom Experiment widget values on Station 5. Requires human Apply."""
    params: dict[str, Any] = {}
    for key, val in [
        ("start", start),
        ("end", end),
        ("fs_method", fs_method),
        ("top_k", top_k),
        ("nested", nested),
        ("models", models),
        ("n_trials", n_trials),
        ("early_stop", early_stop),
        ("n_splits", n_splits),
        ("train_window", train_window),
        ("test_size", test_size),
        ("gap", gap),
    ]:
        if val is not None:
            params[key] = val
    if not params:
        return _err("Provide at least one custom experiment param.")
    if models is not None:
        allowed = {"lgbm", "xgb", "logistic"}
        params["models"] = [m for m in models if m in allowed]
    pending = make_pending(
        "set_comparison_custom",
        {"params": params},
        f"Stage custom experiment: {list(params.keys())}",
    )
    return _ok(pending)


WRITE_TOOLS = [
    propose_set_globals,
    propose_set_features,
    propose_set_model_config,
    propose_set_comparison_browse,
    propose_set_comparison_custom,
]
