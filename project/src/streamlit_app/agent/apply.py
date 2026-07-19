"""Shared commit writers for AI Apply and Global Settings load."""
from __future__ import annotations

import uuid
from typing import Any

import streamlit as st

from project.src.models.registry import MODEL_REGISTRY
from project.src.streamlit_app.precomputed import (
    TEACHING_HORIZONS,
    TEACHING_SYMBOLS,
    cached_load_precomputed,
    date_bounds,
    extract_xy_from_wide,
    feature_column_names,
    parquet_available,
)
from project.src.streamlit_app.session_manager import (
    SS_CV_GAP,
    SS_CV_N_SPLITS,
    SS_CV_TEST_SIZE,
    SS_CV_TRAIN_WINDOW,
    SS_ENRICHED_DF,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_META_DF,
    SS_N_TRIALS,
    SS_RAW_DF,
    SS_SELECTED_FEATURES,
    SS_SELECTED_MODELS,
    SS_TARGET_COL,
    SS_TRAIN_RATIO,
    SS_X,
    SS_Y,
    clear_downstream,
    clear_from_feature_eng,
)
from project.src.streamlit_app.workspace import (
    apply_parquet_slice_to_session,
    checkpoint_current_workspace,
    request_global_setting_widget_sync,
)


def make_pending(op: str, payload: dict[str, Any], summary: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "status": "pending_approval",
        "op": op,
        "payload": payload,
        "summary": summary,
    }


def apply_action(action: dict[str, Any]) -> str:
    """Commit a pending action. Returns a short success message."""
    op = action.get("op")
    payload = dict(action.get("payload") or {})
    if op == "set_globals":
        return _apply_globals(payload)
    if op == "set_features":
        return _apply_features(payload)
    if op == "set_model_config":
        return _apply_model_config(payload)
    if op == "set_comparison_browse":
        return _apply_comparison_browse(payload)
    if op == "set_comparison_custom":
        return _apply_comparison_custom(payload)
    raise ValueError(f"Unknown action op: {op}")


def _apply_globals(payload: dict[str, Any]) -> str:
    symbol = str(payload.get("symbol", "SPY")).upper()
    horizon = int(payload.get("horizon", 5))
    start = str(payload.get("date_start", ""))[:10]
    end = str(payload.get("date_end", ""))[:10]

    if symbol not in TEACHING_SYMBOLS:
        raise ValueError(f"Symbol must be one of {TEACHING_SYMBOLS}")
    if horizon not in TEACHING_HORIZONS:
        raise ValueError(f"Horizon must be one of {TEACHING_HORIZONS}")
    if not parquet_available(symbol, horizon):
        raise FileNotFoundError(f"Teaching pack missing for {symbol}_{horizon}d")

    full = cached_load_precomputed(symbol, horizon)
    dmin, dmax = date_bounds(full)
    if start < str(dmin.date()) or end > str(dmax.date()) or start > end:
        raise ValueError(
            f"Dates must satisfy {dmin.date()} ≤ start ≤ end ≤ {dmax.date()}"
        )

    clear_from_feature_eng()
    apply_parquet_slice_to_session(symbol, horizon, start, end)
    # Widgets may already exist this run; sync gs_* on next run before they render.
    request_global_setting_widget_sync()
    checkpoint_current_workspace("ai apply globals")
    return f"Loaded {symbol} · {horizon}d · {start} → {end}"


def commit_features(
    selected_features: list[str],
    label_method: str = "expanding",
    *,
    checkpoint_reason: str = "ai apply features",
) -> str:
    """Rebuild X/y from current raw_df (parquet or live-compatible wide columns)."""
    raw_df = st.session_state.get(SS_RAW_DF)
    if raw_df is None:
        raise ValueError("No data loaded. Load a teaching pack first.")

    horizon = int(st.session_state.get(SS_HORIZON, 5))
    label_method = label_method if label_method in ("expanding", "median") else "expanding"

    available = set(feature_column_names()) | set(raw_df.columns)
    selected = [f for f in selected_features if f in available and f in raw_df.columns]
    if not selected:
        raise ValueError("No valid features to apply (none found in current data).")

    clear_downstream("feature_eng")
    X, y, meta_df, full_data = extract_xy_from_wide(
        raw_df,
        horizon=horizon,
        label_method=label_method,
        selected_features=selected,
    )
    st.session_state[SS_ENRICHED_DF] = raw_df
    st.session_state[SS_X] = X
    st.session_state[SS_Y] = y
    st.session_state[SS_META_DF] = meta_df
    st.session_state["_full_data"] = full_data
    st.session_state[SS_HORIZON] = horizon
    st.session_state[SS_LABEL_METHOD] = label_method
    st.session_state[SS_TARGET_COL] = f"rv_fwd_{horizon}d"
    st.session_state[SS_SELECTED_FEATURES] = selected
    st.session_state["_fs_version"] = st.session_state.get("_fs_version", 0) + 1
    if checkpoint_reason:
        checkpoint_current_workspace(checkpoint_reason)
    return f"Features committed: {len(selected)} cols, X={X.shape}"


def _apply_features(payload: dict[str, Any]) -> str:
    return commit_features(
        list(payload.get("selected_features") or []),
        str(payload.get("label_method", "expanding")),
    )


def _apply_model_config(payload: dict[str, Any]) -> str:
    if "cv_n_splits" in payload:
        st.session_state[SS_CV_N_SPLITS] = int(payload["cv_n_splits"])
    if "cv_train_window" in payload:
        st.session_state[SS_CV_TRAIN_WINDOW] = int(payload["cv_train_window"])
    if "cv_test_size" in payload:
        st.session_state[SS_CV_TEST_SIZE] = int(payload["cv_test_size"])
    if "cv_gap_size" in payload:
        st.session_state[SS_CV_GAP] = int(payload["cv_gap_size"])
    if "train_ratio" in payload:
        st.session_state[SS_TRAIN_RATIO] = float(payload["train_ratio"])
    if "n_trials" in payload:
        st.session_state[SS_N_TRIALS] = int(payload["n_trials"])
        st.session_state["md_n_trials"] = int(payload["n_trials"])

    models = payload.get("selected_models")
    if models is not None:
        allowed = set(MODEL_REGISTRY.keys())
        selected = [m for m in models if m in allowed]
        if not selected:
            raise ValueError(f"selected_models must intersect {sorted(allowed)}")
        st.session_state[SS_SELECTED_MODELS] = selected
        st.session_state["md_selected_models"] = selected

    # Invalidate CV object so student recreates with new params
    if st.session_state.get("cv") is not None and any(
        k in payload
        for k in ("cv_n_splits", "cv_train_window", "cv_test_size", "cv_gap_size")
    ):
        st.session_state["cv"] = None

    checkpoint_current_workspace("ai apply model config")
    return "Station 3 model/CV settings updated (recreate CV if params changed)."


def _apply_comparison_browse(payload: dict[str, Any]) -> str:
    filters = dict(payload.get("filters") or {})
    mapping = {
        "symbol": "qb_sym",
        "horizon": "qb_hrz",
        "feature_selection": "qb_sel",
        "model": "qb_model",
        "date_range": "qb_dates",
        "top_k": "qb_topk",
    }
    for src, key in mapping.items():
        if src in filters and filters[src] is not None:
            st.session_state[key] = str(filters[src])

    if "selected_ids" in payload and payload["selected_ids"] is not None:
        st.session_state["qb_compare_ids"] = [str(x) for x in payload["selected_ids"]]

    return "Comparison Quick Browse filters updated."


def _apply_comparison_custom(payload: dict[str, Any]) -> str:
    params = dict(payload.get("params") or payload)
    key_map = {
        "start": "ce_start",
        "end": "ce_end",
        "fs_method": "ce_fs",
        "top_k": "ce_topk",
        "nested": "ce_nested",
        "models": "ce_models",
        "n_trials": "ce_trials",
        "early_stop": "ce_early",
        "n_splits": "ce_ns",
        "train_window": "ce_tw",
        "test_size": "ce_ts",
        "gap": "ce_gap",
    }
    for src, key in key_map.items():
        if src not in params or params[src] is None:
            continue
        val = params[src]
        if src in ("start", "end"):
            import pandas as pd

            st.session_state[key] = pd.Timestamp(val).date()
        else:
            st.session_state[key] = val
    return "Comparison Custom Experiment params staged."
