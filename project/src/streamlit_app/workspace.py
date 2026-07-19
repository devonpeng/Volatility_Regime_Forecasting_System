"""Workspace restore, checkpoint, and tab-version sync."""
from __future__ import annotations

import datetime as dt
import json
import re
from typing import Any

import pandas as pd
import streamlit as st

from project.src.streamlit_app import db
from project.src.streamlit_app.precomputed import (
    cached_load_precomputed,
    slice_by_dates,
)
from project.src.streamlit_app.session_manager import (
    GS_END,
    GS_HORIZON,
    GS_START,
    GS_SYMBOL,
    SS_CUSTOM_EXPERIMENTS,
    SS_CV_GAP,
    SS_CV_N_SPLITS,
    SS_CV_TEST_SIZE,
    SS_CV_TRAIN_WINDOW,
    SS_DATA_SOURCE,
    SS_DATE_END,
    SS_DATE_START,
    SS_ENRICHED_DF,
    SS_FULL_DF,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_META_DF,
    SS_MODEL_RESULTS,
    SS_OOF_METRICS,
    SS_PENDING_GS_SYNC,
    SS_RAW_DF,
    SS_SELECTED_FEATURES,
    SS_SELECTED_MODELS,
    SS_SYMBOL,
    SS_TRAIN_RATIO,
    SS_WELCOME_DONE,
    SS_WORKSPACE_CONFIG,
    SS_WORKSPACE_ID,
    SS_WORKSPACE_NAME,
    SS_WS_VERSION,
    SS_X,
    SS_Y,
    build_workspace_config_snapshot,
)

WORKSPACE_EXPORT_FORMAT = "rv_lab_workspace"
WORKSPACE_EXPORT_VERSION = 1


def export_workspace_envelope() -> dict[str, Any]:
    """Build a JSON-serializable export payload for the current workspace."""
    cfg = build_workspace_config_snapshot()
    name = st.session_state.get(SS_WORKSPACE_NAME) or "workspace"
    return {
        "format": WORKSPACE_EXPORT_FORMAT,
        "version": WORKSPACE_EXPORT_VERSION,
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "name": str(name),
        "config": cfg,
    }


def export_workspace_json_bytes() -> tuple[bytes, str]:
    """Return (json_bytes, suggested_filename) for download."""
    envelope = export_workspace_envelope()
    raw_name = str(envelope.get("name") or "workspace")
    safe = re.sub(r"[^\w\-]+", "_", raw_name).strip("_") or "workspace"
    filename = f"workspace_{safe}.json"
    payload = json.dumps(envelope, indent=2, default=str).encode("utf-8")
    return payload, filename


def parse_workspace_import(raw: bytes | str) -> dict[str, Any]:
    """Validate an uploaded workspace JSON and return ``{name, config}``."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = raw
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Workspace file must be a JSON object.")
    if data.get("format") != WORKSPACE_EXPORT_FORMAT:
        raise ValueError(
            f"Unsupported format (expected {WORKSPACE_EXPORT_FORMAT!r})."
        )
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("Workspace file missing a config object.")
    name = str(data.get("name") or "Imported").strip() or "Imported"
    return {"name": name, "config": cfg}


def import_workspace_as_new(user_id: str, name: str, config: dict[str, Any]) -> str:
    """Create a new workspace from import and restore it. Returns workspace id."""
    ws = db.create_workspace(user_id, name, config)
    restore_workspace(ws["id"])
    return ws["id"]


def import_workspace_overwrite_current(config: dict[str, Any]) -> str:
    """Overwrite the active workspace config and restore. Returns workspace id."""
    ws_id = st.session_state.get(SS_WORKSPACE_ID)
    if not ws_id:
        raise ValueError("No active workspace to overwrite.")
    version = db.checkpoint_workspace(ws_id, config, bump_version=True)
    st.session_state[SS_WS_VERSION] = version
    st.session_state[SS_WORKSPACE_CONFIG] = config
    restore_workspace(ws_id)
    return ws_id


def sync_global_setting_widgets(*, force: bool = False) -> None:
    """Mirror applied SS_* globals into gs_* draft widget keys.

    Must run *before* widgets with those keys are instantiated.
    """
    if force or GS_SYMBOL not in st.session_state:
        st.session_state[GS_SYMBOL] = st.session_state.get(SS_SYMBOL, "SPY")
    if force or GS_HORIZON not in st.session_state:
        st.session_state[GS_HORIZON] = int(st.session_state.get(SS_HORIZON, 5))

    def _as_date(val: Any, fallback: str) -> dt.date:
        try:
            return pd.Timestamp(val).date()
        except Exception:
            return pd.Timestamp(fallback).date()

    if force or GS_START not in st.session_state:
        st.session_state[GS_START] = _as_date(
            st.session_state.get(SS_DATE_START), "2010-01-01"
        )
    if force or GS_END not in st.session_state:
        st.session_state[GS_END] = _as_date(
            st.session_state.get(SS_DATE_END), "2024-12-31"
        )


def request_global_setting_widget_sync() -> None:
    """Defer gs_* sync until the next run, before widgets are created."""
    st.session_state[SS_PENDING_GS_SYNC] = True


def flush_pending_global_setting_sync() -> None:
    """Apply a pending deferred sync (safe only before gs_* widgets exist)."""
    if st.session_state.pop(SS_PENDING_GS_SYNC, False):
        sync_global_setting_widgets(force=True)


def checkpoint_current_workspace(reason: str = "") -> None:
    """Persist config snapshot and bump version."""
    ws_id = st.session_state.get(SS_WORKSPACE_ID)
    if not ws_id:
        return
    cfg = build_workspace_config_snapshot()
    st.session_state[SS_WORKSPACE_CONFIG] = cfg
    version = db.checkpoint_workspace(ws_id, cfg, bump_version=True)
    st.session_state[SS_WS_VERSION] = version
    if reason:
        st.toast(f"Workspace saved ({reason})")


def check_tab_sync_warning() -> None:
    ws_id = st.session_state.get(SS_WORKSPACE_ID)
    if not ws_id:
        return
    remote = db.get_workspace_version(ws_id)
    local = int(st.session_state.get(SS_WS_VERSION, 0))
    if remote > local:
        st.warning(
            "This workspace was updated in another tab. Refresh to load the latest state."
        )
        if st.button("Refresh workspace now", key="ws_refresh_btn"):
            restore_workspace(ws_id)
            st.rerun()


def apply_parquet_slice_to_session(
    symbol: str,
    horizon: int,
    start: str,
    end: str,
) -> None:
    full_df = cached_load_precomputed(symbol, horizon)
    raw_df = slice_by_dates(full_df, start, end)
    st.session_state[SS_FULL_DF] = full_df
    st.session_state[SS_RAW_DF] = raw_df
    st.session_state[SS_ENRICHED_DF] = raw_df
    st.session_state[SS_SYMBOL] = symbol.upper()
    st.session_state[SS_HORIZON] = int(horizon)
    st.session_state[SS_DATE_START] = str(start)[:10]
    st.session_state[SS_DATE_END] = str(end)[:10]
    st.session_state[SS_DATA_SOURCE] = "parquet"
    st.session_state[SS_WELCOME_DONE] = True


def _clear_pipeline_frames() -> None:
    for key in (
        SS_RAW_DF,
        SS_FULL_DF,
        SS_ENRICHED_DF,
        SS_X,
        SS_Y,
        SS_META_DF,
        SS_MODEL_RESULTS,
        SS_OOF_METRICS,
        "X_train",
        "X_test",
        "y_train",
        "y_test",
        "cv",
        "_full_data",
        "_feature_importances",
        "_shap_bundle",
        "_llm_report",
    ):
        st.session_state[key] = None if key != SS_MODEL_RESULTS else None


def restore_workspace(workspace_id: str) -> None:
    ws = db.get_workspace(workspace_id)
    if ws is None:
        raise ValueError(f"Workspace not found: {workspace_id}")

    cfg: dict[str, Any] = ws.get("config") or {}
    _clear_pipeline_frames()

    st.session_state[SS_WORKSPACE_ID] = ws["id"]
    st.session_state[SS_WORKSPACE_NAME] = ws["name"]
    st.session_state[SS_WS_VERSION] = int(ws.get("version", 0))
    st.session_state[SS_WORKSPACE_CONFIG] = cfg
    st.session_state[SS_CUSTOM_EXPERIMENTS] = list(cfg.get("experiments") or [])

    welcome_done = bool(cfg.get("welcome_done"))
    symbol = cfg.get("symbol", "SPY")
    horizon = int(cfg.get("horizon", 5))
    date_range = cfg.get("date_range") or ["2010-01-01", "2024-12-31"]
    start, end = date_range[0], date_range[1]
    data_source = cfg.get("data_source", "parquet")

    st.session_state[SS_SYMBOL] = symbol
    st.session_state[SS_HORIZON] = horizon
    st.session_state[SS_DATE_START] = start
    st.session_state[SS_DATE_END] = end
    st.session_state[SS_LABEL_METHOD] = cfg.get("label_method", "expanding")
    st.session_state[SS_SELECTED_FEATURES] = list(cfg.get("selected_features") or [])
    st.session_state[SS_DATA_SOURCE] = data_source
    st.session_state[SS_WELCOME_DONE] = welcome_done
    if cfg.get("selected_models"):
        st.session_state[SS_SELECTED_MODELS] = list(cfg["selected_models"])
    if "train_ratio" in cfg:
        st.session_state[SS_TRAIN_RATIO] = float(cfg["train_ratio"])
    if "n_trials" in cfg:
        st.session_state["n_trials"] = int(cfg["n_trials"])

    cv = cfg.get("cv_config") or {}
    if cv:
        st.session_state[SS_CV_N_SPLITS] = cv.get("n_splits", 12)
        st.session_state[SS_CV_TRAIN_WINDOW] = cv.get("train_window", 252)
        st.session_state[SS_CV_TEST_SIZE] = cv.get("test_size", 21)
        st.session_state[SS_CV_GAP] = cv.get("gap", 5)

    sync_global_setting_widgets(force=True)

    if welcome_done and data_source == "parquet":
        try:
            apply_parquet_slice_to_session(symbol, horizon, start, end)
            sync_global_setting_widgets(force=True)
            # Rebuild X/y if features were saved
            feats = list(cfg.get("selected_features") or [])
            if feats:
                from project.src.streamlit_app.precomputed import extract_xy_from_wide

                X, y, meta_df, full_data = extract_xy_from_wide(
                    st.session_state[SS_RAW_DF],
                    horizon=horizon,
                    label_method=cfg.get("label_method", "expanding"),
                    selected_features=feats,
                )
                st.session_state[SS_X] = X
                st.session_state[SS_Y] = y
                st.session_state[SS_META_DF] = meta_df
                st.session_state["_full_data"] = full_data
                st.session_state["target_col"] = f"rv_fwd_{horizon}d"
        except FileNotFoundError as exc:
            st.session_state[SS_WELCOME_DONE] = False
            st.warning(str(exc))
    elif welcome_done and data_source == "live":
        st.session_state[SS_DATA_SOURCE] = "live"
        st.session_state[SS_WELCOME_DONE] = True
