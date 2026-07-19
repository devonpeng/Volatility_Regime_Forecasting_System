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
    SS_BEST_MODEL_NAME,
    SS_BEST_PARAMS,
    SS_CUSTOM_EXPERIMENTS,
    SS_CV,
    SS_CV_GAP,
    SS_CV_N_SPLITS,
    SS_CV_TEST_SIZE,
    SS_CV_TRAIN_WINDOW,
    SS_DATA_SOURCE,
    SS_DATE_END,
    SS_DATE_START,
    SS_ENRICHED_DF,
    SS_FEATURE_SELECTION,
    SS_FULL_DF,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_META_DF,
    SS_MODEL_RESULTS,
    SS_N_TRIALS,
    SS_OOF_METRICS,
    SS_PENDING_COMPARE_SYNC,
    SS_PENDING_FEATURE_SYNC,
    SS_PENDING_GS_SYNC,
    SS_PENDING_MODEL_SYNC,
    SS_RAW_DF,
    SS_SELECTED_FEATURES,
    SS_SELECTED_MODELS,
    SS_SYMBOL,
    SS_TOP_K,
    SS_TRAIN_RATIO,
    SS_WELCOME_DONE,
    SS_WORKSPACE_CONFIG,
    SS_WORKSPACE_ID,
    SS_WORKSPACE_NAME,
    SS_WS_VERSION,
    SS_X,
    SS_X_TEST,
    SS_X_TRAIN,
    SS_Y,
    SS_Y_TEST,
    SS_Y_TRAIN,
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


def request_feature_widget_sync() -> None:
    """Defer feature multiselect / label widget seeding until next run."""
    st.session_state[SS_PENDING_FEATURE_SYNC] = True


def flush_pending_feature_sync() -> None:
    """Bump feat_sel_* keys and seed from SS_SELECTED_FEATURES / label."""
    if not st.session_state.pop(SS_PENDING_FEATURE_SYNC, False):
        return
    selected = list(st.session_state.get(SS_SELECTED_FEATURES) or [])
    label = st.session_state.get(SS_LABEL_METHOD, "expanding")
    if label not in ("expanding", "median"):
        label = "expanding"
    v = int(st.session_state.get("_fs_version", 0)) + 1
    st.session_state["_fs_version"] = v
    st.session_state[f"feat_sel_{v}"] = selected
    st.session_state[f"feat_sel_live_{v}"] = selected
    st.session_state["feng_label_method"] = label
    st.session_state["feng_label_method_live"] = label


def request_model_widget_sync() -> None:
    """Defer Station 3 md_* widget seeding until next run."""
    st.session_state[SS_PENDING_MODEL_SYNC] = True


def flush_pending_model_sync() -> None:
    """Copy SS_* model/CV settings into md_* widget keys."""
    if not st.session_state.pop(SS_PENDING_MODEL_SYNC, False):
        return
    st.session_state["md_train_ratio"] = float(st.session_state.get(SS_TRAIN_RATIO, 0.8))
    st.session_state["md_cv_n_splits"] = int(st.session_state.get(SS_CV_N_SPLITS, 12))
    st.session_state["md_cv_train_window"] = int(
        st.session_state.get(SS_CV_TRAIN_WINDOW, 252)
    )
    st.session_state["md_cv_test_size"] = int(st.session_state.get(SS_CV_TEST_SIZE, 21))
    st.session_state["md_cv_gap"] = int(st.session_state.get(SS_CV_GAP, 5))
    st.session_state["md_n_trials"] = int(st.session_state.get(SS_N_TRIALS, 30))
    models = list(st.session_state.get(SS_SELECTED_MODELS) or [])
    if models:
        st.session_state["md_selected_models"] = models


def request_comparison_widget_sync() -> None:
    """Defer Station 5 qb_*/ce_* widget seeding until next run."""
    st.session_state[SS_PENDING_COMPARE_SYNC] = True


def flush_pending_comparison_sync() -> None:
    """Apply staged comparison browse/custom payloads into widget keys."""
    if not st.session_state.pop(SS_PENDING_COMPARE_SYNC, False):
        return

    browse = st.session_state.pop("_pending_compare_browse", None)
    if isinstance(browse, dict):
        filters = dict(browse.get("filters") or {})
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
        if browse.get("selected_ids") is not None:
            st.session_state["qb_compare_ids"] = [
                str(x) for x in browse["selected_ids"]
            ]

    custom = st.session_state.pop("_pending_compare_custom", None)
    if isinstance(custom, dict):
        params = dict(custom.get("params") or custom)
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
        if params.get("start") is not None or params.get("end") is not None:
            st.session_state["ce_date_preset"] = "custom"
        for src, key in key_map.items():
            if src not in params or params[src] is None:
                continue
            val = params[src]
            if src in ("start", "end"):
                st.session_state[key] = pd.Timestamp(val).date()
            else:
                st.session_state[key] = val


def flush_pending_station_widget_syncs() -> None:
    """Flush all deferred station widget syncs (call before page render)."""
    flush_pending_feature_sync()
    flush_pending_model_sync()
    flush_pending_comparison_sync()


def rehydrate_from_workspace_config(
    cfg: dict[str, Any] | None = None,
    *,
    overwrite_selection: bool = False,
) -> list[str]:
    """Rebuild missing pipeline objects from workspace config + parquet.

    When *overwrite_selection* is False (Station 2/3/4 soft rehydrate), do **not**
    clobber in-progress UI feature/label drafts from an older workspace snapshot.

    Returns a list of human-readable steps that were performed.
    """
    cfg = dict(cfg if cfg is not None else (st.session_state.get(SS_WORKSPACE_CONFIG) or {}))
    done: list[str] = []
    if not cfg:
        return done

    symbol = cfg.get("symbol", st.session_state.get(SS_SYMBOL, "SPY"))
    horizon = int(cfg.get("horizon", st.session_state.get(SS_HORIZON, 5)))
    date_range = cfg.get("date_range") or [
        st.session_state.get(SS_DATE_START, "2010-01-01"),
        st.session_state.get(SS_DATE_END, "2024-12-31"),
    ]
    start, end = str(date_range[0])[:10], str(date_range[1])[:10]
    data_source = cfg.get("data_source", st.session_state.get(SS_DATA_SOURCE, "parquet"))
    label_method = cfg.get(
        "label_method", st.session_state.get(SS_LABEL_METHOD, "expanding")
    )
    cfg_feats = list(cfg.get("selected_features") or [])

    # Committed selection from workspace — only force into session on full restore.
    if overwrite_selection:
        if cfg.get("selected_features") is not None:
            st.session_state[SS_SELECTED_FEATURES] = list(cfg_feats)
        if "label_method" in cfg:
            st.session_state[SS_LABEL_METHOD] = label_method

    # XY rebuild uses committed config features, not a live UI draft.
    feats = cfg_feats
    label_for_xy = st.session_state.get(SS_LABEL_METHOD, label_method)

    if "train_ratio" in cfg and (
        overwrite_selection or st.session_state.get(SS_TRAIN_RATIO) is None
    ):
        st.session_state[SS_TRAIN_RATIO] = float(cfg["train_ratio"])
    if "n_trials" in cfg and (
        overwrite_selection or st.session_state.get(SS_N_TRIALS) is None
    ):
        st.session_state[SS_N_TRIALS] = int(cfg["n_trials"])
    if cfg.get("selected_models") and (
        overwrite_selection or not st.session_state.get(SS_SELECTED_MODELS)
    ):
        st.session_state[SS_SELECTED_MODELS] = list(cfg["selected_models"])
    cv_cfg = cfg.get("cv_config") or {}
    if cv_cfg:
        # CV params are safe to re-sync; they don't fight Station 2 multiselect.
        st.session_state[SS_CV_N_SPLITS] = int(cv_cfg.get("n_splits", 12))
        st.session_state[SS_CV_TRAIN_WINDOW] = int(cv_cfg.get("train_window", 252))
        st.session_state[SS_CV_TEST_SIZE] = int(cv_cfg.get("test_size", 21))
        st.session_state[SS_CV_GAP] = int(cv_cfg.get("gap", 5))
    if cfg.get("best_model_name") and (
        overwrite_selection or not st.session_state.get(SS_BEST_MODEL_NAME)
    ):
        st.session_state[SS_BEST_MODEL_NAME] = cfg["best_model_name"]
    if cfg.get("best_params") is not None and (
        overwrite_selection or not st.session_state.get(SS_BEST_PARAMS)
    ):
        st.session_state[SS_BEST_PARAMS] = dict(cfg.get("best_params") or {})
    if "feature_selection" in cfg and (
        overwrite_selection
        or SS_FEATURE_SELECTION not in st.session_state
        or st.session_state.get(SS_FEATURE_SELECTION) is None
    ):
        st.session_state[SS_FEATURE_SELECTION] = cfg.get("feature_selection")
    if "top_k" in cfg and overwrite_selection:
        st.session_state[SS_TOP_K] = int(cfg.get("top_k", 5))

    if st.session_state.get(SS_RAW_DF) is None and data_source == "parquet":
        try:
            apply_parquet_slice_to_session(str(symbol), horizon, start, end)
            done.append("raw_df")
        except FileNotFoundError:
            return done

    raw_df = st.session_state.get(SS_RAW_DF)
    if (
        st.session_state.get(SS_X) is None or st.session_state.get(SS_Y) is None
    ) and raw_df is not None and feats:
        from project.src.streamlit_app.precomputed import extract_xy_from_wide

        X, y, meta_df, full_data = extract_xy_from_wide(
            raw_df,
            horizon=horizon,
            label_method=label_for_xy,
            selected_features=feats,
        )
        st.session_state[SS_X] = X
        st.session_state[SS_Y] = y
        st.session_state[SS_META_DF] = meta_df
        st.session_state["_full_data"] = full_data
        st.session_state[SS_ENRICHED_DF] = raw_df
        st.session_state["target_col"] = f"rv_fwd_{horizon}d"
        done.append("X/y")

    X = st.session_state.get(SS_X)
    y = st.session_state.get(SS_Y)
    if (
        st.session_state.get(SS_X_TRAIN) is None
        and X is not None
        and y is not None
    ):
        ratio = float(st.session_state.get(SS_TRAIN_RATIO, cfg.get("train_ratio", 0.8)))
        split_idx = int(len(X) * ratio)
        if 0 < split_idx < len(X):
            st.session_state[SS_X_TRAIN] = X.iloc[:split_idx].copy().reset_index(drop=True)
            st.session_state[SS_X_TEST] = X.iloc[split_idx:].copy().reset_index(drop=True)
            st.session_state[SS_Y_TRAIN] = y.iloc[:split_idx].copy().reset_index(drop=True)
            st.session_state[SS_Y_TEST] = y.iloc[split_idx:].copy().reset_index(drop=True)
            st.session_state[SS_TRAIN_RATIO] = ratio
            done.append("train/test split")

    X_train = st.session_state.get(SS_X_TRAIN)
    if st.session_state.get(SS_CV) is None and X_train is not None:
        from project.src.data.spliter import TimeSeriesSplitWithPurge

        n_splits = int(st.session_state.get(SS_CV_N_SPLITS, 12))
        train_window = int(st.session_state.get(SS_CV_TRAIN_WINDOW, 252))
        test_size = int(st.session_state.get(SS_CV_TEST_SIZE, 21))
        gap = int(st.session_state.get(SS_CV_GAP, 5))
        try:
            cv = TimeSeriesSplitWithPurge(
                n_splits=n_splits,
                train_window_size=train_window,
                test_size=test_size,
                gap_size=gap,
            )
            list(cv.split(X_train))
            st.session_state[SS_CV] = cv
            done.append("CV")
        except Exception:
            pass

    if st.session_state.get(SS_MODEL_RESULTS) is None:
        summaries = list(cfg.get("model_results") or [])
        if summaries:
            from project.src.models.tuner import create_study
            from project.src.models.types import ModelResult

            results: list[Any] = []
            for item in summaries:
                name = str(item.get("name") or "")
                if not name:
                    continue
                results.append(
                    ModelResult(
                        name=name,
                        best_auc=float(item.get("best_auc") or 0.0),
                        best_params=dict(item.get("best_params") or {}),
                        study=create_study(),
                    )
                )
            if results:
                st.session_state[SS_MODEL_RESULTS] = results
                st.session_state["_model_results_restored"] = True
                if not st.session_state.get(SS_BEST_MODEL_NAME):
                    best = max(results, key=lambda r: r.best_auc)
                    st.session_state[SS_BEST_MODEL_NAME] = best.name
                    st.session_state[SS_BEST_PARAMS] = best.best_params
                done.append("model_results")

    if st.session_state.get(SS_OOF_METRICS) is None and cfg.get("oof_metrics"):
        st.session_state[SS_OOF_METRICS] = dict(cfg["oof_metrics"])
        done.append("oof_metrics")

    return done


def ensure_pipeline_rehydrated() -> list[str]:
    """If pipeline objects are missing, rebuild from SS_WORKSPACE_CONFIG.

    Soft mode: never overwrite the user's current Station 2 feature/label draft.
    """
    cfg = st.session_state.get(SS_WORKSPACE_CONFIG) or {}
    if not cfg:
        return []
    needs = (
        st.session_state.get(SS_RAW_DF) is None
        or st.session_state.get(SS_X) is None
        or st.session_state.get(SS_X_TRAIN) is None
        or st.session_state.get(SS_CV) is None
        or st.session_state.get(SS_MODEL_RESULTS) is None
        or (
            st.session_state.get(SS_OOF_METRICS) is None
            and bool(cfg.get("oof_metrics"))
        )
    )
    if not needs:
        return []
    return rehydrate_from_workspace_config(cfg, overwrite_selection=False)


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
            rehydrate_from_workspace_config(cfg, overwrite_selection=True)
            request_feature_widget_sync()
            request_model_widget_sync()
        except FileNotFoundError as exc:
            st.session_state[SS_WELCOME_DONE] = False
            st.warning(str(exc))
    elif welcome_done and data_source == "live":
        st.session_state[SS_DATA_SOURCE] = "live"
        st.session_state[SS_WELCOME_DONE] = True
        rehydrate_from_workspace_config(cfg, overwrite_selection=True)
        request_feature_widget_sync()
        request_model_widget_sync()
