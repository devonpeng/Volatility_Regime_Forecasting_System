"""Session state keys, defaults, and helpers for the Streamlit workbench."""
from __future__ import annotations

from typing import Any

import streamlit as st

# Session-state key constants
SS_RAW_DF = "raw_df"
SS_FULL_DF = "full_df"
SS_ENRICHED_DF = "enriched_df"
SS_SELECTED_FEATURES = "selected_features"
SS_X = "X"
SS_Y = "y"
SS_META_DF = "meta_df"
SS_TARGET_COL = "target_col"
SS_HORIZON = "horizon"
SS_LABEL_METHOD = "label_method"
SS_SYMBOL = "symbol"
SS_DATE_START = "date_start"
SS_DATE_END = "date_end"
SS_DATA_SOURCE = "data_source"  # "parquet" | "live"

SS_TRAIN_RATIO = "train_ratio"
SS_X_TRAIN = "X_train"
SS_X_TEST = "X_test"
SS_Y_TRAIN = "y_train"
SS_Y_TEST = "y_test"
SS_CV = "cv"
SS_CV_N_SPLITS = "cv_n_splits"
SS_CV_TRAIN_WINDOW = "cv_train_window"
SS_CV_TEST_SIZE = "cv_test_size"
SS_CV_GAP = "cv_gap_size"
SS_SELECTED_MODELS = "selected_models"

SS_MODEL_RESULTS = "model_results"
SS_BEST_MODEL_NAME = "best_model_name"
SS_BEST_PARAMS = "best_params"
SS_N_TRIALS = "n_trials"
SS_FEATURE_SELECTION = "feature_selection"
SS_TOP_K = "top_k"

SS_OOF_METRICS = "oof_metrics"
SS_OOF_PROBA_FULL = "oof_proba_full"
SS_OOF_PROBA_VALID = "oof_proba_valid"
SS_OOF_TRUE_VALID = "oof_true_valid"

SS_USER_ID = "user_id"
SS_USERNAME = "username"
SS_WORKSPACE_ID = "workspace_id"
SS_WORKSPACE_NAME = "workspace_name"
SS_WS_VERSION = "ws_version"
SS_WORKSPACE_CONFIG = "workspace_config"
SS_WELCOME_DONE = "welcome_done"
SS_CUSTOM_EXPERIMENTS = "custom_experiments"
SS_COMPARED_EXPERIMENTS = "compared_experiments"
SS_CHAT_HISTORY = "chat_history"
SS_LLM_API_KEY = "llm_api_key_override"
SS_LLM_MODEL = "llm_model_override"
SS_AUTHENTICATED = "authenticated"

# AI agent / HITL
SS_PENDING_ACTIONS = "pending_actions"
SS_AI_THREAD_ID = "ai_thread_id"
SS_AI_INTERRUPT = "ai_interrupt"  # clarify payloads
SS_AI_CLARIFY = "ai_clarify"  # {question, options?}

# Incremental training controls
SS_TRAIN_STATUS = "train_status"  # idle | running
SS_TRAIN_STOP = "train_stop"
SS_TRAIN_SKIP = "train_skip"
SS_TRAIN_QUEUE = "train_queue"
SS_TRAIN_RESULTS = "train_partial_results"
SS_TRAIN_STUDY = "train_study"
SS_TRAIN_CURRENT = "train_current_model"
SS_TRAIN_TRIAL_I = "train_trial_i"
SS_TRAIN_N_TRIALS = "train_n_trials"

# Global settings draft widget keys (left sidebar)
GS_SYMBOL = "gs_symbol"
GS_HORIZON = "gs_horizon"
GS_START = "gs_start"
GS_END = "gs_end"
SS_PENDING_GS_SYNC = "_pending_gs_sync"
SS_PENDING_FEATURE_SYNC = "_pending_feature_sync"
SS_PENDING_MODEL_SYNC = "_pending_model_sync"
SS_PENDING_COMPARE_SYNC = "_pending_compare_sync"

# Right AI rail collapse
SS_AI_RAIL_COLLAPSED = "ai_rail_collapsed"

# Page order (used for downstream clearing)
PAGE_ORDER = ["etl", "feature_eng", "model_design", "implementation", "comparison"]

# All keys grouped by the page that *sets* them
_PAGE_KEYS: dict[str, list[str]] = {
    "etl": [
        SS_RAW_DF,
        SS_FULL_DF,
        SS_DATA_SOURCE,
    ],
    "feature_eng": [
        SS_ENRICHED_DF,
        SS_SELECTED_FEATURES,
        SS_X,
        SS_Y,
        SS_META_DF,
        SS_TARGET_COL,
        SS_LABEL_METHOD,
    ],
    "model_design": [
        SS_TRAIN_RATIO,
        SS_X_TRAIN,
        SS_X_TEST,
        SS_Y_TRAIN,
        SS_Y_TEST,
        SS_CV,
        SS_CV_N_SPLITS,
        SS_CV_TRAIN_WINDOW,
        SS_CV_TEST_SIZE,
        SS_CV_GAP,
        SS_SELECTED_MODELS,
    ],
    "implementation": [
        SS_MODEL_RESULTS,
        SS_BEST_MODEL_NAME,
        SS_BEST_PARAMS,
        SS_N_TRIALS,
        SS_FEATURE_SELECTION,
        SS_TOP_K,
    ],
    "evaluation": [
        SS_OOF_METRICS,
        SS_OOF_PROBA_FULL,
        SS_OOF_PROBA_VALID,
        SS_OOF_TRUE_VALID,
    ],
    "comparison": [
        SS_CUSTOM_EXPERIMENTS,
        SS_COMPARED_EXPERIMENTS,
    ],
}


def init_session_state() -> None:
    """Set default values for all session state keys on first load."""
    defaults: dict[str, Any] = {
        SS_RAW_DF: None,
        SS_FULL_DF: None,
        SS_ENRICHED_DF: None,
        SS_SELECTED_FEATURES: [],
        SS_X: None,
        SS_Y: None,
        SS_META_DF: None,
        SS_TARGET_COL: "",
        SS_HORIZON: 5,
        SS_LABEL_METHOD: "expanding",
        SS_SYMBOL: "SPY",
        SS_DATE_START: "2010-01-01",
        SS_DATE_END: "2024-12-31",
        SS_DATA_SOURCE: "parquet",
        SS_TRAIN_RATIO: 0.8,
        SS_X_TRAIN: None,
        SS_X_TEST: None,
        SS_Y_TRAIN: None,
        SS_Y_TEST: None,
        SS_CV: None,
        SS_CV_N_SPLITS: 12,
        SS_CV_TRAIN_WINDOW: 252,
        SS_CV_TEST_SIZE: 21,
        SS_CV_GAP: 5,
        SS_SELECTED_MODELS: ["lgbm", "xgb", "logistic"],
        SS_MODEL_RESULTS: None,
        SS_BEST_MODEL_NAME: "",
        SS_BEST_PARAMS: {},
        SS_N_TRIALS: 30,
        SS_FEATURE_SELECTION: None,
        SS_TOP_K: 5,
        SS_OOF_METRICS: None,
        SS_OOF_PROBA_FULL: None,
        SS_OOF_PROBA_VALID: None,
        SS_OOF_TRUE_VALID: None,
        SS_USER_ID: None,
        SS_USERNAME: None,
        SS_WORKSPACE_ID: None,
        SS_WORKSPACE_NAME: None,
        SS_WS_VERSION: 0,
        SS_WORKSPACE_CONFIG: {},
        SS_WELCOME_DONE: False,
        SS_CUSTOM_EXPERIMENTS: [],
        SS_COMPARED_EXPERIMENTS: [],
        SS_CHAT_HISTORY: [],
        SS_LLM_API_KEY: "",
        SS_LLM_MODEL: "",
        SS_AUTHENTICATED: False,
        SS_PENDING_ACTIONS: [],
        SS_AI_THREAD_ID: "",
        SS_AI_INTERRUPT: None,
        SS_AI_CLARIFY: None,
        SS_TRAIN_STATUS: "idle",
        SS_TRAIN_STOP: False,
        SS_TRAIN_SKIP: False,
        SS_TRAIN_QUEUE: [],
        SS_TRAIN_RESULTS: [],
        SS_TRAIN_STUDY: None,
        SS_TRAIN_CURRENT: "",
        SS_TRAIN_TRIAL_I: 0,
        SS_TRAIN_N_TRIALS: 30,
        SS_AI_RAIL_COLLAPSED: False,
        SS_PENDING_GS_SYNC: False,
        SS_PENDING_FEATURE_SYNC: False,
        SS_PENDING_MODEL_SYNC: False,
        SS_PENDING_COMPARE_SYNC: False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_pipeline_keep_globals() -> None:
    """Clear station pipeline state; keep login, workspace, GS_*, applied symbol/dates, AI chat."""
    # Do not touch GS_* (widget keys may already be instantiated this run).
    preserve = {
        SS_SYMBOL,
        SS_HORIZON,
        SS_DATE_START,
        SS_DATE_END,
        SS_USER_ID,
        SS_USERNAME,
        SS_AUTHENTICATED,
        SS_WORKSPACE_ID,
        SS_WORKSPACE_NAME,
        SS_WS_VERSION,
        SS_WORKSPACE_CONFIG,
        SS_CHAT_HISTORY,
        SS_PENDING_ACTIONS,
        SS_AI_THREAD_ID,
        SS_AI_INTERRUPT,
        SS_AI_CLARIFY,
        SS_LLM_API_KEY,
        SS_LLM_MODEL,
        SS_AI_RAIL_COLLAPSED,
    }
    preserved_vals = {
        k: st.session_state[k] for k in preserve if k in st.session_state
    }

    for page_keys in _PAGE_KEYS.values():
        for key in page_keys:
            if key in st.session_state and key not in preserve:
                del st.session_state[key]

    for key in (
        "_full_data",
        "_feature_importances",
        "_shap_bundle",
        "_shap_ai",
        "_llm_report",
        "target_col",
        "_fs_version",
        SS_WELCOME_DONE,
        SS_TRAIN_STATUS,
        SS_TRAIN_STOP,
        SS_TRAIN_SKIP,
        SS_TRAIN_QUEUE,
        SS_TRAIN_RESULTS,
        SS_TRAIN_STUDY,
        SS_TRAIN_CURRENT,
        SS_TRAIN_TRIAL_I,
        SS_TRAIN_N_TRIALS,
        SS_BEST_MODEL_NAME,
        SS_BEST_PARAMS,
    ):
        if key in st.session_state and key not in preserve:
            del st.session_state[key]

    init_session_state()
    for key, value in preserved_vals.items():
        st.session_state[key] = value
    st.session_state[SS_WELCOME_DONE] = False


def clear_downstream(from_page: str) -> None:
    """Clear session state for all pages downstream of *from_page*."""
    try:
        idx = PAGE_ORDER.index(from_page)
    except ValueError:
        return
    for page in PAGE_ORDER[idx + 1 :]:
        for key in _PAGE_KEYS.get(page, []):
            if key in st.session_state:
                del st.session_state[key]
    # Also clear evaluation keys when clearing from model_design or earlier
    if from_page in ("etl", "feature_eng", "model_design"):
        for key in _PAGE_KEYS.get("evaluation", []):
            if key in st.session_state:
                del st.session_state[key]
    # Re-init cleared keys to defaults
    init_session_state()


def clear_from_feature_eng() -> None:
    """Clear feature_eng and everything downstream (keep applied globals + etl frames)."""
    clear_downstream("etl")


def has_data_loaded() -> bool:
    return st.session_state.get(SS_RAW_DF) is not None


def has_features_built() -> bool:
    return (
        st.session_state.get(SS_X) is not None
        and st.session_state.get(SS_Y) is not None
    )


def has_train_test_split() -> bool:
    return (
        st.session_state.get(SS_X_TRAIN) is not None
        and st.session_state.get(SS_X_TEST) is not None
    )


def has_model_results() -> bool:
    return st.session_state.get(SS_MODEL_RESULTS) is not None


def is_parquet_workspace() -> bool:
    return st.session_state.get(SS_DATA_SOURCE, "parquet") == "parquet"


def build_workspace_config_snapshot() -> dict[str, Any]:
    """Serialize session config (no DataFrames) for SQLite / export JSON."""
    model_results = st.session_state.get(SS_MODEL_RESULTS)
    model_summaries = []
    if model_results:
        for r in model_results:
            model_summaries.append(
                {
                    "name": getattr(r, "name", None),
                    "best_auc": float(getattr(r, "best_auc", 0.0)),
                    "best_params": getattr(r, "best_params", {}),
                }
            )
    oof = st.session_state.get(SS_OOF_METRICS) or {}
    llm_report = st.session_state.get("_llm_report")
    return {
        "symbol": st.session_state.get(SS_SYMBOL, "SPY"),
        "horizon": int(st.session_state.get(SS_HORIZON, 5)),
        "date_range": [
            str(st.session_state.get(SS_DATE_START, "")),
            str(st.session_state.get(SS_DATE_END, "")),
        ],
        "selected_features": list(st.session_state.get(SS_SELECTED_FEATURES) or []),
        "label_method": st.session_state.get(SS_LABEL_METHOD, "expanding"),
        "data_source": st.session_state.get(SS_DATA_SOURCE, "parquet"),
        "welcome_done": bool(st.session_state.get(SS_WELCOME_DONE, False)),
        "cv_config": {
            "n_splits": st.session_state.get(SS_CV_N_SPLITS, 12),
            "train_window": st.session_state.get(SS_CV_TRAIN_WINDOW, 252),
            "test_size": st.session_state.get(SS_CV_TEST_SIZE, 21),
            "gap": st.session_state.get(SS_CV_GAP, 5),
        },
        "selected_models": list(st.session_state.get(SS_SELECTED_MODELS) or []),
        "best_model_name": st.session_state.get(SS_BEST_MODEL_NAME, ""),
        "best_params": st.session_state.get(SS_BEST_PARAMS) or {},
        "model_results": model_summaries,
        "oof_metrics": oof,
        "feature_selection": st.session_state.get(SS_FEATURE_SELECTION),
        "top_k": st.session_state.get(SS_TOP_K, 5),
        "n_trials": st.session_state.get(SS_N_TRIALS, 30),
        "train_ratio": st.session_state.get(SS_TRAIN_RATIO, 0.8),
        "experiments": list(st.session_state.get(SS_CUSTOM_EXPERIMENTS) or []),
        "compared_experiments": list(
            st.session_state.get(SS_COMPARED_EXPERIMENTS) or []
        ),
        "chat_history": list(st.session_state.get(SS_CHAT_HISTORY) or []),
        "_llm_report": llm_report,
        "model_artifacts": list(st.session_state.get("_model_artifact_meta") or []),
    }
