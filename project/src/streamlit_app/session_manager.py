"""Session state keys, defaults, and helpers for the Streamlit workbench."""
from __future__ import annotations

from typing import Any

import streamlit as st

# Session-state key constants
SS_RAW_DF = "raw_df"
SS_ENRICHED_DF = "enriched_df"
SS_SELECTED_FEATURES = "selected_features"
SS_X = "X"
SS_Y = "y"
SS_META_DF = "meta_df"
SS_TARGET_COL = "target_col"
SS_HORIZON = "horizon"
SS_LABEL_METHOD = "label_method"

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

# Page order (used for downstream clearing)
PAGE_ORDER = ["data_feature", "train_test", "model_train", "evaluation"]

# All keys grouped by the page that *sets* them
_PAGE_KEYS: dict[str, list[str]] = {
    "data_feature": [
        SS_RAW_DF,
        SS_ENRICHED_DF,
        SS_SELECTED_FEATURES,
        SS_X,
        SS_Y,
        SS_META_DF,
        SS_TARGET_COL,
        SS_HORIZON,
        SS_LABEL_METHOD,
    ],
    "train_test": [
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
    ],
    "model_train": [
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
}


def init_session_state() -> None:
    """Set default values for all session state keys on first load."""
    defaults: dict[str, Any] = {
        SS_RAW_DF: None,
        SS_ENRICHED_DF: None,
        SS_SELECTED_FEATURES: [],
        SS_X: None,
        SS_Y: None,
        SS_META_DF: None,
        SS_TARGET_COL: "",
        SS_HORIZON: 5,
        SS_LABEL_METHOD: "expanding",
        SS_TRAIN_RATIO: 0.8,
        SS_X_TRAIN: None,
        SS_X_TEST: None,
        SS_Y_TRAIN: None,
        SS_Y_TEST: None,
        SS_CV: None,
        SS_CV_N_SPLITS: 5,
        SS_CV_TRAIN_WINDOW: 252,
        SS_CV_TEST_SIZE: 21,
        SS_CV_GAP: 5,
        SS_MODEL_RESULTS: None,
        SS_BEST_MODEL_NAME: "",
        SS_BEST_PARAMS: {},
        SS_N_TRIALS: 10,
        SS_FEATURE_SELECTION: None,
        SS_TOP_K: 5,
        SS_OOF_METRICS: None,
        SS_OOF_PROBA_FULL: None,
        SS_OOF_PROBA_VALID: None,
        SS_OOF_TRUE_VALID: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
