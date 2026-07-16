"""Station 3: Model Design — Train/Test Split, CV, Model Training."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "project").is_dir()),
    None,
)
if _REPO_ROOT is not None and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.models.registry import MODEL_REGISTRY
from project.src.models.tuner import tune_model
from project.src.streamlit_app.plots import fig_cv_splits
from project.src.streamlit_app.session_manager import (
    SS_BEST_MODEL_NAME,
    SS_BEST_PARAMS,
    SS_CV,
    SS_CV_GAP,
    SS_CV_N_SPLITS,
    SS_CV_TEST_SIZE,
    SS_CV_TRAIN_WINDOW,
    SS_FEATURE_SELECTION,
    SS_MODEL_RESULTS,
    SS_N_TRIALS,
    SS_TOP_K,
    SS_TRAIN_RATIO,
    SS_X,
    SS_X_TEST,
    SS_X_TRAIN,
    SS_Y,
    SS_Y_TEST,
    SS_Y_TRAIN,
    clear_downstream,
    has_features_built,
    has_train_test_split,
)


PAGE_NAME = "model_design"


def render() -> None:
    st.header("🧠 Station 3: Model Design")

    if not has_features_built():
        st.warning("Complete Station 2 (Feature Engineering) first to build X and y.")
        return

    X = st.session_state[SS_X]
    y = st.session_state[SS_Y]

    # ── 3a: Time-ordered split ────────────────────────────────────────
    st.subheader("3a. Train / Test Split")
    train_ratio = st.slider(
        "Train ratio", 0.5, 0.9, st.session_state.get(SS_TRAIN_RATIO, 0.8), 0.05
    )
    st.session_state[SS_TRAIN_RATIO] = train_ratio

    split_idx = int(len(X) * train_ratio)

    if st.button("Split Data", type="primary"):
        clear_downstream(PAGE_NAME)
        X_train = X.iloc[:split_idx].copy().reset_index(drop=True)
        X_test = X.iloc[split_idx:].copy().reset_index(drop=True)
        y_train = y.iloc[:split_idx].copy().reset_index(drop=True)
        y_test = y.iloc[split_idx:].copy().reset_index(drop=True)

        st.session_state[SS_X_TRAIN] = X_train
        st.session_state[SS_X_TEST] = X_test
        st.session_state[SS_Y_TRAIN] = y_train
        st.session_state[SS_Y_TEST] = y_test
        st.success("Data split complete.")
        st.rerun()

    X_train = st.session_state.get(SS_X_TRAIN)
    X_test = st.session_state.get(SS_X_TEST)
    y_train = st.session_state.get(SS_Y_TRAIN)
    y_test = st.session_state.get(SS_Y_TEST)

    if X_train is not None:
        st.subheader("Split Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Train samples", len(X_train))
        col2.metric("Test samples", len(X_test))
        if y_train is not None:
            col3.metric("Train class 1", f"{y_train.sum()}/{len(y_train)}")
        if y_test is not None:
            col4.metric("Test class 1", f"{y_test.sum()}/{len(y_test)}")

        with st.expander("Train preview"):
            st.dataframe(X_train.head(10), use_container_width=True)
        with st.expander("Test preview"):
            st.dataframe(X_test.head(10), use_container_width=True)

    # ── 3b: Cross-validation configuration ─────────────────────────────
    st.subheader("3b. Cross-Validation Configuration")
    st.caption("TimeSeriesSplitWithPurge — rolling window with purge gap")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_splits = st.number_input("n_splits", 2, 30, st.session_state.get(SS_CV_N_SPLITS, 12), 1)
    with col2:
        train_window = st.number_input(
            "train_window_size", 50, 1000, st.session_state.get(SS_CV_TRAIN_WINDOW, 252), 10
        )
    with col3:
        test_size = st.number_input("test_size", 5, 252, st.session_state.get(SS_CV_TEST_SIZE, 21), 1)
    with col4:
        gap_size = st.number_input("gap_size", 0, 30, st.session_state.get(SS_CV_GAP, 5), 1)

    if st.button("Create CV Object", type="primary"):
        if X_train is None:
            st.warning("Split data first.")
        else:
            try:
                cv = TimeSeriesSplitWithPurge(
                    n_splits=n_splits,
                    train_window_size=train_window,
                    test_size=test_size,
                    gap_size=gap_size,
                )
                list(cv.split(X_train))
                st.session_state[SS_CV] = cv
                st.session_state[SS_CV_N_SPLITS] = n_splits
                st.session_state[SS_CV_TRAIN_WINDOW] = train_window
                st.session_state[SS_CV_TEST_SIZE] = test_size
                st.session_state[SS_CV_GAP] = gap_size
                st.success("CV object created and validated.")
                st.rerun()
            except Exception as exc:
                st.error(f"CV creation failed: {exc}")

    cv = st.session_state.get(SS_CV)
    if cv is not None and X_train is not None:
        st.subheader("CV Split Visualization")
        fig = fig_cv_splits(cv, X_train)
        st.plotly_chart(fig, use_container_width=True)

    # ── 3c: Model training ────────────────────────────────────────────
    if not has_train_test_split():
        return

    X_train = st.session_state[SS_X_TRAIN]
    y_train = st.session_state[SS_Y_TRAIN]

    if cv is None:
        st.warning("Create a CV object above before training.")
        return

    st.subheader("3c. Model Configuration & Training")
    available_models = list(MODEL_REGISTRY.keys())
    selected_models = st.multiselect(
        "Select models to train",
        options=available_models,
        default=available_models,
    )

    n_trials = st.number_input(
        "Optuna trials", 3, 100, st.session_state.get(SS_N_TRIALS, 30), 1
    )

    if st.button("Train Models", type="primary"):
        if not selected_models:
            st.warning("Select at least one model.")
            return

        clear_downstream(PAGE_NAME)

        status_container = st.status("Training models...", expanded=True)
        progress_bar = st.progress(0)

        try:
            results = []
            for i, model_name in enumerate(selected_models):
                status_container.write(f"Tuning **{model_name}** ({n_trials} trials)...")
                result = tune_model(
                    model_name,
                    X_train,
                    y_train,
                    cv,
                    n_trials=n_trials,
                    nested_feature_selection=False,
                )
                results.append(result)
                progress_bar.progress((i + 1) / len(selected_models))

            best = max(results, key=lambda r: r.best_auc)

            st.session_state[SS_MODEL_RESULTS] = results
            st.session_state[SS_BEST_MODEL_NAME] = best.name
            st.session_state[SS_BEST_PARAMS] = best.best_params
            st.session_state[SS_N_TRIALS] = n_trials
            st.session_state[SS_FEATURE_SELECTION] = None
            st.session_state[SS_TOP_K] = 5

            status_container.update(label="Training complete!", state="complete")
            st.rerun()
        except Exception as exc:
            status_container.update(label=f"Training failed: {exc}", state="error")
            return

    # ── Results ────────────────────────────────────────────────────────
    model_results = st.session_state.get(SS_MODEL_RESULTS)
    if model_results is not None:
        st.subheader("Training Results")
        rows = []
        for result in model_results:
            rows.append({
                "Model": result.name,
                "Best AUC": f"{result.best_auc:.4f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        best_name = st.session_state.get(SS_BEST_MODEL_NAME, "")
        st.success(f"✅ Best model: **{best_name}** (AUC = {max(r.best_auc for r in model_results):.4f})")
