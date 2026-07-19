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
from project.src.models.tuner import create_study, run_one_trial, study_to_result
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
    SS_SELECTED_MODELS,
    SS_TOP_K,
    SS_TRAIN_CURRENT,
    SS_TRAIN_N_TRIALS,
    SS_TRAIN_QUEUE,
    SS_TRAIN_RESULTS,
    SS_TRAIN_SKIP,
    SS_TRAIN_STATUS,
    SS_TRAIN_STOP,
    SS_TRAIN_STUDY,
    SS_TRAIN_TRIAL_I,
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
from project.src.streamlit_app.workspace import checkpoint_current_workspace


PAGE_NAME = "model_design"


def _reset_training_state() -> None:
    st.session_state[SS_TRAIN_STATUS] = "idle"
    st.session_state[SS_TRAIN_STOP] = False
    st.session_state[SS_TRAIN_SKIP] = False
    st.session_state[SS_TRAIN_QUEUE] = []
    st.session_state[SS_TRAIN_RESULTS] = []
    st.session_state[SS_TRAIN_STUDY] = None
    st.session_state[SS_TRAIN_CURRENT] = ""
    st.session_state[SS_TRAIN_TRIAL_I] = 0


def _finalize_training(results: list) -> None:
    if results:
        best = max(results, key=lambda r: r.best_auc)
        st.session_state[SS_MODEL_RESULTS] = results
        st.session_state[SS_BEST_MODEL_NAME] = best.name
        st.session_state[SS_BEST_PARAMS] = best.best_params
        st.session_state[SS_FEATURE_SELECTION] = None
        st.session_state[SS_TOP_K] = 5
        checkpoint_current_workspace("model training")
    _reset_training_state()


def _advance_or_finish(results: list, queue: list[str]) -> None:
    if not queue:
        _finalize_training(results)
        return
    next_model = queue[0]
    st.session_state[SS_TRAIN_QUEUE] = queue[1:]
    st.session_state[SS_TRAIN_CURRENT] = next_model
    st.session_state[SS_TRAIN_STUDY] = create_study()
    st.session_state[SS_TRAIN_TRIAL_I] = 0
    st.session_state[SS_TRAIN_SKIP] = False


def _tick_training() -> None:
    """Run one Optuna trial or handle stop/skip between trials."""
    if st.session_state.get(SS_TRAIN_STATUS) != "running":
        return

    X_train = st.session_state[SS_X_TRAIN]
    y_train = st.session_state[SS_Y_TRAIN]
    cv = st.session_state[SS_CV]
    n_trials = int(st.session_state.get(SS_TRAIN_N_TRIALS, 30))
    results = list(st.session_state.get(SS_TRAIN_RESULTS) or [])
    queue = list(st.session_state.get(SS_TRAIN_QUEUE) or [])
    current = st.session_state.get(SS_TRAIN_CURRENT) or ""
    study = st.session_state.get(SS_TRAIN_STUDY)
    trial_i = int(st.session_state.get(SS_TRAIN_TRIAL_I, 0))

    if st.session_state.get(SS_TRAIN_STOP):
        if current and study is not None and trial_i > 0:
            results.append(study_to_result(current, study))
        _finalize_training(results)
        st.warning("Training stopped. Keeping completed models only.")
        st.rerun()
        return

    if st.session_state.get(SS_TRAIN_SKIP):
        # discard current incomplete study; do not append
        _advance_or_finish(results, queue)
        if st.session_state.get(SS_TRAIN_STATUS) == "idle":
            st.info("Skipped remaining work / finished after skip.")
        st.rerun()
        return

    if not current or study is None:
        if queue:
            _advance_or_finish(results, queue)
            st.rerun()
        else:
            _finalize_training(results)
            st.rerun()
        return

    # Run one trial
    try:
        study = run_one_trial(
            current,
            X_train,
            y_train,
            cv,
            study,
            nested_feature_selection=False,
            should_stop=lambda: bool(st.session_state.get(SS_TRAIN_STOP)),
        )
        trial_i += 1
        st.session_state[SS_TRAIN_STUDY] = study
        st.session_state[SS_TRAIN_TRIAL_I] = trial_i
    except Exception as exc:
        st.error(f"Training failed on {current}: {exc}")
        _finalize_training(results)
        st.rerun()
        return

    if trial_i >= n_trials or st.session_state.get(SS_TRAIN_STOP):
        if st.session_state.get(SS_TRAIN_STOP):
            if trial_i > 0:
                results.append(study_to_result(current, study))
            st.session_state[SS_TRAIN_RESULTS] = results
            _finalize_training(results)
            st.warning("Training stopped.")
            st.rerun()
            return
        results.append(study_to_result(current, study))
        st.session_state[SS_TRAIN_RESULTS] = results
        _advance_or_finish(results, queue)
        st.rerun()
        return

    st.rerun()


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
        "Train ratio",
        0.5,
        0.9,
        float(st.session_state.get(SS_TRAIN_RATIO, 0.8)),
        0.05,
        key="md_train_ratio",
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
        n_splits = st.number_input(
            "n_splits",
            2,
            30,
            int(st.session_state.get(SS_CV_N_SPLITS, 12)),
            1,
            key="md_cv_n_splits",
        )
    with col2:
        train_window = st.number_input(
            "train_window_size",
            50,
            1000,
            int(st.session_state.get(SS_CV_TRAIN_WINDOW, 252)),
            10,
            key="md_cv_train_window",
        )
    with col3:
        test_size = st.number_input(
            "test_size",
            5,
            252,
            int(st.session_state.get(SS_CV_TEST_SIZE, 21)),
            1,
            key="md_cv_test_size",
        )
    with col4:
        gap_size = st.number_input(
            "gap_size",
            0,
            30,
            int(st.session_state.get(SS_CV_GAP, 5)),
            1,
            key="md_cv_gap",
        )

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
    default_models = list(st.session_state.get(SS_SELECTED_MODELS) or available_models)
    default_models = [m for m in default_models if m in available_models] or available_models

    if "md_selected_models" not in st.session_state:
        st.session_state["md_selected_models"] = default_models
    selected_models = st.multiselect(
        "Select models to train",
        options=available_models,
        key="md_selected_models",
    )
    st.session_state[SS_SELECTED_MODELS] = selected_models

    if "md_n_trials" not in st.session_state:
        st.session_state["md_n_trials"] = int(st.session_state.get(SS_N_TRIALS, 30))
    n_trials = st.number_input(
        "Optuna trials",
        3,
        100,
        key="md_n_trials",
    )
    st.session_state[SS_N_TRIALS] = int(n_trials)

    training = st.session_state.get(SS_TRAIN_STATUS) == "running"

    b1, b2, b3 = st.columns(3)
    with b1:
        start = st.button("Train Models", type="primary", disabled=training)
    with b2:
        if st.button("Stop all", disabled=not training, key="train_stop_btn"):
            st.session_state[SS_TRAIN_STOP] = True
    with b3:
        if st.button("Skip current model", disabled=not training, key="train_skip_btn"):
            st.session_state[SS_TRAIN_SKIP] = True

    if start:
        if not selected_models:
            st.warning("Select at least one model.")
        else:
            clear_downstream(PAGE_NAME)
            # re-set CV etc after clear? clear_downstream clears model_design keys including CV!
            # BUG: clear_downstream(PAGE_NAME) clears from implementation onward only for model_design
            # Looking at clear_downstream - it clears pages AFTER from_page, so clear_downstream("model_design")
            # clears implementation + comparison, NOT model_design's own keys. Good.
            # Wait - PAGE_ORDER and clear_downstream: for page in PAGE_ORDER[idx+1:], so from model_design
            # clears implementation and comparison. But evaluation is not in PAGE_ORDER!
            # We also need to clear previous model_results - those are under "implementation" keys. Good.

            st.session_state[SS_TRAIN_STATUS] = "running"
            st.session_state[SS_TRAIN_STOP] = False
            st.session_state[SS_TRAIN_SKIP] = False
            st.session_state[SS_TRAIN_QUEUE] = list(selected_models[1:])
            st.session_state[SS_TRAIN_CURRENT] = selected_models[0]
            st.session_state[SS_TRAIN_RESULTS] = []
            st.session_state[SS_TRAIN_STUDY] = create_study()
            st.session_state[SS_TRAIN_TRIAL_I] = 0
            st.session_state[SS_TRAIN_N_TRIALS] = int(n_trials)
            st.session_state[SS_N_TRIALS] = int(n_trials)
            st.rerun()

    if training:
        current = st.session_state.get(SS_TRAIN_CURRENT, "")
        trial_i = int(st.session_state.get(SS_TRAIN_TRIAL_I, 0))
        n_tot = int(st.session_state.get(SS_TRAIN_N_TRIALS, 30))
        queue = list(st.session_state.get(SS_TRAIN_QUEUE) or [])
        done_models = list(st.session_state.get(SS_TRAIN_RESULTS) or [])
        total_models = len(done_models) + 1 + len(queue)
        st.info(
            f"Tuning **{current}** — trial {trial_i}/{n_tot} · "
            f"models done {len(done_models)}/{total_models}"
        )
        progress = 0.0
        if total_models > 0 and n_tot > 0:
            progress = (len(done_models) + trial_i / n_tot) / total_models
        st.progress(min(1.0, progress))
        _tick_training()

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
        st.success(
            f"✅ Best model: **{best_name}** "
            f"(AUC = {max(r.best_auc for r in model_results):.4f})"
        )
