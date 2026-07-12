"""Page 3: Model Training & Auto-Hyperparameter Tuning."""
from __future__ import annotations

import streamlit as st

from project.src.models.registry import MODEL_REGISTRY
from project.src.models.tuner import tune_model, tune_models
from project.src.streamlit_app.session_manager import (
    SS_BEST_MODEL_NAME,
    SS_BEST_PARAMS,
    SS_CV,
    SS_FEATURE_SELECTION,
    SS_MODEL_RESULTS,
    SS_N_TRIALS,
    SS_TOP_K,
    SS_X_TRAIN,
    SS_Y_TRAIN,
    clear_downstream,
    has_train_test_split,
)


def render() -> None:
    st.header("Model Training & Auto-Tuning")

    if not has_train_test_split():
        st.warning("Complete Page 2 (Train/Test Split) first.")
        return

    X_train = st.session_state[SS_X_TRAIN]
    y_train = st.session_state[SS_Y_TRAIN]
    cv = st.session_state.get(SS_CV)

    if cv is None:
        st.warning("Create a CV object on Page 2 before training.")
        return

    # ── Model selection ────────────────────────────────────────────────
    st.subheader("Model Configuration")
    available_models = list(MODEL_REGISTRY.keys())
    selected_models = st.multiselect(
        "Select models to train",
        options=available_models,
        default=available_models,
    )

    # ── Optuna config ──────────────────────────────────────────────────
    n_trials = st.number_input(
        "Optuna trials", 3, 100, st.session_state.get(SS_N_TRIALS, 10), 1
    )
    fs_method = None
    top_k = 5

    # ── Training ───────────────────────────────────────────────────────
    if st.button("Train Models", type="primary"):
        if not selected_models:
            st.warning("Select at least one model.")
            return

        clear_downstream("model_train")

        fs_arg = fs_method  # nested feature selection disabled

        status_container = st.status("Training models...", expanded=True)
        progress_bar = st.progress(0)
        results_text = st.empty()

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
                    feature_selection=fs_arg,
                    top_k_features=top_k,
                    nested_feature_selection=False,
                )
                results.append(result)
                progress_bar.progress((i + 1) / len(selected_models))

            best = max(results, key=lambda r: r.best_auc)

            st.session_state[SS_MODEL_RESULTS] = results
            st.session_state[SS_BEST_MODEL_NAME] = best.name
            st.session_state[SS_BEST_PARAMS] = best.best_params
            st.session_state[SS_N_TRIALS] = n_trials
            st.session_state[SS_FEATURE_SELECTION] = fs_arg
            st.session_state[SS_TOP_K] = top_k

            status_container.update(label="Training complete!", state="complete")
        except Exception as exc:
            status_container.update(label=f"Training failed: {exc}", state="error")
            return

    # ── Results ────────────────────────────────────────────────────────
    model_results = st.session_state.get(SS_MODEL_RESULTS)
    if model_results is not None:
        st.subheader("Training Results")

        # Summary table
        rows = []
        for result in model_results:
            rows.append({
                "Model": result.name,
                "Best AUC": f"{result.best_auc:.4f}",
                "Best Params": str(result.best_params),
            })
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        best_name = st.session_state.get(SS_BEST_MODEL_NAME, "")
        best_params = st.session_state.get(SS_BEST_PARAMS, {})
        st.info(f"Best model: **{best_name}** with params: {best_params}")
