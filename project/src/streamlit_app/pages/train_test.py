"""Page 2: Train/Test Split & CV Configuration."""
from __future__ import annotations

import streamlit as st

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.streamlit_app.plots import fig_cv_splits
from project.src.streamlit_app.session_manager import (
    SS_CV,
    SS_CV_GAP,
    SS_CV_N_SPLITS,
    SS_CV_TEST_SIZE,
    SS_CV_TRAIN_WINDOW,
    SS_TRAIN_RATIO,
    SS_X,
    SS_X_TEST,
    SS_X_TRAIN,
    SS_Y,
    SS_Y_TEST,
    SS_Y_TRAIN,
    clear_downstream,
    has_features_built,
)


def render() -> None:
    st.header("Train / Test Split")

    if not has_features_built():
        st.warning("Complete Page 1 (Data & Feature Engineering) first to build X and y.")
        return

    X = st.session_state[SS_X]
    y = st.session_state[SS_Y]

    # ── Time-ordered split ────────────────────────────────────────────
    st.subheader("Time-Ordered Split")
    train_ratio = st.slider(
        "Train ratio", 0.5, 0.9, st.session_state.get(SS_TRAIN_RATIO, 0.8), 0.05
    )
    st.session_state[SS_TRAIN_RATIO] = train_ratio

    split_idx = int(len(X) * train_ratio)

    if st.button("Split Data", type="primary"):
        clear_downstream("train_test")
        X_train = X.iloc[:split_idx].copy().reset_index(drop=True)
        X_test = X.iloc[split_idx:].copy().reset_index(drop=True)
        y_train = y.iloc[:split_idx].copy().reset_index(drop=True)
        y_test = y.iloc[split_idx:].copy().reset_index(drop=True)

        st.session_state[SS_X_TRAIN] = X_train
        st.session_state[SS_X_TEST] = X_test
        st.session_state[SS_Y_TRAIN] = y_train
        st.session_state[SS_Y_TEST] = y_test
        st.success("Data split complete.")

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

    # ── Cross-validation configuration ─────────────────────────────────
    st.subheader("Cross-Validation Configuration")
    st.caption("TimeSeriesSplitWithPurge — rolling window with purge gap")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_splits = st.number_input("n_splits", 2, 20, st.session_state.get(SS_CV_N_SPLITS, 5), 1)
    with col2:
        train_window = st.number_input(
            "train_window_size", 50, 1000, st.session_state.get(SS_CV_TRAIN_WINDOW, 252), 10
        )
    with col3:
        test_size = st.number_input("test_size", 5, 252, st.session_state.get(SS_CV_TEST_SIZE, 21), 1)
    with col4:
        gap_size = st.number_input("gap_size", 0, 30, st.session_state.get(SS_CV_GAP, 5), 1)

    if st.button("Create CV Object"):
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
                # Validate the CV object
                list(cv.split(X_train))
                st.session_state[SS_CV] = cv
                st.session_state[SS_CV_N_SPLITS] = n_splits
                st.session_state[SS_CV_TRAIN_WINDOW] = train_window
                st.session_state[SS_CV_TEST_SIZE] = test_size
                st.session_state[SS_CV_GAP] = gap_size
                st.success("CV object created and validated.")
            except Exception as exc:
                st.error(f"CV creation failed: {exc}")

    cv = st.session_state.get(SS_CV)
    if cv is not None and X_train is not None:
        st.subheader("CV Split Visualization")
        fig = fig_cv_splits(cv, X_train)
        st.pyplot(fig)
