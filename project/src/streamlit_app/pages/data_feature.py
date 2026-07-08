"""Page 1: Data Loading & Feature Engineering."""
from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from project.src.data.load_data import get_data_with_fallback
from project.src.features.core import (
    DEFAULT_FEATURE_STEPS,
    FEATURE_REGISTRY,
    FeatureStep,
    add_target_and_label,
    apply_features,
)
from project.src.streamlit_app.llm_helper import generate_feature, safe_exec_feature
from project.src.streamlit_app.session_manager import (
    SS_ENRICHED_DF,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_META_DF,
    SS_RAW_DF,
    SS_SELECTED_FEATURES,
    SS_TARGET_COL,
    SS_X,
    SS_Y,
    clear_downstream,
)


@st.cache_data(show_spinner="Downloading market data...")
def _load_data_cached(symbol: str, start_date: str, end_date: str, use_synthetic: bool) -> pd.DataFrame:
    return get_data_with_fallback(symbol, start_date, end_date, use_synthetic=use_synthetic)


def render() -> None:
    st.header("Data & Feature Engineering")

    # ── Data loading controls ──────────────────────────────────────────
    with st.expander("Data Source", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input("Symbol", value="SPY")
        with col2:
            start_date = st.date_input("Start date", value=pd.Timestamp("2020-01-01"))
        with col3:
            end_date = st.date_input("End date", value=pd.Timestamp("2024-12-31"))
        use_synthetic = st.checkbox("Use synthetic data (no network)", value=False)

        if st.button("Load Data", type="primary"):
            clear_downstream("data_feature")
            with st.spinner("Loading data..."):
                raw_df = _load_data_cached(
                    symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), use_synthetic
                )
            st.session_state[SS_RAW_DF] = raw_df
            st.success(f"Loaded {len(raw_df)} rows (source: {raw_df.attrs.get('source', 'unknown')})")

    raw_df = st.session_state.get(SS_RAW_DF)
    if raw_df is None:
        st.info("Load data to get started.")
        return

    st.subheader("Raw Data Preview")
    col_top, col_tail = st.columns(2)
    with col_top:
        st.caption("Head")
        st.dataframe(raw_df.head(10), use_container_width=True)
    with col_tail:
        st.caption("Tail")
        st.dataframe(raw_df.tail(10), use_container_width=True)

    # ── Feature selection ──────────────────────────────────────────────
    st.subheader("Feature Selection")
    all_names = [step.name for step in DEFAULT_FEATURE_STEPS]
    selected = st.multiselect(
        "Select features",
        options=all_names,
        default=st.session_state.get(SS_SELECTED_FEATURES) or all_names,
    )
    st.session_state[SS_SELECTED_FEATURES] = selected

    # ── Transparent computation view ───────────────────────────────────
    with st.expander("Feature Computation Details"):
        for step in DEFAULT_FEATURE_STEPS:
            if step.name in selected:
                with st.expander(f"{step.name}"):
                    st.code(inspect.getsource(step.fn), language="python")
                    if raw_df is not None:
                        try:
                            preview_df = step.fn(raw_df.copy())
                            if step.name in preview_df.columns:
                                st.write(preview_df[[step.name]].head(10))
                        except Exception:
                            st.caption("Preview unavailable for this feature.")

    # ── Build features ─────────────────────────────────────────────────
    if st.button("Build Features", type="primary"):
        if not selected:
            st.warning("Select at least one feature.")
        else:
            clear_downstream("data_feature")
            steps = [FEATURE_REGISTRY[name] for name in selected]
            try:
                enriched_df, feature_names = apply_features(raw_df, steps=steps)
                st.session_state[SS_ENRICHED_DF] = enriched_df
                st.session_state[SS_SELECTED_FEATURES] = selected
                st.success(f"Built {len(feature_names)} features.")
            except Exception as exc:
                st.error(f"Feature build failed: {exc}")
                return

    enriched_df = st.session_state.get(SS_ENRICHED_DF)
    if enriched_df is not None:
        st.subheader("Enriched DataFrame Preview")
        st.dataframe(enriched_df.head(20), use_container_width=True)

    # ── LLM feature generation ────────────────────────────────────────
    # st.subheader("LLM Feature Generation")
    # llm_desc = st.text_input(
    #     "Describe a new feature (e.g., 'multiply ret_1d and hl_range')",
    #     value="",
    # )
    # if st.button("Generate Feature with Gemini"):
    #     if not llm_desc:
    #         st.warning("Enter a feature description.")
    #     elif enriched_df is None:
    #         st.warning("Build features first before generating new ones.")
    #     else:
    #         try:
    #             code, feature_name = generate_feature(
    #                 llm_desc, enriched_df, list(enriched_df.columns)
    #             )
    #             st.session_state["_llm_code"] = code
    #             st.session_state["_llm_feature_name"] = feature_name
    #             st.success(f"Generated feature: `{feature_name}`")
    #         except Exception as exc:
    #             st.error(f"LLM generation failed: {exc}")

    # if "_llm_code" in st.session_state:
    #     st.code(st.session_state["_llm_code"], language="python")
    #     if st.button("Apply Generated Feature"):
    #         try:
    #             result = safe_exec_feature(st.session_state["_llm_code"], enriched_df)
    #             new_name = st.session_state.get("_llm_feature_name", "llm_feature")
    #             enriched_df[new_name] = result
    #             st.session_state[SS_ENRICHED_DF] = enriched_df
    #             # Also register as a custom feature step
    #             def _make_fn(code_str):
    #                 def _fn(df):
    #                     ns = {"__builtins__": {}, "pd": pd, "np": __import__("numpy"), "df": df}
    #                     exec(code_str, ns)
    #                     df[new_name] = ns["compute"](df)
    #                     return df
    #                 return _fn

    #             custom_step = FeatureStep(new_name, _make_fn(st.session_state["_llm_code"]))
    #             FEATURE_REGISTRY[new_name] = custom_step
    #             selected.append(new_name)
    #             st.session_state[SS_SELECTED_FEATURES] = selected
    #             st.success(f"Applied `{new_name}` — DataFrame now has {len(enriched_df.columns)} columns.")
    #             del st.session_state["_llm_code"]
    #             del st.session_state["_llm_feature_name"]
    #         except Exception as exc:
    #             st.error(f"Apply failed: {exc}")

    # ── Label config ──────────────────────────────────────────────────
    st.subheader("Label Configuration")
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.slider("Horizon (days)", 1, 20, st.session_state.get(SS_HORIZON, 5))
    with col2:
        label_method = st.radio("Label method", ["expanding", "median"], index=0)

    if st.button("Build Dataset (X, y)", type="primary"):
        if enriched_df is None:
            st.warning("Build features first.")
        else:
            clear_downstream("data_feature")
            # Ensure ret_1d exists (required by add_target_and_label)
            if "ret_1d" not in enriched_df.columns and "ret_1d" in FEATURE_REGISTRY:
                enriched_df = FEATURE_REGISTRY["ret_1d"].fn(enriched_df)
                st.session_state[SS_ENRICHED_DF] = enriched_df
            try:
                X, y, meta_df = _build_dataset_cached(
                    enriched_df, horizon, label_method, selected
                )
                st.session_state[SS_X] = X
                st.session_state[SS_Y] = y
                st.session_state[SS_META_DF] = meta_df
                st.session_state[SS_HORIZON] = horizon
                st.session_state[SS_LABEL_METHOD] = label_method
                st.session_state[SS_TARGET_COL] = f"rv_fwd_{horizon}d"
                st.success(
                    f"Dataset ready: X={X.shape}, y={y.shape}, "
                    f"class balance: {y.value_counts().to_dict()}"
                )
            except Exception as exc:
                st.error(f"Dataset build failed: {exc}")

    X = st.session_state.get(SS_X)
    if X is not None:
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", len(X))
        col2.metric("Features", X.shape[1])
        y = st.session_state.get(SS_Y)
        if y is not None:
            col3.metric("Class balance", str(y.value_counts().to_dict()))


@st.cache_data(show_spinner="Building dataset...")
def _build_dataset_cached(
    df: pd.DataFrame,
    horizon: int,
    label_method: str,
    feature_names: list[str],
) -> tuple:
    data, target_col = add_target_and_label(df, horizon=horizon, label_method=label_method)

    meta_cols = ["Date", "Close", target_col, "y"]
    meta_cols = [c for c in meta_cols if c in data.columns]

    required_cols = feature_names + [target_col, "y"]
    valid = data[required_cols].notna().all(axis=1)
    data = data.loc[valid].reset_index(drop=True)

    X = data[feature_names].copy()
    y = data["y"].copy()
    meta_df = data[meta_cols].copy()
    return X, y, meta_df
