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
    apply_feature,
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


@st.cache_data
def _get_feature_source(name: str) -> str:
    """Cache the source code of a feature step — only read once per name."""
    step = FEATURE_REGISTRY[name]
    return inspect.getsource(step.fn)


def render_etl() -> None:
    st.header("📥 Station 1: ETL — Data Loading")

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
            clear_downstream("etl")
            with st.spinner("Loading data..."):
                raw_df = _load_data_cached(
                    symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), use_synthetic
                )
            st.session_state[SS_RAW_DF] = raw_df
            st.success(f"Loaded {len(raw_df)} rows (source: {raw_df.attrs.get('source', 'unknown')})")
            st.rerun()

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


def render_feng() -> None:
    st.header("🔧 Station 2: Feature Engineering")

    raw_df = st.session_state.get(SS_RAW_DF)
    if raw_df is None:
        st.warning("Complete Station 1 (ETL) first to load data.")
        return

    # ── Feature selection ──────────────────────────────────────────────
    st.subheader("Feature Selection")
    all_names = [step.name for step in DEFAULT_FEATURE_STEPS]
    # rv_hist_5d / rv_hist_20d are prerequisites for rv_ratio_5_20, rv_accel, vix_rv_spread
    CORE_BASE_FEATURES = ["ret_1d", "rv_hist_5d", "rv_hist_20d"]
    RECOMMENDED_FEATURES = [
        "parkinson_rv", "vix_level", "vix_term_structure", "kalman_trend_medium",
        "ret_1d", "kalman_price_medium", "kalman_ret_1d_medium", "vix_rv_spread",
        "rv_ratio_5_20", "rv_accel", "overnight_ret", "vix_change_1d",
    ]
    def _on_preset_change() -> None:
        st.session_state["_fs_version"] = st.session_state.get("_fs_version", 0) + 1
        st.session_state[SS_SELECTED_FEATURES] = []

    preset = st.radio(
        "Feature preset",
        options=["All Features (49)", "LASSO Recommended"],
        horizontal=True,
        index=0,
        key="feature_preset",
        on_change=_on_preset_change,
    )
    default_selection = (
        all_names if preset == "All Features (49)" else RECOMMENDED_FEATURES
    )
    if "_fs_version" not in st.session_state:
        st.session_state["_fs_version"] = 0
    selected = st.multiselect(
        "Select features",
        options=all_names,
        default=st.session_state.get(SS_SELECTED_FEATURES) or default_selection,
        key=f"feat_sel_{st.session_state['_fs_version']}",
    )
    st.session_state[SS_SELECTED_FEATURES] = selected

    enriched_df = st.session_state.get(SS_ENRICHED_DF)

    # ── Transparent computation view ───────────────────────────────────
    with st.expander("Feature Computation Details", expanded=False):
        for step in DEFAULT_FEATURE_STEPS:
            if step.name in selected:
                with st.expander(f"{step.name}"):
                    st.code(_get_feature_source(step.name), language="python")

    # ── Build features ─────────────────────────────────────────────────
    if st.button("Build Features", type="primary"):
        if not selected:
            st.warning("Select at least one feature.")
        else:
            clear_downstream("feature_eng")
            # Auto-inject core base features required by many derived features
            missing_base = [f for f in CORE_BASE_FEATURES if f not in selected]
            if missing_base:
                selected = missing_base + selected
            # ret_1d must be first — rv_hist_5d, rv_hist_20d and many others depend on it
            if "ret_1d" in selected:
                selected.remove("ret_1d")
                selected.insert(0, "ret_1d")
            steps = [FEATURE_REGISTRY[name] for name in selected]
            try:
                enriched_df = raw_df.copy()
                feature_names = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, step in enumerate(steps):
                    status_text.text(f"Building feature {i+1}/{len(steps)}: {step.name}")
                    enriched_df = apply_feature(enriched_df, step)
                    feature_names.append(step.name)
                    progress_bar.progress((i + 1) / len(steps))
                st.session_state[SS_ENRICHED_DF] = enriched_df
                st.session_state[SS_SELECTED_FEATURES] = selected
                st.success(f"Built {len(feature_names)} features.")
                st.rerun()
            except Exception as exc:
                st.error(f"Feature build failed: {exc}")
                return

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
            clear_downstream("feature_eng")
            # Ensure ret_1d exists (required by add_target_and_label)
            if "ret_1d" not in enriched_df.columns and "ret_1d" in FEATURE_REGISTRY:
                enriched_df = FEATURE_REGISTRY["ret_1d"].fn(enriched_df)
                st.session_state[SS_ENRICHED_DF] = enriched_df
            try:
                X, y, meta_df, full_data = _build_dataset_cached(
                    enriched_df, horizon, label_method, selected
                )
                st.session_state[SS_X] = X
                st.session_state[SS_Y] = y
                st.session_state[SS_META_DF] = meta_df
                st.session_state["_full_data"] = full_data
                st.session_state[SS_HORIZON] = horizon
                st.session_state[SS_LABEL_METHOD] = label_method
                st.session_state[SS_TARGET_COL] = f"rv_fwd_{horizon}d"
                st.success(
                    f"Dataset ready: X={X.shape}, y={y.shape}, "
                    f"class balance: {y.value_counts().to_dict()}"
                )
                st.rerun()
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

    # ── Auto feature selection (after Build Dataset, for All Features) ─
    enriched_df = st.session_state.get(SS_ENRICHED_DF)
    if (
        preset == "All Features (49)"
        and raw_df is not None
        and X is not None
        and enriched_df is not None
    ):
        with st.container():
            st.markdown("**Auto feature selection**")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                auto_top_k = st.number_input(
                    "Top K", 3, 30, 12, 1, key="auto_top_k"
                )
            with col_b:
                auto_method = st.selectbox(
                    "Method",
                    ["lasso", "tree_topk", "ttest"],
                    index=1,
                    key="auto_method",
                )

            if st.button("Auto-Select Features", type="primary"):
                from project.src.features.selection import select_features_on_fold

                with st.spinner("Running feature selection..."):
                    full_df, _ = apply_features(
                        raw_df.copy(), steps=DEFAULT_FEATURE_STEPS
                    )
                    data, target_col = add_target_and_label(
                        full_df,
                        horizon=st.session_state.get(SS_HORIZON, 5),
                        label_method=st.session_state.get(SS_LABEL_METHOD, "expanding"),
                    )
                    req_cols = all_names + [target_col, "y"]
                    valid = data[req_cols].notna().all(axis=1)
                    data_lasso = data.loc[valid].reset_index(drop=True)

                    if len(data_lasso) < 100:
                        st.warning(
                            f"Only {len(data_lasso)} valid samples — "
                            "selection may be unreliable."
                        )

                    _, _, fs_result = select_features_on_fold(
                        data_lasso[all_names],
                        data_lasso["y"],
                        data_lasso[all_names],
                        method=auto_method,
                        top_k=auto_top_k,
                        random_state=42,
                    )
                    new_selected = fs_result.selected_features

                    X_clean, y_clean, meta_clean, _ = _build_dataset_cached(
                        full_df,
                        st.session_state.get(SS_HORIZON, 5),
                        st.session_state.get(SS_LABEL_METHOD, "expanding"),
                        new_selected,
                    )

                    clear_downstream("feature_eng")
                    st.session_state[SS_SELECTED_FEATURES] = new_selected
                    st.session_state[SS_X] = X_clean
                    st.session_state[SS_Y] = y_clean
                    st.session_state[SS_META_DF] = meta_clean
                    st.session_state[SS_HORIZON] = st.session_state.get(SS_HORIZON, 5)
                    st.session_state[SS_LABEL_METHOD] = st.session_state.get(
                        SS_LABEL_METHOD, "expanding"
                    )
                    st.session_state[SS_TARGET_COL] = target_col
                    st.session_state["_auto_dataset_built"] = True
                    st.session_state["_fs_version"] += 1

                    st.success(
                        f"Selected {len(new_selected)}/{len(all_names)} features. "
                        f"Dataset ready: X={X_clean.shape}, "
                        f"class balance: {y_clean.value_counts().to_dict()}"
                    )
                    with st.expander(
                        "Selected Features & Scores", expanded=True
                    ):
                        scores_df = fs_result.scores.to_frame(
                            "score"
                        ).sort_values("score", ascending=False)
                        st.dataframe(scores_df, use_container_width=True)
                    st.rerun()


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
    full_data = data.copy()
    return X, y, meta_df, full_data
