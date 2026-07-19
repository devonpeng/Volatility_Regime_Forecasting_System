"""Page 1–2: ETL preview / live escape + Feature Engineering."""
from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from project.src.data.load_data import get_data_with_fallback
from project.src.features.core import (
    DEFAULT_FEATURE_STEPS,
    FEATURE_REGISTRY,
    add_target_and_label,
    apply_feature,
    apply_features,
)
from project.src.streamlit_app.precomputed import (
    extract_xy_from_wide,
    feature_column_names,
)
from project.src.streamlit_app.session_manager import (
    SS_DATA_SOURCE,
    SS_DATE_END,
    SS_DATE_START,
    SS_ENRICHED_DF,
    SS_FULL_DF,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_META_DF,
    SS_RAW_DF,
    SS_SELECTED_FEATURES,
    SS_SYMBOL,
    SS_TARGET_COL,
    SS_X,
    SS_Y,
    clear_downstream,
    is_parquet_workspace,
)
from project.src.streamlit_app.workspace import checkpoint_current_workspace


@st.cache_data(show_spinner="Downloading market data...")
def _load_data_cached(symbol: str, start_date: str, end_date: str, use_synthetic: bool) -> pd.DataFrame:
    return get_data_with_fallback(symbol, start_date, end_date, use_synthetic=use_synthetic)


@st.cache_data
def _get_feature_source(name: str) -> str:
    step = FEATURE_REGISTRY[name]
    return inspect.getsource(step.fn)


def render_etl() -> None:
    st.header("Station 1: ETL — Data Loading")

    raw_df = st.session_state.get(SS_RAW_DF)
    data_source = st.session_state.get(SS_DATA_SOURCE, "parquet")

    if data_source == "parquet" and raw_df is not None:
        st.success(
            f"Teaching pack loaded: **{st.session_state.get(SS_SYMBOL)}** · "
            f"horizon={st.session_state.get(SS_HORIZON)}d · "
            f"{st.session_state.get(SS_DATE_START)} → {st.session_state.get(SS_DATE_END)} · "
            f"{len(raw_df):,} rows (read-only preview)."
        )
        st.info(
            "Change symbol / horizon / dates in the left **Global Settings** sidebar "
            "and click **Load Teaching Pack**. "
            "Features and dual labels are already in the Parquet — use Station 2 to select features. "
            "Comparison (Station 5) is available on this path."
        )
    elif data_source == "live" and raw_df is not None:
        st.warning(
            f"Live data mode: **{st.session_state.get(SS_SYMBOL)}** · "
            f"{st.session_state.get(SS_DATE_START)} → {st.session_state.get(SS_DATE_END)} · "
            f"{len(raw_df):,} rows. Station 5 (Comparison) is disabled."
        )
    elif raw_df is None:
        st.info(
            "Load a teaching pack from **Global Settings** (left sidebar), "
            "or use **Load Live Data** below."
        )

    with st.expander("Advanced: Live download / synthetic", expanded=raw_df is None):
        st.warning(
            "Live mode rebuilds features in-session and **disables Station 5 (Comparison)**. "
            "Prefer Global Settings → Load Teaching Pack for coursework."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input(
                "Live symbol",
                value=st.session_state.get(SS_SYMBOL, "SPY"),
                key="live_symbol",
            )
        with col2:
            start_date = st.date_input(
                "Live start",
                value=pd.Timestamp(st.session_state.get(SS_DATE_START, "2020-01-01")),
                key="live_start",
            )
        with col3:
            end_date = st.date_input(
                "Live end",
                value=pd.Timestamp(st.session_state.get(SS_DATE_END, "2024-12-31")),
                key="live_end",
            )
        use_synthetic = st.checkbox(
            "Use synthetic data (no network)", value=False, key="live_synthetic"
        )

        if st.button("Load Live Data", type="secondary", key="load_live_data_btn"):
            clear_downstream("etl")
            with st.spinner("Loading data..."):
                live_df = _load_data_cached(
                    symbol,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    use_synthetic,
                )
            st.session_state[SS_RAW_DF] = live_df
            st.session_state[SS_FULL_DF] = None
            st.session_state[SS_ENRICHED_DF] = None
            st.session_state[SS_SYMBOL] = symbol.upper()
            st.session_state[SS_DATE_START] = start_date.strftime("%Y-%m-%d")
            st.session_state[SS_DATE_END] = end_date.strftime("%Y-%m-%d")
            st.session_state[SS_DATA_SOURCE] = "live"
            st.session_state["welcome_done"] = True
            checkpoint_current_workspace("live data load")
            st.success(
                f"Loaded {len(live_df)} rows (source: {live_df.attrs.get('source', 'unknown')})"
            )
            st.rerun()

    raw_df = st.session_state.get(SS_RAW_DF)
    if raw_df is None:
        return

    st.subheader("Data Preview")
    preview_cols = [
        c
        for c in (
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "vix",
            "vix3m",
            "vvix",
        )
        if c in raw_df.columns
    ]
    show = raw_df[preview_cols] if preview_cols else raw_df
    col_top, col_tail = st.columns(2)
    with col_top:
        st.caption("Head")
        st.dataframe(show.head(10), use_container_width=True)
    with col_tail:
        st.caption("Tail")
        st.dataframe(show.tail(10), use_container_width=True)

    st.caption(f"Columns: {len(raw_df.columns)} · Rows: {len(raw_df):,}")


def render_feng() -> None:
    st.header("Station 2: Feature Engineering")

    raw_df = st.session_state.get(SS_RAW_DF)
    if raw_df is None:
        st.warning("Complete Station 1 (ETL) first to load data.")
        return

    if is_parquet_workspace():
        _render_feng_parquet(raw_df)
    else:
        _render_feng_live(raw_df)


def _render_feng_parquet(raw_df: pd.DataFrame) -> None:
    st.subheader("Inspect & Select Features (precomputed)")
    st.caption(
        "All engineered features are already in the teaching pack. "
        "Select a subset and choose the label method — no rebuild needed."
    )

    all_names = [c for c in feature_column_names() if c in raw_df.columns]
    RECOMMENDED = [
        "parkinson_rv",
        "vix_level",
        "vix_term_structure",
        "kalman_trend_medium",
        "ret_1d",
        "kalman_price_medium",
        "kalman_ret_1d_medium",
        "vix_rv_spread",
        "rv_ratio_5_20",
        "rv_accel",
        "overnight_ret",
        "vix_change_1d",
    ]
    RECOMMENDED = [f for f in RECOMMENDED if f in all_names]

    def _on_preset_change() -> None:
        st.session_state["_fs_version"] = st.session_state.get("_fs_version", 0) + 1
        st.session_state[SS_SELECTED_FEATURES] = []

    preset = st.radio(
        "Feature preset",
        options=["All Features", "LASSO Recommended"],
        horizontal=True,
        index=0,
        key="feature_preset",
        on_change=_on_preset_change,
    )
    default_selection = all_names if preset == "All Features" else RECOMMENDED
    if "_fs_version" not in st.session_state:
        st.session_state["_fs_version"] = 0
    selected = st.multiselect(
        "Select features",
        options=all_names,
        default=st.session_state.get(SS_SELECTED_FEATURES) or default_selection,
        key=f"feat_sel_{st.session_state['_fs_version']}",
    )
    st.session_state[SS_SELECTED_FEATURES] = selected

    with st.expander("Feature Computation Details", expanded=False):
        for name in selected:
            if name in FEATURE_REGISTRY:
                with st.expander(name):
                    st.code(_get_feature_source(name), language="python")

    # Distributions for a picked feature
    if selected:
        pick = st.selectbox("Inspect feature distribution", selected)
        if pick in raw_df.columns:
            st.line_chart(raw_df.set_index("Date")[pick] if "Date" in raw_df.columns else raw_df[pick])

    st.subheader("Label Configuration")
    horizon = int(st.session_state.get(SS_HORIZON, 5))
    st.write(f"Horizon (from Global Settings): **{horizon}-day**")
    label_method = st.radio(
        "Label method",
        ["expanding", "median"],
        index=0 if st.session_state.get(SS_LABEL_METHOD, "expanding") == "expanding" else 1,
        help="Uses precomputed y_expanding / y_median columns.",
    )

    if st.button("Confirm Selection (build X, y)", type="primary"):
        if not selected:
            st.warning("Select at least one feature.")
        else:
            try:
                from project.src.streamlit_app.agent.apply import commit_features

                msg = commit_features(
                    selected,
                    label_method,
                    checkpoint_reason="station2 selection",
                )
                st.success(msg)
                st.rerun()
            except Exception as exc:
                st.error(f"Dataset build failed: {exc}")

    _show_dataset_summary()
    _auto_select_block(raw_df, all_names, preset == "All Features")


def _render_feng_live(raw_df: pd.DataFrame) -> None:
    """Original in-session feature build path for live / synthetic data."""
    st.subheader("Feature Selection")
    all_names = [step.name for step in DEFAULT_FEATURE_STEPS]
    CORE_BASE_FEATURES = ["ret_1d", "rv_hist_5d", "rv_hist_20d"]
    RECOMMENDED_FEATURES = [
        "parkinson_rv",
        "vix_level",
        "vix_term_structure",
        "kalman_trend_medium",
        "ret_1d",
        "kalman_price_medium",
        "kalman_ret_1d_medium",
        "vix_rv_spread",
        "rv_ratio_5_20",
        "rv_accel",
        "overnight_ret",
        "vix_change_1d",
    ]

    def _on_preset_change() -> None:
        st.session_state["_fs_version"] = st.session_state.get("_fs_version", 0) + 1
        st.session_state[SS_SELECTED_FEATURES] = []

    preset = st.radio(
        "Feature preset",
        options=["All Features (49)", "LASSO Recommended"],
        horizontal=True,
        index=0,
        key="feature_preset_live",
        on_change=_on_preset_change,
    )
    default_selection = (
        all_names if preset.startswith("All") else RECOMMENDED_FEATURES
    )
    if "_fs_version" not in st.session_state:
        st.session_state["_fs_version"] = 0
    selected = st.multiselect(
        "Select features",
        options=all_names,
        default=st.session_state.get(SS_SELECTED_FEATURES) or default_selection,
        key=f"feat_sel_live_{st.session_state['_fs_version']}",
    )
    st.session_state[SS_SELECTED_FEATURES] = selected

    enriched_df = st.session_state.get(SS_ENRICHED_DF)

    with st.expander("Feature Computation Details", expanded=False):
        for step in DEFAULT_FEATURE_STEPS:
            if step.name in selected:
                with st.expander(step.name):
                    st.code(_get_feature_source(step.name), language="python")

    if st.button("Build Features", type="primary"):
        if not selected:
            st.warning("Select at least one feature.")
        else:
            clear_downstream("feature_eng")
            missing_base = [f for f in CORE_BASE_FEATURES if f not in selected]
            if missing_base:
                selected = missing_base + selected
            if "ret_1d" in selected:
                selected = ["ret_1d"] + [f for f in selected if f != "ret_1d"]
            steps = [FEATURE_REGISTRY[name] for name in selected]
            try:
                enriched_df = raw_df.copy()
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, step in enumerate(steps):
                    status_text.text(f"Building feature {i+1}/{len(steps)}: {step.name}")
                    enriched_df = apply_feature(enriched_df, step)
                    progress_bar.progress((i + 1) / len(steps))
                st.session_state[SS_ENRICHED_DF] = enriched_df
                st.session_state[SS_SELECTED_FEATURES] = selected
                st.success(f"Built {len(selected)} features.")
                st.rerun()
            except Exception as exc:
                st.error(f"Feature build failed: {exc}")
                return

    if enriched_df is not None:
        st.subheader("Enriched DataFrame Preview")
        st.dataframe(enriched_df.head(20), use_container_width=True)

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
                checkpoint_current_workspace("station2 live dataset")
                st.success(
                    f"Dataset ready: X={X.shape}, y={y.shape}, "
                    f"class balance: {y.value_counts().to_dict()}"
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Dataset build failed: {exc}")

    _show_dataset_summary()


def _show_dataset_summary() -> None:
    X = st.session_state.get(SS_X)
    if X is not None:
        st.subheader("Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", len(X))
        col2.metric("Features", X.shape[1])
        y = st.session_state.get(SS_Y)
        if y is not None:
            col3.metric("Class balance", str(y.value_counts().to_dict()))


def _auto_select_block(raw_df: pd.DataFrame, all_names: list[str], enabled: bool) -> None:
    X = st.session_state.get(SS_X)
    if not enabled or X is None:
        return

    st.markdown("**Auto feature selection**")
    col_a, col_b = st.columns(2)
    with col_a:
        auto_top_k = st.number_input("Top K", 3, 30, 12, 1, key="auto_top_k")
    with col_b:
        auto_method = st.selectbox(
            "Method", ["lasso", "tree_topk", "ttest"], index=1, key="auto_method"
        )

    if st.button("Auto-Select Features", type="primary"):
        from project.src.features.selection import select_features_on_fold

        with st.spinner("Running feature selection..."):
            horizon = st.session_state.get(SS_HORIZON, 5)
            label_method = st.session_state.get(SS_LABEL_METHOD, "expanding")
            try:
                X_all, y_all, _, full = extract_xy_from_wide(
                    raw_df,
                    horizon=horizon,
                    label_method=label_method,
                    selected_features=all_names,
                )
                _, _, fs_result = select_features_on_fold(
                    X_all,
                    y_all,
                    X_all,
                    method=auto_method,
                    top_k=int(auto_top_k),
                    random_state=42,
                )
                new_selected = fs_result.selected_features
                X_clean, y_clean, meta_clean, full_data = extract_xy_from_wide(
                    raw_df,
                    horizon=horizon,
                    label_method=label_method,
                    selected_features=new_selected,
                )
                clear_downstream("feature_eng")
                st.session_state[SS_SELECTED_FEATURES] = new_selected
                st.session_state[SS_X] = X_clean
                st.session_state[SS_Y] = y_clean
                st.session_state[SS_META_DF] = meta_clean
                st.session_state["_full_data"] = full_data
                st.session_state[SS_TARGET_COL] = f"rv_fwd_{horizon}d"
                st.session_state["_fs_version"] = st.session_state.get("_fs_version", 0) + 1
                checkpoint_current_workspace("auto feature select")
                st.success(f"Selected {len(new_selected)} features. X={X_clean.shape}")
                st.dataframe(
                    fs_result.scores.to_frame("score").sort_values("score", ascending=False),
                    use_container_width=True,
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Auto-select failed: {exc}")


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
