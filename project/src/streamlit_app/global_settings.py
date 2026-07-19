"""Left-sidebar Global Settings: symbol / horizon / date range + Load Teaching Pack."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from project.src.streamlit_app.precomputed import (
    DEFAULT_PACK_START,
    TEACHING_HORIZONS,
    TEACHING_SYMBOLS,
    cached_load_precomputed,
    date_bounds,
    parquet_available,
)
from project.src.streamlit_app.session_manager import (
    GS_END,
    GS_HORIZON,
    GS_START,
    GS_SYMBOL,
    SS_DATE_END,
    SS_DATE_START,
    SS_HORIZON,
    SS_SYMBOL,
    SS_WELCOME_DONE,
    clear_from_feature_eng,
    reset_pipeline_keep_globals,
)
from project.src.streamlit_app.workspace import (
    apply_parquet_slice_to_session,
    checkpoint_current_workspace,
    flush_pending_global_setting_sync,
    sync_global_setting_widgets,
)


def load_teaching_pack(
    symbol: str,
    horizon: int,
    start: str,
    end: str,
    *,
    checkpoint_reason: str = "load teaching pack",
) -> None:
    """Apply parquet slice, clear downstream feature pipeline, checkpoint.

    Does not write gs_* keys (widgets may already exist in this run).
    Draft widgets already hold the values being loaded.
    """
    clear_from_feature_eng()
    apply_parquet_slice_to_session(symbol, int(horizon), start, end)
    if checkpoint_reason:
        checkpoint_current_workspace(checkpoint_reason)


def render_global_settings_sidebar() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Global Settings**")

    missing = [
        f"{s}_{h}d"
        for s in TEACHING_SYMBOLS
        for h in TEACHING_HORIZONS
        if not parquet_available(s, h)
    ]
    if missing:
        st.sidebar.error(
            "Missing packs: "
            + ", ".join(missing)
            + ". Run `uv run python scripts/precompute_data.py`."
        )
        return

    # Pending sync from AI apply / restore must run before widgets exist.
    flush_pending_global_setting_sync()
    sync_global_setting_widgets(force=False)

    symbols = list(TEACHING_SYMBOLS)
    horizons = list(TEACHING_HORIZONS)

    if st.session_state.get(GS_SYMBOL) not in symbols:
        st.session_state[GS_SYMBOL] = symbols[0]
    st.sidebar.selectbox("Symbol", symbols, key=GS_SYMBOL)

    if int(st.session_state.get(GS_HORIZON, 5)) not in horizons:
        st.session_state[GS_HORIZON] = horizons[0]
    st.sidebar.radio(
        "Horizon",
        horizons,
        format_func=lambda h: f"{h}-day",
        key=GS_HORIZON,
        horizontal=True,
    )

    symbol = st.session_state[GS_SYMBOL]
    horizon = int(st.session_state[GS_HORIZON])

    try:
        probe = cached_load_precomputed(symbol, horizon)
        dmin, dmax = date_bounds(probe)
    except Exception as exc:
        st.sidebar.error(str(exc))
        return

    default_start = max(pd.Timestamp(DEFAULT_PACK_START), dmin)
    default_end = min(pd.Timestamp("2024-12-31"), dmax)
    if default_start > default_end:
        default_start, default_end = dmin, dmax

    def _clamp(val) -> dt.date:
        try:
            d = pd.Timestamp(val).date()
        except Exception:
            d = default_start.date()
        if d < dmin.date():
            return dmin.date()
        if d > dmax.date():
            return dmax.date()
        return d

    st.session_state[GS_START] = _clamp(st.session_state.get(GS_START, default_start.date()))
    st.session_state[GS_END] = _clamp(st.session_state.get(GS_END, default_end.date()))

    start = st.sidebar.date_input(
        "Start date",
        min_value=dmin.date(),
        max_value=dmax.date(),
        key=GS_START,
    )
    end = st.sidebar.date_input(
        "End date",
        min_value=dmin.date(),
        max_value=dmax.date(),
        key=GS_END,
    )

    applied_sym = st.session_state.get(SS_SYMBOL, "?")
    applied_h = st.session_state.get(SS_HORIZON, "?")
    applied_ds = st.session_state.get(SS_DATE_START, "?")
    applied_de = st.session_state.get(SS_DATE_END, "?")
    loaded = bool(st.session_state.get(SS_WELCOME_DONE)) and st.session_state.get("raw_df") is not None
    if loaded:
        st.sidebar.caption(
            f"Loaded: {applied_sym} · {applied_h}d · {applied_ds} → {applied_de}"
        )
    else:
        st.sidebar.caption("No teaching pack loaded yet.")

    if start > end:
        st.sidebar.warning("Start must be on or before end.")
    elif st.sidebar.button("Load Teaching Pack", type="primary", key="gs_load_btn"):
        load_teaching_pack(
            symbol,
            horizon,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        st.rerun()

    if st.sidebar.button("Reset All", key="gs_reset_all_btn"):
        reset_pipeline_keep_globals()
        checkpoint_current_workspace("reset all")
        st.rerun()
