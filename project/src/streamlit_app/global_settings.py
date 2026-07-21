"""Left-sidebar Global Settings: symbol / horizon + Reset All.

Date range + Load Teaching Pack live in Station 1 (see ``render_teaching_pack_loader``).
"""
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


def _as_ymd(val) -> str:
    return pd.Timestamp(val).strftime("%Y-%m-%d")


def _draft_differs_from_loaded() -> bool:
    if not (
        st.session_state.get(SS_WELCOME_DONE)
        and st.session_state.get("raw_df") is not None
    ):
        return False
    try:
        if st.session_state.get(GS_SYMBOL) != st.session_state.get(SS_SYMBOL):
            return True
        if int(st.session_state.get(GS_HORIZON, 5)) != int(
            st.session_state.get(SS_HORIZON, 5)
        ):
            return True
        if _as_ymd(st.session_state.get(GS_START)) != str(
            st.session_state.get(SS_DATE_START, "")
        ):
            return True
        if _as_ymd(st.session_state.get(GS_END)) != str(
            st.session_state.get(SS_DATE_END, "")
        ):
            return True
    except Exception:
        return True
    return False


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

    applied_sym = st.session_state.get(SS_SYMBOL, "?")
    applied_h = st.session_state.get(SS_HORIZON, "?")
    applied_ds = st.session_state.get(SS_DATE_START, "?")
    applied_de = st.session_state.get(SS_DATE_END, "?")
    loaded = bool(st.session_state.get(SS_WELCOME_DONE)) and st.session_state.get("raw_df") is not None
    if loaded:
        st.sidebar.caption(
            f"Loaded: {applied_sym} · {applied_h}d · {applied_ds} → {applied_de}"
        )
        if _draft_differs_from_loaded():
            st.sidebar.warning(
                "Settings changed — go to Station 1 and **Load Teaching Pack**."
            )
    else:
        st.sidebar.caption(
            "No teaching pack loaded yet. Set dates in Station 1 and Load Teaching Pack."
        )

    if st.sidebar.button("Reset All", key="gs_reset_all_btn"):
        reset_pipeline_keep_globals()
        checkpoint_current_workspace("reset all")
        st.rerun()


def render_teaching_pack_loader() -> None:
    """Station 1: date range + Load Teaching Pack (uses sidebar Symbol/Horizon drafts)."""
    missing = [
        f"{s}_{h}d"
        for s in TEACHING_SYMBOLS
        for h in TEACHING_HORIZONS
        if not parquet_available(s, h)
    ]
    if missing:
        st.error(
            "Missing packs: "
            + ", ".join(missing)
            + ". Run `uv run python scripts/precompute_data.py`."
        )
        return

    symbol = st.session_state.get(GS_SYMBOL, TEACHING_SYMBOLS[0])
    horizon = int(st.session_state.get(GS_HORIZON, TEACHING_HORIZONS[0]))

    try:
        probe = cached_load_precomputed(symbol, horizon)
        dmin, dmax = date_bounds(probe)
    except Exception as exc:
        st.error(str(exc))
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

    st.markdown(
        f"**Teaching pack** · draft symbol/horizon from sidebar: "
        f"`{symbol}` · `{horizon}-day`"
    )
    col_s, col_e, col_btn = st.columns([2, 2, 2])
    with col_s:
        start = st.date_input(
            "Start date",
            min_value=dmin.date(),
            max_value=dmax.date(),
            key=GS_START,
        )
    with col_e:
        end = st.date_input(
            "End date",
            min_value=dmin.date(),
            max_value=dmax.date(),
            key=GS_END,
        )
    with col_btn:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        load_clicked = st.button(
            "Load Teaching Pack", type="primary", key="station1_load_pack_btn"
        )

    if start > end:
        st.warning("Start must be on or before end.")
    elif load_clicked:
        load_teaching_pack(
            symbol,
            horizon,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        st.rerun()
