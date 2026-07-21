"""Streamlit RV Prediction Lab — Entry Point."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from project.src.streamlit_app.ai_tutor import render_ai_rail
from project.src.streamlit_app.auth import ensure_authenticated, render_workspace_sidebar
from project.src.streamlit_app.global_settings import render_global_settings_sidebar
from project.src.streamlit_app.session_manager import (
    PAGE_ORDER,
    SS_AI_RAIL_COLLAPSED,
    has_data_loaded,
    has_features_built,
    has_model_results,
    has_train_test_split,
    init_session_state,
    is_parquet_workspace,
)
from project.src.streamlit_app.workspace import check_tab_sync_warning, flush_pending_station_widget_syncs

st.set_page_config(
    page_title="RV Prediction Lab",
    page_icon="📈",
    layout="wide",
)

init_session_state()

if not ensure_authenticated():
    st.stop()

if not render_workspace_sidebar():
    st.title("RV Prediction Lab")
    st.info("Create or select a workspace in the sidebar to continue.")
    st.stop()

check_tab_sync_warning()

render_global_settings_sidebar()

# ── Sidebar navigation ─────────────────────────────────────────────────
PAGES = {
    "Station 1: ETL": "etl",
    "Station 2: Feature Engineering": "feature_eng",
    "Station 3: Model Design": "model_design",
    "Station 4: Implementation": "implementation",
    "Station 5: Comparison": "comparison",
}
PAGE_LABELS = list(PAGES.keys())
PAGE_ID_TO_LABEL = {pid: label for label, pid in PAGES.items()}
STATION_NAV_KEY = "station_nav"
STATION_NAV_PENDING = "_pending_station_nav"


def _render_station_nav_footer(current_page_id: str) -> None:
    """Bottom-left Previous / Next station buttons (endpoints disabled)."""
    try:
        idx = PAGE_ORDER.index(current_page_id)
    except ValueError:
        return
    labels = [PAGE_ID_TO_LABEL[pid] for pid in PAGE_ORDER]
    st.markdown("---")
    left, _ = st.columns([1, 3])
    with left:
        c_prev, c_next = st.columns(2)
        with c_prev:
            prev_disabled = idx <= 0
            if st.button(
                "← Previous station",
                key="station_prev_btn",
                disabled=prev_disabled,
                width="stretch",
            ):
                # Cannot mutate widget key after radio is instantiated — defer.
                st.session_state[STATION_NAV_PENDING] = labels[idx - 1]
                st.rerun()
        with c_next:
            next_disabled = idx >= len(PAGE_ORDER) - 1
            if st.button(
                "Next station →",
                key="station_next_btn",
                disabled=next_disabled,
                width="stretch",
            ):
                st.session_state[STATION_NAV_PENDING] = labels[idx + 1]
                st.rerun()


# Apply deferred footer navigation *before* the radio widget is created.
_pending_nav = st.session_state.pop(STATION_NAV_PENDING, None)
if _pending_nav in PAGE_LABELS:
    st.session_state[STATION_NAV_KEY] = _pending_nav

st.sidebar.markdown("---")
st.sidebar.markdown("**Navigate**")
page_label = st.sidebar.radio(
    "Station",
    PAGE_LABELS,
    label_visibility="collapsed",
    key=STATION_NAV_KEY,
)
current_page = PAGES[page_label]

if current_page == "comparison" and not is_parquet_workspace():
    st.sidebar.warning("Comparison disabled on live path")

st.sidebar.markdown("---")
st.sidebar.markdown("**Session Status**")
st.sidebar.markdown(f"- Data loaded: {'🟢' if has_data_loaded() else '🔴'}")
st.sidebar.markdown(f"- Features ready: {'🟢' if has_features_built() else '🔴'}")
st.sidebar.markdown(f"- Train/Test split: {'🟢' if has_train_test_split() else '🔴'}")
st.sidebar.markdown(f"- Model trained: {'🟢' if has_model_results() else '🔴'}")
st.sidebar.markdown(
    f"- Data source: {st.session_state.get('data_source', '?')}"
)

# ── Main: content | AI rail ────────────────────────────────────────────
ai_collapsed = bool(st.session_state.get(SS_AI_RAIL_COLLAPSED))
if ai_collapsed:
    content_col, ai_col = st.columns([48, 1], gap="small")
else:
    content_col, ai_col = st.columns([3, 1], gap="medium")

with content_col:
    flush_pending_station_widget_syncs()

    if not has_data_loaded():
        st.title("RV Prediction Lab")
        st.info(
            "In **Station 1**, set the date range and click **Load Teaching Pack** "
            "(Symbol / Horizon are in the left **Global Settings** sidebar). "
            "Or use **Load Live Data** on Station 1."
        )

    if current_page == "etl":
        from project.src.streamlit_app.pages.data_feature import render_etl

        render_etl()
    elif current_page == "feature_eng":
        from project.src.streamlit_app.pages.data_feature import render_feng

        render_feng()
    elif current_page == "model_design":
        from project.src.streamlit_app.pages.model_design import render

        render()
    elif current_page == "implementation":
        from project.src.streamlit_app.pages.implementation import render

        render()
    elif current_page == "comparison":
        from project.src.streamlit_app.pages.comparison import render

        render()

    _render_station_nav_footer(current_page)

with ai_col:
    if ai_collapsed:
        if st.button("◀", key="ai_rail_expand_btn", help="Expand AI panel"):
            st.session_state[SS_AI_RAIL_COLLAPSED] = False
            st.rerun()
        st.caption("AI")
    else:
        if st.button("▶", key="ai_rail_collapse_btn", help="Collapse AI panel"):
            st.session_state[SS_AI_RAIL_COLLAPSED] = True
            st.rerun()
        render_ai_rail(current_page, st)
