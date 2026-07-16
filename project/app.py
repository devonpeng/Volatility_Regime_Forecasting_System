"""Streamlit ML Experiment Workbench — Entry Point."""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from project.src.streamlit_app.session_manager import (
    init_session_state,
    has_data_loaded,
    has_features_built,
    has_train_test_split,
    has_model_results,
)

st.set_page_config(
    page_title="ML Workbench",
    page_icon="🧪",
    layout="wide",
)

init_session_state()

# ── Sidebar navigation ─────────────────────────────────────────────────
PAGES = {
    "📥 Station 1: ETL": "etl",
    "🔧 Station 2: Feature Engineering": "feature_eng",
    "🧠 Station 3: Model Design": "model_design",
    "📊 Station 4: Implementation": "implementation",
}

PAGES_EXTRA = {
    "DDF ML Pipeline": None,
    "📐 Feature Comparison": "comparison",
}

st.sidebar.title("🧪 DDF ML Pipeline")
page_label = st.sidebar.radio("Navigate", list(PAGES.keys()))
current_page = PAGES[page_label]

st.sidebar.markdown("---")
st.sidebar.markdown("**Reference**")
extra_label = st.sidebar.radio("", list(PAGES_EXTRA.keys()), key="extra_radio")
current_extra = PAGES_EXTRA[extra_label]

# ── Session state summary in sidebar ───────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Session Status**")
st.sidebar.markdown(f"- Data loaded: {'✅' if has_data_loaded() else '❌'}")
st.sidebar.markdown(f"- Features built: {'✅' if has_features_built() else '❌'}")
st.sidebar.markdown(f"- Train/Test split: {'✅' if has_train_test_split() else '❌'}")
st.sidebar.markdown(f"- Model trained: {'✅' if has_model_results() else '❌'}")

# ── Route to the selected page ─────────────────────────────────────────
# Reference pages (extra) take priority over main stations
if current_extra is not None and current_extra == "comparison":
    from project.src.streamlit_app.pages.comparison import render
    render()
elif current_page == "etl":
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
