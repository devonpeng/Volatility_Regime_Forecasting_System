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
    "📊 Data & Feature": "data_feature",
    "✂️ Train/Test Split": "train_test",
    "🤖 Model Training": "model_train",
    "📈 Evaluation & Insight": "evaluation",
}

st.sidebar.title("🧪 ML Workbench")
page_label = st.sidebar.radio("Navigate", list(PAGES.keys()))
current_page = PAGES[page_label]

# ── Session state summary in sidebar ───────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Session Status**")
st.sidebar.markdown(f"- Data loaded: {'✅' if has_data_loaded() else '❌'}")
st.sidebar.markdown(f"- Features built: {'✅' if has_features_built() else '❌'}")
st.sidebar.markdown(f"- Train/Test split: {'✅' if has_train_test_split() else '❌'}")
st.sidebar.markdown(f"- Model trained: {'✅' if has_model_results() else '❌'}")

# ── Route to the selected page ─────────────────────────────────────────
if current_page == "data_feature":
    from project.src.streamlit_app.pages.data_feature import render
    render()
elif current_page == "train_test":
    from project.src.streamlit_app.pages.train_test import render
    render()
elif current_page == "model_train":
    from project.src.streamlit_app.pages.model_train import render
    render()
elif current_page == "evaluation":
    from project.src.streamlit_app.pages.evaluation import render
    render()
