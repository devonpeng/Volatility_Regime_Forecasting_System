"""Legacy welcome helpers — Global Settings sidebar replaced the full-page form."""
from __future__ import annotations

from project.src.streamlit_app.session_manager import SS_WELCOME_DONE
import streamlit as st


def needs_welcome() -> bool:
    """True when no teaching pack / live data has been loaded yet."""
    if st.session_state.get(SS_WELCOME_DONE) and st.session_state.get("raw_df") is not None:
        return False
    return not bool(st.session_state.get(SS_WELCOME_DONE))


def render_welcome() -> None:
    """Deprecated: use Station 1 Load Teaching Pack."""
    st.info(
        "In **Station 1**, set dates and click **Load Teaching Pack** "
        "(Symbol / Horizon are in the left **Global Settings** sidebar). "
        "Or use **Load Live Data** for the escape hatch."
    )
