"""Unit tests for ai_tutor helpers (no Streamlit UI render paths)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from project.src.streamlit_app import ai_tutor
from project.src.streamlit_app.session_manager import (
    SS_DATA_SOURCE,
    SS_DATE_END,
    SS_DATE_START,
    SS_HORIZON,
    SS_LLM_API_KEY,
    SS_LLM_MODEL,
    SS_RAW_DF,
    SS_SYMBOL,
    SS_USER_ID,
)


def test_get_active_llm_creds_session_overrides_db_and_env():
    session = {
        SS_LLM_API_KEY: "session-key",
        SS_LLM_MODEL: "session-model",
        SS_USER_ID: 1,
    }
    mock_st = MagicMock()
    mock_st.session_state = session

    with (
        patch.object(ai_tutor, "st", mock_st),
        patch.object(ai_tutor, "load_dotenv"),
        patch.object(
            ai_tutor.db,
            "get_user_by_id",
            return_value={"api_key": "db-key"},
        ),
        patch.dict(
            "os.environ",
            {"LLM_API_KEY": "env-key", "LLM_MODEL": "env-model"},
            clear=False,
        ),
    ):
        key, model = ai_tutor.get_active_llm_creds()

    assert key == "session-key"
    assert model == "session-model"


def test_get_active_llm_creds_falls_back_to_env():
    mock_st = MagicMock()
    mock_st.session_state = {SS_LLM_API_KEY: "", SS_LLM_MODEL: "", SS_USER_ID: None}

    with (
        patch.object(ai_tutor, "st", mock_st),
        patch.object(ai_tutor, "load_dotenv"),
        patch.dict(
            "os.environ",
            {"LLM_API_KEY": "env-only", "LLM_MODEL": "env-model"},
            clear=False,
        ),
    ):
        key, model = ai_tutor.get_active_llm_creds()

    assert key == "env-only"
    assert model == "env-model"


def test_get_page_context_etl():
    raw = pd.DataFrame({"Close": [1, 2, 3]})
    session = {
        SS_SYMBOL: "SPY",
        SS_HORIZON: 5,
        SS_DATE_START: "2020-01-01",
        SS_DATE_END: "2020-12-31",
        SS_RAW_DF: raw,
        SS_DATA_SOURCE: "parquet",
    }
    mock_st = MagicMock()
    mock_st.session_state = session

    with patch.object(ai_tutor, "st", mock_st):
        ctx = ai_tutor.get_page_context("etl")
        unknown = ai_tutor.get_page_context("not_a_page")

    assert "SPY" in ctx
    assert "horizon=5d" in ctx
    assert "3 rows" in ctx
    assert "source=parquet" in ctx
    assert unknown == ""


def test_format_trace_step():
    assert (
        ai_tutor._format_trace_step({"kind": "thought", "summary": "thinking"})
        == "Thought: thinking"
    )
    assert (
        ai_tutor._format_trace_step(
            {"kind": "tool", "name": "search", "summary": "ok"}
        )
        == "Tool `search` — ok"
    )
    assert (
        ai_tutor._format_trace_step(
            {"kind": "tool_result", "name": "search", "summary": "done"}
        )
        == "Result `search` — done"
    )


def test_compute_shap_bundle_logistic_coefficient_fallback():
    X = pd.DataFrame(
        {
            "f0": np.linspace(0, 1, 20),
            "f1": np.linspace(1, 0, 20),
            "f2": np.ones(20),
        }
    )
    model = SimpleNamespace(coef_=np.array([[0.1, 2.0, 0.5]]))

    with patch("shap.LinearExplainer", side_effect=RuntimeError("no linear")):
        result = ai_tutor.compute_shap_bundle(
            model, X, model_name="logistic", max_rows=10
        )

    assert result["mode"] == "coefficients"
    assert result["n_rows"] == 10  # truncated by max_rows
    assert set(result["mean_abs"]) == {"f0", "f1", "f2"}
    assert result["mean_abs"]["f1"] == 2.0
    assert "summary_fig" in result
