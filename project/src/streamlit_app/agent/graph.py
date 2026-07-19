"""LangGraph ReAct tutor agent."""
from __future__ import annotations

from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from project.src.streamlit_app.agent.tools_clarify import ask_user
from project.src.streamlit_app.agent.tools_read import READ_TOOLS
from project.src.streamlit_app.agent.tools_write import WRITE_TOOLS

SYSTEM_PROMPT = """You are an AI tutor for a realized-volatility (RV) prediction teaching lab built in Streamlit.

Stations:
1. ETL — teaching pack preview or live data escape hatch
2. Feature Engineering — select features / label method
3. Model Design — train/test split, CV, Optuna tuning
4. Implementation — OOF metrics, SHAP
5. Comparison — Quick Browse grid + Custom Experiment (parquet workspaces only)

Rules:
- Prefer read tools (get_page_context, ls_features, get_feature, list_*) before proposing writes.
- Use ls_features to browse feature names + short descriptions; use get_feature for formula/source.
- Use ask_user when symbol, horizon, dates, features, or model settings are underspecified.
- Write tools ONLY propose changes (propose_set_*). Never claim a setting already changed.
- Tell the student they must click Apply on the proposed action in the chat for writes to take effect.
- Be concise and pedagogical (under ~200 words for normal replies).
- Teaching symbols: SPY, QQQ. Horizons: 5, 10.
"""

_checkpointer = MemorySaver()
_agents: dict[tuple[str, str], Any] = {}


def get_tutor_agent(*, api_key: str, model_name: str):
    """Return a compiled ReAct agent cached per (api_key, model)."""
    cache_key = (api_key, model_name)
    if cache_key not in _agents:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
        )
        tools = [*READ_TOOLS, *WRITE_TOOLS, ask_user]
        _agents[cache_key] = create_react_agent(
            llm,
            tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=_checkpointer,
        )
    return _agents[cache_key]


def clear_agent_cache() -> None:
    _agents.clear()
