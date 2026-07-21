"""Human-clarification tool (LangGraph interrupt)."""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def ask_user(question: str, options: list[str] | None = None) -> str:
    """Ask the student a clarifying question before proposing writes. Pauses until they answer."""
    payload: dict[str, Any] = {
        "type": "clarify",
        "question": question,
        "options": list(options or []),
    }
    answer = interrupt(payload)
    return str(answer)
