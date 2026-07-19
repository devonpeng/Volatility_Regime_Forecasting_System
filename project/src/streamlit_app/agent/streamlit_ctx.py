"""Bridge Streamlit ScriptRunContext into LangGraph tool worker threads.

LangGraph's ToolNode runs tools on a ContextThreadPoolExecutor. That copies
contextvars, but Streamlit stores ScriptRunContext on the *thread object*, so
worker threads see an empty ``st.session_state`` unless we re-attach the ctx.
"""
from __future__ import annotations

import threading
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

_ACTIVE_SCRIPT_RUN_CTX: Any | None = None


def bind_script_run_ctx() -> None:
    """Capture the current Streamlit ScriptRunContext for later tool threads."""
    global _ACTIVE_SCRIPT_RUN_CTX
    _ACTIVE_SCRIPT_RUN_CTX = get_script_run_ctx(suppress_warning=True)


def clear_script_run_ctx() -> None:
    """Drop the captured context after a tutor turn."""
    global _ACTIVE_SCRIPT_RUN_CTX
    _ACTIVE_SCRIPT_RUN_CTX = None


def ensure_script_run_ctx() -> bool:
    """Attach the bound ScriptRunContext to this thread if missing.

    Returns True when a ScriptRunContext is available on this thread afterward.
    """
    if get_script_run_ctx(suppress_warning=True) is not None:
        return True
    ctx = _ACTIVE_SCRIPT_RUN_CTX
    if ctx is None:
        return False
    add_script_run_ctx(threading.current_thread(), ctx)
    return get_script_run_ctx(suppress_warning=True) is not None


def wrap_tool_with_streamlit_ctx(tool: BaseTool) -> BaseTool:
    """Return a copy of *tool* that re-attaches ScriptRunContext before running."""
    if isinstance(tool, StructuredTool):
        orig_func = tool.func
        orig_coro = tool.coroutine
        updates: dict[str, Any] = {}
        if orig_func is not None:

            def wrapped_func(*args: Any, **kwargs: Any) -> Any:
                ensure_script_run_ctx()
                return orig_func(*args, **kwargs)

            wrapped_func.__name__ = getattr(orig_func, "__name__", "wrapped_func")
            wrapped_func.__doc__ = getattr(orig_func, "__doc__", None)
            updates["func"] = wrapped_func
        if orig_coro is not None:

            async def wrapped_coro(*args: Any, **kwargs: Any) -> Any:
                ensure_script_run_ctx()
                return await orig_coro(*args, **kwargs)

            wrapped_coro.__name__ = getattr(orig_coro, "__name__", "wrapped_coro")
            wrapped_coro.__doc__ = getattr(orig_coro, "__doc__", None)
            updates["coroutine"] = wrapped_coro
        if not updates:
            return tool
        return tool.model_copy(update=updates)

    # Non-StructuredTool fallback: wrap sync/async run entrypoints if present.
    orig_run = getattr(tool, "_run", None)
    if callable(orig_run):

        def wrapped_run(*args: Any, **kwargs: Any) -> Any:
            ensure_script_run_ctx()
            return orig_run(*args, **kwargs)

        object.__setattr__(tool, "_run", wrapped_run)
    orig_arun = getattr(tool, "_arun", None)
    if callable(orig_arun):

        async def wrapped_arun(*args: Any, **kwargs: Any) -> Any:
            ensure_script_run_ctx()
            return await orig_arun(*args, **kwargs)

        object.__setattr__(tool, "_arun", wrapped_arun)
    return tool


def wrap_tools_with_streamlit_ctx(tools: list[BaseTool]) -> list[BaseTool]:
    return [wrap_tool_with_streamlit_ctx(t) for t in tools]
