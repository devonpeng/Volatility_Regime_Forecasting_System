"""Invoke / resume helpers bridging Streamlit ↔ LangGraph tutor."""
from __future__ import annotations

import json
import uuid
from typing import Any, Callable

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from project.src.streamlit_app.agent.graph import get_tutor_agent
from project.src.streamlit_app.session_manager import (
    SS_AI_CLARIFY,
    SS_AI_THREAD_ID,
    SS_CHAT_HISTORY,
    SS_PENDING_ACTIONS,
    SS_WORKSPACE_ID,
)

TraceStepCallback = Callable[[dict[str, Any]], None]


def ensure_thread_id() -> str:
    tid = st.session_state.get(SS_AI_THREAD_ID) or ""
    if not tid:
        ws = st.session_state.get(SS_WORKSPACE_ID) or "local"
        tid = f"{ws}-{uuid.uuid4().hex[:10]}"
        st.session_state[SS_AI_THREAD_ID] = tid
    return tid


def reset_thread_id() -> str:
    """Mint a new LangGraph thread (drops old MemorySaver history)."""
    ws = st.session_state.get(SS_WORKSPACE_ID) or "local"
    tid = f"{ws}-{uuid.uuid4().hex[:10]}"
    st.session_state[SS_AI_THREAD_ID] = tid
    return tid


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "get"):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _summarize_tool_content(raw: Any, *, limit: int = 220) -> str:
    text = raw if isinstance(raw, str) else _extract_text(raw)
    if not text:
        return "(empty)"
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if data.get("status") == "pending_approval":
                return f"pending: {data.get('summary') or data.get('op') or 'action'}"
            if data.get("status") == "error":
                return f"error: {data.get('error', text)}"[:limit]
            if "error" in data:
                return f"error: {data.get('error')}"[:limit]
        text = json.dumps(data, default=str)
    except (json.JSONDecodeError, TypeError):
        pass
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tool_call_summary(tc: Any) -> tuple[str, str]:
    if isinstance(tc, dict):
        name = str(tc.get("name") or "tool")
        args = tc.get("args") or tc.get("arguments") or {}
    else:
        name = str(getattr(tc, "name", None) or "tool")
        args = getattr(tc, "args", None) or {}
    try:
        arg_s = json.dumps(args, default=str)
    except TypeError:
        arg_s = str(args)
    arg_s = " ".join(arg_s.split())
    if len(arg_s) > 160:
        arg_s = arg_s[:159] + "…"
    return name, arg_s


def _collect_pending_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        raw = msg.content
        if not isinstance(raw, str):
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("status") == "pending_approval":
            pending.append(data)
    return pending


def _last_ai_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = _extract_text(msg.content)
            if text:
                return text
    return ""


def _parse_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    inter = result.get("__interrupt__")
    if not inter:
        return None
    first = inter[0] if isinstance(inter, (list, tuple)) else inter
    value = getattr(first, "value", first)
    if isinstance(value, dict):
        return value
    return {"type": "clarify", "question": str(value), "options": []}


def _merge_pending(pending: list[dict[str, Any]]) -> list[str]:
    """Merge pending actions into session; return new ids."""
    if not pending:
        return []
    existing = list(st.session_state.get(SS_PENDING_ACTIONS) or [])
    seen = {p.get("id") for p in existing}
    new_ids: list[str] = []
    for p in pending:
        pid = p.get("id")
        if pid not in seen:
            existing.append(p)
            seen.add(pid)
            if pid:
                new_ids.append(str(pid))
    st.session_state[SS_PENDING_ACTIONS] = existing
    return new_ids


def _apply_invoke_result(
    result: dict[str, Any],
    *,
    trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    messages = list(result.get("messages") or [])
    pending = _collect_pending_from_messages(messages)
    new_ids = _merge_pending(pending)

    clarify = _parse_interrupt(result)
    if clarify and clarify.get("type") == "clarify":
        st.session_state[SS_AI_CLARIFY] = clarify
    else:
        st.session_state[SS_AI_CLARIFY] = None

    reply = _last_ai_text(messages)
    if clarify and not reply:
        reply = f"I need a bit more detail: {clarify.get('question', '')}"

    return {
        "reply": reply,
        "pending": pending,
        "pending_ids": new_ids,
        "clarify": clarify,
        "interrupted": clarify is not None,
        "trace": list(trace or []),
        "messages": messages,
    }


def _trace_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Build a displayable trace from a full message list (non-streaming fallback)."""
    steps: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            text = _extract_text(msg.content)
            if text:
                steps.append({"kind": "thought", "name": None, "summary": text})
            for tc in getattr(msg, "tool_calls", None) or []:
                name, arg_s = _tool_call_summary(tc)
                steps.append(
                    {
                        "kind": "tool",
                        "name": name,
                        "summary": f"args: {arg_s}" if arg_s else "calling…",
                    }
                )
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            steps.append(
                {
                    "kind": "tool_result",
                    "name": str(name),
                    "summary": _summarize_tool_content(msg.content),
                }
            )
    return steps


def _emit_trace_steps(
    messages: list[Any],
    *,
    seen: int,
    on_step: TraceStepCallback | None,
    trace: list[dict[str, Any]],
) -> int:
    """Process messages[seen:] into trace steps; return new seen count."""
    for msg in messages[seen:]:
        if isinstance(msg, AIMessage):
            text = _extract_text(msg.content)
            if text:
                step = {"kind": "thought", "name": None, "summary": text}
                trace.append(step)
                if on_step:
                    on_step(step)
            for tc in getattr(msg, "tool_calls", None) or []:
                name, arg_s = _tool_call_summary(tc)
                step = {
                    "kind": "tool",
                    "name": name,
                    "summary": f"args: {arg_s}" if arg_s else "calling…",
                }
                trace.append(step)
                if on_step:
                    on_step(step)
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or "tool"
            step = {
                "kind": "tool_result",
                "name": str(name),
                "summary": _summarize_tool_content(msg.content),
            }
            trace.append(step)
            if on_step:
                on_step(step)
    return len(messages)


def stream_tutor(
    user_text: str,
    *,
    api_key: str,
    model_name: str,
    page_name: str = "",
    on_step: TraceStepCallback | None = None,
) -> dict[str, Any]:
    """Stream a tutor turn; optionally invoke *on_step* for live UI updates."""
    st.session_state["_current_page"] = page_name
    agent = get_tutor_agent(api_key=api_key, model_name=model_name)
    thread_id = ensure_thread_id()
    config = {"configurable": {"thread_id": thread_id}}

    resuming = bool(st.session_state.get(SS_AI_CLARIFY))
    if resuming:
        payload: Any = Command(resume=user_text)
        st.session_state[SS_AI_CLARIFY] = None
    else:
        prefix = f"[Current page: {page_name}] " if page_name else ""
        payload = {"messages": [HumanMessage(content=prefix + user_text)]}

    trace: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}
    seen = 0

    try:
        for chunk in agent.stream(payload, config=config, stream_mode="values"):
            if not isinstance(chunk, dict):
                continue
            final_state = chunk
            messages = list(chunk.get("messages") or [])
            seen = _emit_trace_steps(messages, seen=seen, on_step=on_step, trace=trace)
    except Exception:
        # Fallback to invoke if stream fails mid-flight after partial progress
        # is undesirable — prefer a clean invoke when stream never started.
        if not final_state:
            result = agent.invoke(payload, config=config)
            if not isinstance(result, dict):
                result = {"messages": []}
            messages = list(result.get("messages") or [])
            trace = _trace_from_messages(messages)
            if on_step:
                for step in trace:
                    on_step(step)
            return _apply_invoke_result(result, trace=trace)
        raise

    # Interrupt payloads may only appear on get_state after stream ends
    if "__interrupt__" not in final_state:
        try:
            snap = agent.get_state(config)
            if snap is not None:
                values = getattr(snap, "values", None) or {}
                if isinstance(values, dict) and values.get("messages"):
                    final_state = {**final_state, **values}
                interrupts = []
                for task in getattr(snap, "tasks", None) or []:
                    for ig in getattr(task, "interrupts", None) or []:
                        interrupts.append(ig)
                if interrupts:
                    final_state = {**final_state, "__interrupt__": interrupts}
        except Exception:
            pass

    if not trace and final_state.get("messages"):
        trace = _trace_from_messages(list(final_state.get("messages") or []))

    return _apply_invoke_result(final_state, trace=trace)


def invoke_tutor(
    user_text: str,
    *,
    api_key: str,
    model_name: str,
    page_name: str = "",
) -> dict[str, Any]:
    """Send a user message into the graph (non-streaming). Prefer stream_tutor for UI."""
    return stream_tutor(
        user_text,
        api_key=api_key,
        model_name=model_name,
        page_name=page_name,
        on_step=None,
    )


def resume_after_apply(note: str, *, api_key: str, model_name: str) -> dict[str, Any]:
    """Optional short ack after Apply / Dismiss (user message into agent)."""
    return invoke_tutor(
        note,
        api_key=api_key,
        model_name=model_name,
        page_name=st.session_state.get("_current_page", ""),
    )


def append_chat(
    role: str,
    content: str,
    *,
    trace: list[dict[str, Any]] | None = None,
    pending_ids: list[str] | None = None,
    clarify: dict[str, Any] | None = None,
    msg_id: str | None = None,
) -> dict[str, Any]:
    history = list(st.session_state.get(SS_CHAT_HISTORY) or [])
    entry: dict[str, Any] = {
        "id": msg_id or uuid.uuid4().hex[:12],
        "role": role,
        "content": content,
    }
    if trace is not None:
        entry["trace"] = list(trace)
    if pending_ids is not None:
        entry["pending_ids"] = list(pending_ids)
    if clarify is not None:
        entry["clarify"] = clarify
    history.append(entry)
    st.session_state[SS_CHAT_HISTORY] = history[-60:]
    return entry


def clear_chat() -> None:
    st.session_state[SS_CHAT_HISTORY] = []
    st.session_state[SS_PENDING_ACTIONS] = []
    st.session_state[SS_AI_CLARIFY] = None
    reset_thread_id()


def delete_chat_from(msg_id: str) -> None:
    """Remove message *msg_id* and everything after it; reset agent thread."""
    history = list(st.session_state.get(SS_CHAT_HISTORY) or [])
    idx = next((i for i, m in enumerate(history) if m.get("id") == msg_id), None)
    if idx is None:
        return
    removed = history[idx:]
    st.session_state[SS_CHAT_HISTORY] = history[:idx]
    pending_drop: set[str] = set()
    for m in removed:
        for pid in m.get("pending_ids") or []:
            pending_drop.add(str(pid))
    if pending_drop:
        st.session_state[SS_PENDING_ACTIONS] = [
            a
            for a in (st.session_state.get(SS_PENDING_ACTIONS) or [])
            if str(a.get("id")) not in pending_drop
        ]
    st.session_state[SS_AI_CLARIFY] = None
    reset_thread_id()


def truncate_chat_after(msg_id: str) -> dict[str, Any] | None:
    """Keep messages up to and including *msg_id*; drop the rest. Reset thread."""
    history = list(st.session_state.get(SS_CHAT_HISTORY) or [])
    idx = next((i for i, m in enumerate(history) if m.get("id") == msg_id), None)
    if idx is None:
        return None
    removed = history[idx + 1 :]
    kept = history[: idx + 1]
    st.session_state[SS_CHAT_HISTORY] = kept
    pending_drop: set[str] = set()
    for m in removed:
        for pid in m.get("pending_ids") or []:
            pending_drop.add(str(pid))
    if pending_drop:
        st.session_state[SS_PENDING_ACTIONS] = [
            a
            for a in (st.session_state.get(SS_PENDING_ACTIONS) or [])
            if str(a.get("id")) not in pending_drop
        ]
    st.session_state[SS_AI_CLARIFY] = None
    reset_thread_id()
    return kept[idx] if kept else None
