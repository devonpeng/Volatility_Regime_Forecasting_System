"""AI Settings + right-rail tutor (LangGraph) + SHAP helpers."""
from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.preprocessing import RobustScaler

from project.src.streamlit_app import db
from project.src.streamlit_app.agent.apply import apply_action
from project.src.streamlit_app.agent.runtime import (
    append_chat,
    clear_chat,
    delete_chat_from,
    stream_tutor,
    truncate_chat_after,
)
from project.src.streamlit_app.llm_helper import (
    analyze_results,
    interpret_shap,
    resolve_llm_credentials,
)
from project.src.streamlit_app.session_manager import (
    SS_AI_CLARIFY,
    SS_BEST_MODEL_NAME,
    SS_CHAT_HISTORY,
    SS_COMPARED_EXPERIMENTS,
    SS_CUSTOM_EXPERIMENTS,
    SS_CV_N_SPLITS,
    SS_DATA_SOURCE,
    SS_DATE_END,
    SS_DATE_START,
    SS_ENRICHED_DF,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_LLM_API_KEY,
    SS_LLM_MODEL,
    SS_N_TRIALS,
    SS_OOF_METRICS,
    SS_PENDING_ACTIONS,
    SS_RAW_DF,
    SS_SELECTED_FEATURES,
    SS_SELECTED_MODELS,
    SS_SYMBOL,
    SS_USER_ID,
)


def get_active_llm_creds() -> tuple[str | None, str | None]:
    """session override → user.api_key → env."""
    load_dotenv()
    session_key = (st.session_state.get(SS_LLM_API_KEY) or "").strip()
    session_model = (st.session_state.get(SS_LLM_MODEL) or "").strip()

    user_key = None
    uid = st.session_state.get(SS_USER_ID)
    if uid:
        user = db.get_user_by_id(uid)
        if user and user.get("api_key"):
            user_key = user["api_key"]

    key = session_key or user_key or os.getenv("LLM_API_KEY", "").strip() or None
    model = session_model or os.getenv("LLM_MODEL", "").strip() or None
    return key, model


def render_ai_settings(container: Any = None, *, show_title: bool = True) -> None:
    """Render AI Settings into *container* (default: st)."""
    c = container or st
    if show_title:
        c.markdown("**AI Settings**")
    load_dotenv()
    env_key = os.getenv("LLM_API_KEY", "").strip()
    env_model = os.getenv("LLM_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"

    if "ai_rail_api_key" not in st.session_state:
        st.session_state["ai_rail_api_key"] = st.session_state.get(SS_LLM_API_KEY, "")
    c.text_input(
        "API Key override",
        type="password",
        help="Optional. Falls back to saved profile key, then .env",
        key="ai_rail_api_key",
    )
    st.session_state[SS_LLM_API_KEY] = st.session_state.get("ai_rail_api_key", "")

    model_options = [
        env_model,
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    seen: set[str] = set()
    model_options = [m for m in model_options if not (m in seen or seen.add(m))]
    if "ai_rail_model" not in st.session_state:
        cur = st.session_state.get(SS_LLM_MODEL) or env_model
        st.session_state["ai_rail_model"] = cur if cur in model_options else model_options[0]
    c.selectbox("Model", options=model_options, key="ai_rail_model")
    st.session_state[SS_LLM_MODEL] = st.session_state.get("ai_rail_model", env_model)

    remember = c.checkbox("Remember key on my account", value=False, key="ai_rail_remember")
    if remember and c.button("Save key to profile", key="ai_rail_save_key"):
        uid = st.session_state.get(SS_USER_ID)
        key = (st.session_state.get(SS_LLM_API_KEY) or "").strip()
        if uid and key:
            db.set_user_api_key(uid, key)
            c.success("Saved.")
        else:
            c.warning("Enter a key first.")

    if c.button("Clear saved profile key", key="ai_rail_clear_key"):
        uid = st.session_state.get(SS_USER_ID)
        if uid:
            db.set_user_api_key(uid, None)
            c.info("Cleared.")

    try:
        resolve_llm_credentials(*get_active_llm_creds())
        c.success("LLM credentials ready")
    except Exception:
        c.warning("LLM not configured")
        if env_key:
            c.caption(".env key present but model may be missing")


def render_ai_settings_sidebar() -> None:
    render_ai_settings(st.sidebar)


def get_page_context(page_name: str) -> str:
    contexts = {
        "etl": (
            f"Viewing {st.session_state.get(SS_SYMBOL)} data, "
            f"horizon={st.session_state.get(SS_HORIZON)}d, "
            f"{st.session_state.get(SS_DATE_START)}→{st.session_state.get(SS_DATE_END)}, "
            f"{len(st.session_state[SS_RAW_DF]) if st.session_state.get(SS_RAW_DF) is not None else 0} rows, "
            f"source={st.session_state.get(SS_DATA_SOURCE)}"
        ),
        "feature_eng": (
            f"Selected features: {st.session_state.get(SS_SELECTED_FEATURES)}, "
            f"label={st.session_state.get(SS_LABEL_METHOD)}, "
            f"enriched cols="
            f"{len(st.session_state[SS_ENRICHED_DF].columns) if st.session_state.get(SS_ENRICHED_DF) is not None else 0}"
        ),
        "model_design": (
            f"CV splits={st.session_state.get(SS_CV_N_SPLITS)}, "
            f"n_trials={st.session_state.get(SS_N_TRIALS)}, "
            f"models={st.session_state.get(SS_SELECTED_MODELS)}, "
            f"features={st.session_state.get(SS_SELECTED_FEATURES)}"
        ),
        "implementation": (
            f"Best model: {st.session_state.get(SS_BEST_MODEL_NAME)}, "
            f"OOF={st.session_state.get(SS_OOF_METRICS)}"
        ),
        "comparison": (
            f"Custom experiments={len(st.session_state.get(SS_CUSTOM_EXPERIMENTS) or [])}, "
            f"compared={len(st.session_state.get(SS_COMPARED_EXPERIMENTS) or [])}"
        ),
    }
    return contexts.get(page_name, "")


def _format_trace_step(step: dict[str, Any]) -> str:
    kind = step.get("kind")
    name = step.get("name")
    summary = step.get("summary") or ""
    if kind == "thought":
        return f"Thought: {summary}"
    if kind == "tool":
        return f"Tool `{name}` — {summary}"
    if kind == "tool_result":
        return f"Result `{name}` — {summary}"
    return str(summary or step)


def _pending_by_id() -> dict[str, dict[str, Any]]:
    return {
        str(a.get("id")): a
        for a in (st.session_state.get(SS_PENDING_ACTIONS) or [])
        if a.get("id")
    }


def _render_trace_expander(msg: dict[str, Any]) -> None:
    trace = msg.get("trace") or []
    if not trace:
        return
    with st.expander("Thinking & tools", expanded=False):
        for step in trace:
            st.caption(_format_trace_step(step))


def _render_inline_pending(msg: dict[str, Any], *, key_prefix: str) -> None:
    pending_map = _pending_by_id()
    ids = [str(pid) for pid in (msg.get("pending_ids") or []) if str(pid) in pending_map]
    if not ids:
        return
    st.markdown("**Proposed actions** (Apply required)")
    for aid in ids:
        action = pending_map[aid]
        st.markdown(f"**{action.get('summary', action.get('op'))}**")
        st.json(action.get("payload") or {})
        c1, c2 = st.columns(2)
        if c1.button("Apply", key=f"{key_prefix}_apply_{aid}", type="primary"):
            try:
                result_msg = apply_action(action)
                append_chat("assistant", f"(Applied) {result_msg}")
                st.session_state[SS_PENDING_ACTIONS] = [
                    a
                    for a in (st.session_state.get(SS_PENDING_ACTIONS) or [])
                    if str(a.get("id")) != aid
                ]
            except Exception as exc:
                st.error(f"Apply failed: {exc}")
            st.rerun()
        if c2.button("Dismiss", key=f"{key_prefix}_dismiss_{aid}"):
            append_chat(
                "assistant",
                f"(Dismissed) {action.get('summary', action.get('op'))}",
            )
            st.session_state[SS_PENDING_ACTIONS] = [
                a
                for a in (st.session_state.get(SS_PENDING_ACTIONS) or [])
                if str(a.get("id")) != aid
            ]
            st.rerun()


def _render_inline_clarify(page_name: str, *, key_prefix: str) -> None:
    clarify = st.session_state.get(SS_AI_CLARIFY)
    if not clarify:
        return
    st.info(clarify.get("question", "Need a bit more detail."))
    options = list(clarify.get("options") or [])
    if options:
        choice = st.radio(
            "Choose an option",
            options,
            key=f"{key_prefix}_clarify_choice",
        )
        if st.button("Submit answer", key=f"{key_prefix}_clarify_submit_opt"):
            _send_tutor_message(str(choice), page_name=page_name)
            st.rerun()
    answer = st.text_input("Or type your answer", key=f"{key_prefix}_clarify_text")
    if st.button("Send clarification", key=f"{key_prefix}_clarify_submit_text") and (
        answer or ""
    ).strip():
        _send_tutor_message(answer.strip(), page_name=page_name)
        st.rerun()


def _send_tutor_message(prompt: str, *, page_name: str) -> None:
    append_chat("user", prompt)
    try:
        key, model = get_active_llm_creds()
        if not key or not model:
            raise RuntimeError("LLM credentials not configured")
        trace_acc: list[dict[str, Any]] = []
        with st.status("Thinking…", expanded=True) as status:

            def _on_step(step: dict[str, Any]) -> None:
                trace_acc.append(step)
                status.write(_format_trace_step(step))

            result = stream_tutor(
                prompt,
                api_key=key,
                model_name=model,
                page_name=page_name,
                on_step=_on_step,
            )
            status.update(label="Done", state="complete", expanded=False)
        reply = result.get("reply") or "(No reply)"
        append_chat(
            "assistant",
            reply,
            trace=result.get("trace") or trace_acc,
            pending_ids=result.get("pending_ids") or [],
            clarify=result.get("clarify"),
        )
    except Exception as exc:
        append_chat("assistant", f"(Tutor unavailable: {exc})")


def _handle_edit_message(msg: dict[str, Any], *, page_name: str) -> None:
    msg_id = str(msg.get("id") or "")
    role = msg.get("role", "assistant")
    edit_key = f"ai_edit_{msg_id}"
    save_key = f"ai_edit_save_{msg_id}"
    cancel_key = f"ai_edit_cancel_{msg_id}"
    if st.session_state.get(f"_editing_{msg_id}"):
        if edit_key not in st.session_state:
            st.session_state[edit_key] = msg.get("content") or ""
        new_text = st.text_area(
            "Edit message",
            key=edit_key,
            height=100,
        )
        c1, c2 = st.columns(2)
        if c1.button("Save", key=save_key, type="primary"):
            text = (new_text or "").strip()
            st.session_state.pop(f"_editing_{msg_id}", None)
            st.session_state.pop(edit_key, None)
            if role == "user":
                truncate_chat_after(msg_id)
                history = list(st.session_state.get(SS_CHAT_HISTORY) or [])
                if history and history[-1].get("id") == msg_id:
                    st.session_state[SS_CHAT_HISTORY] = history[:-1]
                if text:
                    _send_tutor_message(text, page_name=page_name)
            else:
                history = list(st.session_state.get(SS_CHAT_HISTORY) or [])
                for i, m in enumerate(history):
                    if m.get("id") == msg_id:
                        history[i] = {**m, "content": text}
                        break
                st.session_state[SS_CHAT_HISTORY] = history
            st.rerun()
        if c2.button("Cancel", key=cancel_key):
            st.session_state.pop(f"_editing_{msg_id}", None)
            st.session_state.pop(edit_key, None)
            st.rerun()


_AI_RAIL_LAYOUT_CSS = """
<style>
/* Keep the AI rail column in view and full viewport height. */
div[data-testid="stHorizontalBlock"]:has(.st-key-ai_rail_root)
  > div[data-testid="stColumn"]:last-child {
  position: sticky !important;
  top: 0 !important;
  align-self: flex-start !important;
  height: 100vh !important;
  max-height: 100vh !important;
  overflow-x: hidden !important;
  overflow-y: auto !important;
}

/* Chat history fills remaining viewport above the pinned input. */
.st-key-ai_tutor_chat_scroll {
  height: calc(100vh - 15rem) !important;
  max-height: calc(100vh - 15rem) !important;
}

/* Pin tutor input to the viewport bottom; width synced via JS. */
.st-key-ai_tutor_chat_input {
  position: fixed !important;
  bottom: 0 !important;
  z-index: 1000 !important;
  background-color: var(--background-color, #ffffff) !important;
  padding: 0.35rem 0.35rem 0.65rem !important;
  box-sizing: border-box !important;
}

.st-key-ai_tutor_chat_input [data-testid="stChatInput"] {
  background-color: var(--background-color, #ffffff) !important;
}
</style>
"""

_AI_RAIL_LAYOUT_JS = """
<script>
(function () {
  function syncAiTutorInput() {
    const rail = document.querySelector(".st-key-ai_rail_root");
    const input = document.querySelector(".st-key-ai_tutor_chat_input");
    if (!rail || !input) return;
    const rect = rail.getBoundingClientRect();
    input.style.left = rect.left + "px";
    input.style.width = Math.max(rect.width, 120) + "px";
    input.style.right = "auto";
  }
  syncAiTutorInput();
  if (!window.__aiTutorInputSyncBound) {
    window.__aiTutorInputSyncBound = true;
    window.addEventListener("resize", syncAiTutorInput);
    const obs = new MutationObserver(function () {
      requestAnimationFrame(syncAiTutorInput);
    });
    obs.observe(document.body, { childList: true, subtree: true, attributes: true });
  }
})();
</script>
"""


def _inject_ai_rail_css() -> None:
    """Style-only HTML goes to the event container (no layout footprint)."""
    st.html(_AI_RAIL_LAYOUT_CSS)


def _inject_ai_rail_js() -> None:
    """Sync fixed chat-input width/left to the right rail."""
    st.html(
        '<div style="position:absolute;width:0;height:0;overflow:hidden">'
        f"{_AI_RAIL_LAYOUT_JS}"
        "</div>",
        unsafe_allow_javascript=True,
    )


def render_ai_tutor(page_name: str, container: Any = None) -> None:
    """Render AI Tutor chat + HITL into *container* (default: st)."""
    c = container or st
    st.session_state["_current_page"] = page_name
    c.markdown("**AI Tutor**")
    c.caption(f"Page: {page_name}")
    ctx = get_page_context(page_name)
    if ctx:
        c.caption(ctx[:200])

    top_l, top_r = c.columns([3, 1])
    if top_r.button("Clear chat", key="ai_tutor_clear"):
        clear_chat()
        st.rerun()
    top_l.caption("Writes need Apply. Edit/Delete resets agent memory.")

    history = st.session_state.get(SS_CHAT_HISTORY) or []
    # Migrate legacy entries missing id
    migrated = False
    for msg in history:
        if not msg.get("id"):
            msg["id"] = uuid.uuid4().hex[:12]
            migrated = True
    if migrated:
        st.session_state[SS_CHAT_HISTORY] = history

    # Fallback pixel height if CSS does not apply; overridden to fill viewport.
    chat_box = c.container(
        height=720,
        border=True,
        key="ai_tutor_chat_scroll",
        autoscroll=True,
    )
    with chat_box:
        for msg in history:
            role = msg.get("role", "assistant")
            msg_id = str(msg.get("id") or "")
            avatar_role = "user" if role == "user" else "assistant"
            with st.chat_message(avatar_role):
                if st.session_state.get(f"_editing_{msg_id}"):
                    _handle_edit_message(msg, page_name=page_name)
                else:
                    st.markdown(msg.get("content") or "")
                    if role == "assistant":
                        _render_trace_expander(msg)
                        _render_inline_pending(msg, key_prefix=f"msg_{msg_id}")
                    b1, b2, _ = st.columns([1, 1, 4])
                    if b1.button("Edit", key=f"ai_msg_edit_{msg_id}"):
                        st.session_state[f"_editing_{msg_id}"] = True
                        st.rerun()
                    if b2.button("Delete", key=f"ai_msg_del_{msg_id}"):
                        delete_chat_from(msg_id)
                        st.rerun()

        shown_ids: set[str] = set()
        for msg in history:
            for pid in msg.get("pending_ids") or []:
                shown_ids.add(str(pid))
        orphan = [
            a
            for a in (st.session_state.get(SS_PENDING_ACTIONS) or [])
            if str(a.get("id")) not in shown_ids
        ]
        if orphan:
            with st.chat_message("assistant"):
                st.markdown("**Pending actions**")
                for action in orphan:
                    aid = str(action.get("id") or "")
                    st.markdown(f"**{action.get('summary', action.get('op'))}**")
                    st.json(action.get("payload") or {})
                    c1, c2 = st.columns(2)
                    if c1.button("Apply", key=f"ai_orphan_apply_{aid}", type="primary"):
                        try:
                            result_msg = apply_action(action)
                            append_chat("assistant", f"(Applied) {result_msg}")
                            st.session_state[SS_PENDING_ACTIONS] = [
                                a
                                for a in (st.session_state.get(SS_PENDING_ACTIONS) or [])
                                if str(a.get("id")) != aid
                            ]
                        except Exception as exc:
                            st.error(f"Apply failed: {exc}")
                        st.rerun()
                    if c2.button("Dismiss", key=f"ai_orphan_dismiss_{aid}"):
                        append_chat(
                            "assistant",
                            f"(Dismissed) {action.get('summary', action.get('op'))}",
                        )
                        st.session_state[SS_PENDING_ACTIONS] = [
                            a
                            for a in (st.session_state.get(SS_PENDING_ACTIONS) or [])
                            if str(a.get("id")) != aid
                        ]
                        st.rerun()

        if st.session_state.get(SS_AI_CLARIFY):
            with st.chat_message("assistant"):
                st.markdown("**Tutor needs clarification**")
                _render_inline_clarify(page_name, key_prefix="live")

    prompt = c.chat_input("Ask the tutor…", key="ai_tutor_chat_input")
    if prompt and prompt.strip():
        _send_tutor_message(prompt.strip(), page_name=page_name)
        st.rerun()


def render_ai_tutor_sidebar(page_name: str) -> None:
    render_ai_tutor(page_name, st.sidebar)


def render_ai_rail(page_name: str, container: Any) -> None:
    """Full right-rail: settings + tutor + HITL."""
    _inject_ai_rail_css()
    rail = container.container(key="ai_rail_root")
    with rail:
        with st.expander("AI Settings", expanded=False):
            render_ai_settings(st, show_title=False)
        st.markdown("---")
        render_ai_tutor(page_name, st)
    _inject_ai_rail_js()


def render_analyze_button(context: dict, key: str = "ai_analyze") -> None:
    if st.button("Analyze these results with AI", key=key):
        with st.spinner("Calling LLM…"):
            try:
                api_key, model = get_active_llm_creds()
                text = analyze_results(context, api_key=api_key, model_name=model)
                st.session_state[f"_ai_analysis_{key}"] = text
            except Exception as exc:
                st.error(f"AI analyze failed: {exc}")
    text = st.session_state.get(f"_ai_analysis_{key}")
    if text:
        st.markdown(text)


def compute_shap_bundle(
    model: Any,
    X: pd.DataFrame,
    *,
    model_name: str,
    max_rows: int = 400,
) -> dict[str, Any]:
    """Return SHAP values + plotly figures for tree models; fallback for logistic."""
    import shap

    feature_names = list(X.columns)
    sample = X
    if len(sample) > max_rows:
        sample = sample.iloc[-max_rows:].copy()

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(sample.to_numpy())
    X_plot = pd.DataFrame(X_scaled, columns=feature_names)

    result: dict[str, Any] = {"feature_names": feature_names, "n_rows": len(sample)}

    if model_name in ("lgbm", "xgb"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_plot)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_arr = np.asarray(shap_values)
    else:
        try:
            explainer = shap.LinearExplainer(model, X_plot)
            shap_arr = np.asarray(explainer.shap_values(X_plot))
        except Exception:
            if hasattr(model, "coef_"):
                coef = np.abs(np.asarray(model.coef_).flatten())
                order = np.argsort(-coef)
                result["mean_abs"] = {
                    feature_names[i]: float(coef[i]) for i in order
                }
                result["mode"] = "coefficients"
                result["summary_fig"] = _bar_importance_fig(
                    [feature_names[i] for i in order[:15]],
                    [float(coef[i]) for i in order[:15]],
                    title="|Coefficient| magnitude (logistic fallback)",
                )
                return result
            raise

    mean_abs = np.abs(shap_arr).mean(axis=0)
    order = np.argsort(-mean_abs)
    result["mean_abs"] = {feature_names[i]: float(mean_abs[i]) for i in order}
    result["mode"] = "shap"
    result["shap_values"] = shap_arr
    result["X_plot"] = X_plot

    top_k = min(12, len(feature_names))
    fig = go.Figure()
    for rank, idx in enumerate(order[:top_k]):
        fig.add_trace(
            go.Scatter(
                x=shap_arr[:, idx],
                y=np.full(len(sample), top_k - rank)
                + np.random.default_rng(0).normal(0, 0.08, len(sample)),
                mode="markers",
                name=feature_names[idx],
                marker=dict(
                    size=5,
                    color=X_plot.iloc[:, idx],
                    colorscale="RdBu",
                    showscale=(rank == 0),
                    colorbar=dict(title="Feature value") if rank == 0 else None,
                ),
                hovertemplate=f"{feature_names[idx]}<br>SHAP=%{{x:.4f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title="SHAP summary (top features)",
        xaxis_title="SHAP value",
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(1, top_k + 1)),
            ticktext=[feature_names[i] for i in order[:top_k]][::-1],
        ),
        height=420,
        showlegend=False,
        template="plotly_white",
    )
    result["summary_fig"] = fig

    deps = []
    for idx in order[:3]:
        fname = feature_names[idx]
        dfig = go.Figure(
            data=[
                go.Scatter(
                    x=X_plot[fname],
                    y=shap_arr[:, idx],
                    mode="markers",
                    marker=dict(size=5, opacity=0.6),
                )
            ]
        )
        dfig.update_layout(
            title=f"Dependence: {fname}",
            xaxis_title=fname,
            yaxis_title="SHAP",
            height=300,
            template="plotly_white",
        )
        deps.append((fname, dfig))
    result["dependence"] = deps
    return result


def _bar_importance_fig(names: list[str], values: list[float], title: str) -> go.Figure:
    fig = go.Figure(go.Bar(x=values[::-1], y=names[::-1], orientation="h"))
    fig.update_layout(title=title, height=360, template="plotly_white")
    return fig


def render_shap_section(model: Any, X: pd.DataFrame, model_name: str) -> None:
    st.subheader("SHAP Analysis")
    if st.button("Compute SHAP", key="compute_shap"):
        with st.spinner("Computing SHAP..."):
            try:
                bundle = compute_shap_bundle(model, X, model_name=model_name)
                st.session_state["_shap_bundle"] = bundle
                try:
                    key, llm_model = get_active_llm_creds()
                    summary = {
                        "mode": bundle.get("mode"),
                        "top_mean_abs": dict(
                            list((bundle.get("mean_abs") or {}).items())[:10]
                        ),
                    }
                    st.session_state["_shap_ai"] = interpret_shap(
                        summary, api_key=key, model_name=llm_model
                    )
                except Exception as exc:
                    st.session_state["_shap_ai"] = f"(SHAP narrative unavailable: {exc})"
            except Exception as exc:
                st.error(f"SHAP failed: {exc}")

    bundle = st.session_state.get("_shap_bundle")
    if not bundle:
        return
    if bundle.get("summary_fig") is not None:
        st.plotly_chart(bundle["summary_fig"], use_container_width=True)
    for fname, dfig in bundle.get("dependence") or []:
        st.plotly_chart(dfig, use_container_width=True)
    ai = st.session_state.get("_shap_ai")
    if ai:
        st.info(ai)
