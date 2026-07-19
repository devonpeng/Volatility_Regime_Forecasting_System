"""LLM-powered helpers via Google Gemini."""
from __future__ import annotations

import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def resolve_llm_credentials(
    api_key: str | None = None,
    model_name: str | None = None,
) -> tuple[str, str]:
    """Resolve API key / model: explicit args → env."""
    load_dotenv()
    key = (api_key or "").strip() or os.getenv("LLM_API_KEY", "").strip()
    model = (model_name or "").strip() or os.getenv("LLM_MODEL", "").strip()
    if not key or not model:
        raise RuntimeError("LLM_API_KEY or LLM_MODEL is missing")
    return key, model


def _call_google_llm(
    prompt: str,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    key, model = resolve_llm_credentials(api_key, model_name)

    import google.generativeai as genai

    genai.configure(api_key=key)
    gm = genai.GenerativeModel(model_name=model)
    response = gm.generate_content(prompt)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Empty response from Google LLM")
    return str(text).strip()


_FEATURE_PROMPT_TEMPLATE = """\
You are a Python feature-engineering assistant for a quantitative finance pipeline.

Given:
- A pandas DataFrame named `df` with the following columns: {columns}
- A 5-row sample:
{sample}

Task: {description}

Write a single Python function with signature:

```python
def compute(df: pd.DataFrame) -> pd.Series:
    ...
```

Rules:
1. Return a `pd.Series` with the same index as `df`.
2. Use only `pandas` and `numpy`. No external APIs or file I/O.
3. The returned Series name should be a short snake_case feature name.
4. Keep the function self-contained; no helper functions.

Return ONLY the function code inside a ```python ``` block.
"""


def generate_feature(
    description: str,
    df_sample: pd.DataFrame,
    available_columns: list[str],
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> Tuple[str, str]:
    sample_str = df_sample.head(5).to_string()
    prompt = _FEATURE_PROMPT_TEMPLATE.format(
        columns=", ".join(available_columns),
        sample=sample_str,
        description=description,
    )
    response = _call_google_llm(prompt, api_key=api_key, model_name=model_name)

    code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else response.strip()

    name_match = re.search(
        r'["\'](\w+)["\']', code.split("return")[-1] if "return" in code else ""
    )
    feature_name = name_match.group(1) if name_match else "llm_feature"
    return code, feature_name


def safe_exec_feature(code: str, df: pd.DataFrame) -> pd.Series:
    namespace: dict = {
        "__builtins__": {},
        "pd": pd,
        "np": np,
        "df": df,
    }
    exec(code, namespace)  # noqa: S102
    compute_fn = namespace.get("compute")
    if compute_fn is None:
        raise ValueError("Generated code does not define a `compute(df)` function.")
    result = compute_fn(df)
    if not isinstance(result, pd.Series):
        raise TypeError(f"compute() must return pd.Series, got {type(result).__name__}")
    return result


def generate_llm_brief(
    model_name: str,
    best_params: dict,
    feature_importances: dict[str, float],
    oof_metrics: dict[str, float],
    *,
    api_key: str | None = None,
    llm_model: str | None = None,
) -> str:
    context = (
        f"Model: {model_name}\n"
        f"Best params: {best_params}\n"
        f"OOF Metrics: {oof_metrics}\n"
        f"Feature importances: {feature_importances}\n"
    )
    prompt = (
        "You are a quantitative analyst. Given the following model evaluation results, "
        "write a concise diagnostic report (3-5 sentences) covering:\n"
        "1. Model performance summary\n"
        "2. Key drivers (top features)\n"
        "3. Any concerns or suggestions\n\n"
        f"{context}"
    )
    return _call_google_llm(prompt, api_key=api_key, model_name=llm_model)


def analyze_results(
    context: dict,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    """Student-friendly analysis of evaluation / comparison results."""
    prompt = f"""
You are a quant finance teaching assistant.
Given these model evaluation results:
{context}

Write a student-friendly analysis covering:
1. Performance summary (is this good? compare to typical finance ML benchmarks)
2. What the top features tell us about volatility drivers
3. Potential issues (overfitting risk, data snooping, regime changes)
4. What the student should try next

Keep it under 400 words. Use clear prose, not bullet dumps.
"""
    return _call_google_llm(prompt, api_key=api_key, model_name=model_name)


def chat_tutor_reply(
    messages: list[dict],
    page_context: str,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    """Legacy single-shot tutor reply. Prefer LangGraph agent via agent.runtime.invoke_tutor."""
    history = "\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in messages[-12:]
    )
    prompt = f"""
You are an AI tutor for a realized-volatility prediction teaching lab.
Current page context: {page_context}

Conversation so far:
{history}

Reply helpfully for a student learning quant ML. Be concise (under 200 words).
"""
    return _call_google_llm(prompt, api_key=api_key, model_name=model_name)



def interpret_shap(
    shap_summary: dict,
    *,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    prompt = f"""
You are a quant finance teaching assistant explaining SHAP results to a student.
SHAP summary data:
{shap_summary}

Write 3-5 sentences explaining:
- which features push probability of high-vol up vs down
- any non-linear / threshold behaviour suggested by the top dependence candidates
- one follow-up experiment idea
"""
    return _call_google_llm(prompt, api_key=api_key, model_name=model_name)
