"""LLM-powered feature generation via Google Gemini."""
from __future__ import annotations

import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv


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


def _call_google_llm(prompt: str) -> str:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "").strip()
    if not api_key or not model_name:
        raise RuntimeError("LLM_API_KEY or LLM_MODEL is missing in .env")

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(prompt)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Empty response from Google LLM")
    return str(text).strip()


def generate_feature(
    description: str,
    df_sample: pd.DataFrame,
    available_columns: list[str],
) -> Tuple[str, str]:
    """Ask Gemini to write a feature function.

    Returns (python_code_str, feature_name).
    """
    sample_str = df_sample.head(5).to_string()
    prompt = _FEATURE_PROMPT_TEMPLATE.format(
        columns=", ".join(available_columns),
        sample=sample_str,
        description=description,
    )
    response = _call_google_llm(prompt)

    # Extract code block
    code_match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = response.strip()

    # Try to extract feature name from the Series name in the code
    name_match = re.search(r'["\'](\w+)["\']', code.split("return")[-1] if "return" in code else "")
    feature_name = name_match.group(1) if name_match else "llm_feature"

    return code, feature_name


def safe_exec_feature(code: str, df: pd.DataFrame) -> pd.Series:
    """Execute generated feature code in a restricted namespace."""
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
) -> str:
    """Generate a diagnostic report via the dual-agent LLM flow."""
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
    return _call_google_llm(prompt)
