"""Read-only LangChain tools for the RV Lab tutor."""
from __future__ import annotations

import inspect
import json
from typing import Any

import streamlit as st
from langchain_core.tools import tool

from project.src.features.core import FEATURE_REGISTRY
from project.src.models.registry import MODEL_REGISTRY
from project.src.streamlit_app.agent.feature_docs import FEATURE_DOCS, get_feature_doc
from project.src.streamlit_app.precomputed import (
    TEACHING_HORIZONS,
    TEACHING_SYMBOLS,
    feature_column_names,
    grid_results_path,
)
from project.src.streamlit_app.session_manager import (
    SS_BEST_MODEL_NAME,
    SS_COMPARED_EXPERIMENTS,
    SS_CUSTOM_EXPERIMENTS,
    SS_CV_N_SPLITS,
    SS_DATA_SOURCE,
    SS_DATE_END,
    SS_DATE_START,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_N_TRIALS,
    SS_OOF_METRICS,
    SS_RAW_DF,
    SS_SELECTED_FEATURES,
    SS_SELECTED_MODELS,
    SS_SYMBOL,
    SS_TRAIN_RATIO,
)


def _json(data: Any) -> str:
    return json.dumps(data, default=str, ensure_ascii=False)


@tool
def get_page_context(page_name: str = "") -> str:
    """Return a snapshot of the student's current lab session / page context."""
    raw = st.session_state.get(SS_RAW_DF)
    ctx = {
        "page": page_name or st.session_state.get("_current_page", ""),
        "symbol": st.session_state.get(SS_SYMBOL),
        "horizon": st.session_state.get(SS_HORIZON),
        "date_start": st.session_state.get(SS_DATE_START),
        "date_end": st.session_state.get(SS_DATE_END),
        "data_source": st.session_state.get(SS_DATA_SOURCE),
        "rows": int(len(raw)) if raw is not None else 0,
        "selected_features": list(st.session_state.get(SS_SELECTED_FEATURES) or [])[:40],
        "n_selected_features": len(st.session_state.get(SS_SELECTED_FEATURES) or []),
        "label_method": st.session_state.get(SS_LABEL_METHOD),
        "cv_n_splits": st.session_state.get(SS_CV_N_SPLITS),
        "train_ratio": st.session_state.get(SS_TRAIN_RATIO),
        "n_trials": st.session_state.get(SS_N_TRIALS),
        "selected_models": list(st.session_state.get(SS_SELECTED_MODELS) or []),
        "best_model": st.session_state.get(SS_BEST_MODEL_NAME),
        "oof_metrics": st.session_state.get(SS_OOF_METRICS),
        "custom_experiments": len(st.session_state.get(SS_CUSTOM_EXPERIMENTS) or []),
        "compared": len(st.session_state.get(SS_COMPARED_EXPERIMENTS) or []),
        "teaching_symbols": list(TEACHING_SYMBOLS),
        "teaching_horizons": list(TEACHING_HORIZONS),
    }
    return _json(ctx)


@tool
def ls_features() -> str:
    """List engineered features with short descriptions (README meanings).

    Returns each available registry feature name plus a brief role description,
    and notes which features are currently selected in the lab session.
    """
    raw = st.session_state.get(SS_RAW_DF)
    names = feature_column_names()
    if raw is not None:
        present = [n for n in names if n in raw.columns]
    else:
        present = names
    selected = set(st.session_state.get(SS_SELECTED_FEATURES) or [])
    features = [
        {
            "name": name,
            "description": FEATURE_DOCS.get(name, {}).get("role", ""),
            "selected": name in selected,
        }
        for name in present
    ]
    return _json(
        {
            "count": len(features),
            "features": features,
            "currently_selected": list(st.session_state.get(SS_SELECTED_FEATURES) or []),
        }
    )


@tool
def get_feature(name: str) -> str:
    """Get calculation, meaning, and source code for one engineered feature.

    Args:
        name: Feature column name (e.g. 'gk_rv', 'vix_term_structure').
    """
    doc = get_feature_doc(name)
    if doc is None and name not in FEATURE_REGISTRY:
        return _json(
            {
                "error": f"Unknown feature: {name}",
                "hint": "Call ls_features to see available names.",
            }
        )

    source = None
    if name in FEATURE_REGISTRY:
        try:
            source = inspect.getsource(FEATURE_REGISTRY[name].fn)
        except (OSError, TypeError) as exc:
            source = f"<unavailable: {exc}>"

    raw = st.session_state.get(SS_RAW_DF)
    selected = set(st.session_state.get(SS_SELECTED_FEATURES) or [])
    payload: dict[str, Any] = {
        "name": name,
        "role": doc["role"] if doc else "",
        "transform": doc["transform"] if doc else "",
        "formula": doc["formula"] if doc else "",
        "source": source,
        "in_registry": name in FEATURE_REGISTRY,
        "present_in_data": bool(raw is not None and name in raw.columns),
        "currently_selected": name in selected,
    }
    if doc is None:
        payload["warning"] = "No README doc entry; source may still be available."
    return _json(payload)


@tool
def list_models() -> str:
    """List train-able model names in MODEL_REGISTRY."""
    return _json({"models": list(MODEL_REGISTRY.keys())})


@tool
def list_comparison_filter_options() -> str:
    """List Quick Browse filter option values from the offline grid (when present)."""
    path = grid_results_path()
    if not path.is_file():
        return _json({"error": "grid_results.parquet missing", "options": {}})
    import pandas as pd

    df = pd.read_parquet(path)
    cols = ["symbol", "horizon", "feature_selection", "model", "date_range", "top_k"]
    options: dict[str, list[str]] = {}
    for col in cols:
        if col in df.columns:
            options[col] = sorted({str(v) for v in df[col].dropna().unique()})
    id_col = "experiment_id" if "experiment_id" in df.columns else None
    sample_ids = (
        df[id_col].astype(str).head(20).tolist() if id_col else []
    )
    return _json({"options": options, "sample_experiment_ids": sample_ids})


@tool
def get_oof_metrics_summary() -> str:
    """Return out-of-fold metrics from Station 4 if available."""
    return _json(
        {
            "best_model": st.session_state.get(SS_BEST_MODEL_NAME),
            "oof_metrics": st.session_state.get(SS_OOF_METRICS),
        }
    )


READ_TOOLS = [
    get_page_context,
    ls_features,
    get_feature,
    list_models,
    list_comparison_filter_options,
    get_oof_metrics_summary,
]
