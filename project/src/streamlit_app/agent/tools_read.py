"""Read-only LangChain tools for the RV Lab tutor."""
from __future__ import annotations

import inspect
import json
import logging
import sys
import threading
from typing import Any

import streamlit as st
from langchain_core.tools import tool
from streamlit.runtime.scriptrunner import get_script_run_ctx

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
    GS_END,
    GS_HORIZON,
    GS_START,
    GS_SYMBOL,
    SS_BEST_MODEL_NAME,
    SS_BEST_PARAMS,
    SS_COMPARED_EXPERIMENTS,
    SS_CUSTOM_EXPERIMENTS,
    SS_CV,
    SS_CV_GAP,
    SS_CV_N_SPLITS,
    SS_CV_TEST_SIZE,
    SS_CV_TRAIN_WINDOW,
    SS_DATA_SOURCE,
    SS_DATE_END,
    SS_DATE_START,
    SS_ENRICHED_DF,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_MODEL_RESULTS,
    SS_N_TRIALS,
    SS_OOF_METRICS,
    SS_RAW_DF,
    SS_SELECTED_FEATURES,
    SS_SELECTED_MODELS,
    SS_SYMBOL,
    SS_TRAIN_CURRENT,
    SS_TRAIN_N_TRIALS,
    SS_TRAIN_QUEUE,
    SS_TRAIN_RATIO,
    SS_TRAIN_RESULTS,
    SS_TRAIN_STATUS,
    SS_TRAIN_TRIAL_I,
    SS_WORKSPACE_CONFIG,
    SS_X,
    SS_X_TEST,
    SS_X_TRAIN,
    SS_Y,
)

_logger = logging.getLogger(__name__)


def _json(data: Any) -> str:
    return json.dumps(data, default=str, ensure_ascii=False)


def _log_snapshot_to_stderr(snapshot: dict[str, Any], *, page_name_arg: str) -> None:
    """Temporary debug: dump full page snapshot JSON to the Streamlit server stderr."""
    # payload = {
    #     "tool": "get_current_page_snapshot",
    #     "page_name_arg": page_name_arg,
    #     "thread_name": threading.current_thread().name,
    #     "has_script_run_ctx": get_script_run_ctx(suppress_warning=True) is not None,
    #     "n_session_keys": len(st.session_state.keys())
    #     if hasattr(st.session_state, "keys")
    #     else None,
    #     "session_current_page": st.session_state.get("_current_page"),
    #     "snapshot": snapshot,
    # }
    pass
    # text = _json(payload)
    # print(
    #     f"[DEBUG get_current_page_snapshot] {text}",
    #     file=sys.stderr,
    #     flush=True,
    # )
    # _logger.warning("get_current_page_snapshot debug dump: %s", text)


def _df_shape(obj: Any) -> list[int] | None:
    if obj is None:
        return None
    try:
        return [int(obj.shape[0]), int(obj.shape[1])]
    except Exception:
        try:
            return [int(len(obj))]
        except Exception:
            return None


def _class_balance(y: Any) -> dict[str, Any] | None:
    if y is None:
        return None
    try:
        import pandas as pd

        s = pd.Series(y)
        counts = s.value_counts(dropna=False).to_dict()
        return {str(k): int(v) for k, v in counts.items()}
    except Exception:
        return None


def _snapshot_etl() -> dict[str, Any]:
    raw = st.session_state.get(SS_RAW_DF)
    cols = list(raw.columns)[:30] if raw is not None else []
    date_min = date_max = None
    if raw is not None and "Date" in getattr(raw, "columns", []):
        try:
            dates = raw["Date"]
            date_min, date_max = str(dates.min())[:10], str(dates.max())[:10]
        except Exception:
            pass
    return {
        "station": "etl",
        "applied": {
            "symbol": st.session_state.get(SS_SYMBOL),
            "horizon": st.session_state.get(SS_HORIZON),
            "date_start": st.session_state.get(SS_DATE_START),
            "date_end": st.session_state.get(SS_DATE_END),
            "data_source": st.session_state.get(SS_DATA_SOURCE),
        },
        "global_settings_draft": {
            "symbol": st.session_state.get(GS_SYMBOL),
            "horizon": st.session_state.get(GS_HORIZON),
            "start": str(st.session_state.get(GS_START, "")),
            "end": str(st.session_state.get(GS_END, "")),
        },
        "raw_df": {
            "loaded": raw is not None,
            "shape": _df_shape(raw),
            "date_min": date_min,
            "date_max": date_max,
            "columns_sample": cols,
            "n_columns": int(len(raw.columns)) if raw is not None else 0,
        },
        "plots": ["ohlcv_preview"] if raw is not None else [],
    }


def _snapshot_feature_eng() -> dict[str, Any]:
    X = st.session_state.get(SS_X)
    y = st.session_state.get(SS_Y)
    selected = list(st.session_state.get(SS_SELECTED_FEATURES) or [])
    return {
        "station": "feature_eng",
        "selected_features": selected[:60],
        "n_selected_features": len(selected),
        "label_method": st.session_state.get(SS_LABEL_METHOD),
        "fs_version": st.session_state.get("_fs_version"),
        "enriched_present": st.session_state.get(SS_ENRICHED_DF) is not None,
        "X": {
            "shape": _df_shape(X),
            "columns": list(X.columns)[:60] if X is not None else [],
        },
        "y": {
            "shape": _df_shape(y),
            "class_balance": _class_balance(y),
        },
        "plots": ["feature_distribution", "dataset_summary"],
    }


def _model_result_summaries(results: Any) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for r in results or []:
        if isinstance(r, dict):
            summaries.append(
                {
                    "name": r.get("name"),
                    "best_auc": r.get("best_auc"),
                    "best_params": r.get("best_params") or {},
                }
            )
            continue
        summaries.append(
            {
                "name": getattr(r, "name", None),
                "best_auc": getattr(r, "best_auc", None),
                "best_params": getattr(r, "best_params", {}) or {},
            }
        )
    return summaries


def _live_or_ss(widget_key: str, ss_key: str, default: Any = None) -> Any:
    """Prefer live Station widget value when present; else logical SS_* key."""
    if widget_key in st.session_state:
        return st.session_state.get(widget_key)
    return st.session_state.get(ss_key, default)


def _snapshot_model_design() -> dict[str, Any]:
    results = st.session_state.get(SS_MODEL_RESULTS)
    summaries = _model_result_summaries(results)
    partial = _model_result_summaries(st.session_state.get(SS_TRAIN_RESULTS))
    cfg = dict(st.session_state.get(SS_WORKSPACE_CONFIG) or {})
    checkpointed_results = list(cfg.get("model_results") or [])
    best_model = st.session_state.get(SS_BEST_MODEL_NAME) or None
    if not best_model and results is None:
        best_model = cfg.get("best_model_name") or None

    n_splits = _live_or_ss("md_cv_n_splits", SS_CV_N_SPLITS, 12)
    train_window = _live_or_ss("md_cv_train_window", SS_CV_TRAIN_WINDOW, 252)
    test_size = _live_or_ss("md_cv_test_size", SS_CV_TEST_SIZE, 21)
    gap = _live_or_ss("md_cv_gap", SS_CV_GAP, 5)
    selected_models = _live_or_ss(
        "md_selected_models", SS_SELECTED_MODELS, []
    )
    n_trials = _live_or_ss("md_n_trials", SS_N_TRIALS, 30)
    train_ratio = _live_or_ss("md_train_ratio", SS_TRAIN_RATIO, 0.8)

    cv_obj = st.session_state.get(SS_CV)
    return {
        "station": "model_design",
        "train_ratio": train_ratio,
        "cv": {
            "object_ready": cv_obj is not None,
            "n_splits": int(n_splits) if n_splits is not None else None,
            "train_window": int(train_window) if train_window is not None else None,
            "test_size": int(test_size) if test_size is not None else None,
            "gap": int(gap) if gap is not None else None,
            "committed_ss": {
                "n_splits": st.session_state.get(SS_CV_N_SPLITS),
                "train_window": st.session_state.get(SS_CV_TRAIN_WINDOW),
                "test_size": st.session_state.get(SS_CV_TEST_SIZE),
                "gap": st.session_state.get(SS_CV_GAP),
            },
            "active_object": {
                "n_splits": getattr(cv_obj, "n_splits", None),
                "train_window": getattr(cv_obj, "train_window_size", None),
                "test_size": getattr(cv_obj, "test_size", None),
                "gap": getattr(cv_obj, "gap_size", None),
            }
            if cv_obj is not None
            else None,
        },
        "selected_models": list(selected_models or []),
        "n_trials": int(n_trials) if n_trials is not None else None,
        "split": {
            "X_train_shape": _df_shape(st.session_state.get(SS_X_TRAIN)),
            "X_test_shape": _df_shape(st.session_state.get(SS_X_TEST)),
        },
        "training": {
            "status": st.session_state.get(SS_TRAIN_STATUS),
            "current_model": st.session_state.get(SS_TRAIN_CURRENT) or None,
            "trial_i": st.session_state.get(SS_TRAIN_TRIAL_I),
            "n_trials": st.session_state.get(SS_TRAIN_N_TRIALS),
            "queue": list(st.session_state.get(SS_TRAIN_QUEUE) or []),
            "partial_results": partial,
        },
        "train_status": st.session_state.get(SS_TRAIN_STATUS),
        "model_results": summaries,
        "has_model_results": results is not None,
        "best_model": best_model,
        "best_params": st.session_state.get(SS_BEST_PARAMS) or {},
        "checkpointed": {
            "best_model_name": cfg.get("best_model_name") or None,
            "model_results": checkpointed_results,
            "cv_config": cfg.get("cv_config"),
            "selected_models": list(cfg.get("selected_models") or []),
            "n_trials": cfg.get("n_trials"),
            "train_ratio": cfg.get("train_ratio"),
            "note": (
                "Use checkpointed.* only when live model_results is empty "
                "(e.g. after reload). Prefer live fields when present."
            ),
        },
        "plots": ["cv_splits", "train_progress"] if cv_obj is not None else [],
    }


def _snapshot_implementation() -> dict[str, Any]:
    oof = st.session_state.get(SS_OOF_METRICS)
    report = st.session_state.get("_llm_report") or st.session_state.get("_rule_report")
    report_text = None
    if isinstance(report, str):
        report_text = report[:1500]
    elif isinstance(report, dict):
        report_text = str(report)[:1500]
    cfg = dict(st.session_state.get(SS_WORKSPACE_CONFIG) or {})
    results = st.session_state.get(SS_MODEL_RESULTS)
    summaries = _model_result_summaries(results)
    best_model = st.session_state.get(SS_BEST_MODEL_NAME) or None
    best_params = st.session_state.get(SS_BEST_PARAMS) or {}
    if results is None and not best_model:
        best_model = cfg.get("best_model_name") or None
    if results is None and not best_params:
        best_params = cfg.get("best_params") or {}
    if results is None and not summaries:
        summaries = _model_result_summaries(cfg.get("model_results") or [])
    if oof is None and cfg.get("oof_metrics"):
        oof = dict(cfg.get("oof_metrics") or {})
    return {
        "station": "implementation",
        "best_model": best_model,
        "best_params": best_params,
        "model_results": summaries,
        "oof_metrics": oof,
        "restored_from_workspace": bool(st.session_state.get("_model_results_restored")),
        "checkpointed": {
            "best_model_name": cfg.get("best_model_name") or None,
            "model_results": list(cfg.get("model_results") or []),
            "oof_metrics": cfg.get("oof_metrics"),
        },
        "artifacts": {
            "feature_importances": st.session_state.get("_feature_importances") is not None,
            "shap_bundle": st.session_state.get("_shap_bundle") is not None,
            "llm_or_rule_report": report is not None,
        },
        "report_excerpt": report_text,
        "plots": [
            "model_comparison",
            "optuna_history",
            "roc",
            "pred_vs_actual",
            "feature_importance",
            "shap",
        ],
    }


def _snapshot_comparison() -> dict[str, Any]:
    custom = list(st.session_state.get(SS_CUSTOM_EXPERIMENTS) or [])
    compared = list(st.session_state.get(SS_COMPARED_EXPERIMENTS) or [])
    return {
        "station": "comparison",
        "quick_browse": {
            "qb_sym": st.session_state.get("qb_sym"),
            "qb_hrz": st.session_state.get("qb_hrz"),
            "qb_sel": st.session_state.get("qb_sel"),
            "qb_model": st.session_state.get("qb_model"),
            "qb_dates": st.session_state.get("qb_dates"),
            "qb_topk": st.session_state.get("qb_topk"),
            "qb_compare_ids": st.session_state.get("qb_compare_ids"),
        },
        "custom_experiment": {
            "ce_date_preset": st.session_state.get("ce_date_preset"),
            "ce_fs": st.session_state.get("ce_fs"),
            "ce_topk": st.session_state.get("ce_topk"),
            "ce_nested": st.session_state.get("ce_nested"),
            "ce_models": st.session_state.get("ce_models"),
            "ce_trials": st.session_state.get("ce_trials"),
            "ce_ns": st.session_state.get("ce_ns"),
            "n_custom_runs": len(custom),
            "latest_custom_id": (custom[0].get("experiment_id") if custom else None),
        },
        "compared_count": len(compared),
        "plots": ["browse_table", "bubble", "radar", "deep_dive"],
    }


def build_current_page_snapshot(page_name: str = "") -> dict[str, Any]:
    # Soft-restore missing pipeline objects from the last workspace checkpoint so
    # the tutor can still describe trained models after a tab reload / re-split.
    try:
        from project.src.streamlit_app.workspace import ensure_pipeline_rehydrated

        ensure_pipeline_rehydrated()
    except Exception:
        pass

    page = (page_name or st.session_state.get("_current_page") or "").strip()
    # Normalize station ids used in app navigation
    alias = {
        "etl": "etl",
        "feature_eng": "feature_eng",
        "model_design": "model_design",
        "implementation": "implementation",
        "comparison": "comparison",
        "1": "etl",
        "2": "feature_eng",
        "3": "model_design",
        "4": "implementation",
        "5": "comparison",
    }
    key = alias.get(page, page)
    builders = {
        "etl": _snapshot_etl,
        "feature_eng": _snapshot_feature_eng,
        "model_design": _snapshot_model_design,
        "implementation": _snapshot_implementation,
        "comparison": _snapshot_comparison,
    }
    body = builders.get(key, _snapshot_etl)()
    cfg = dict(st.session_state.get(SS_WORKSPACE_CONFIG) or {})
    has_results = st.session_state.get(SS_MODEL_RESULTS) is not None or bool(
        cfg.get("model_results")
    )
    return {
        "page": key or page,
        "globals": {
            "symbol": st.session_state.get(SS_SYMBOL),
            "horizon": st.session_state.get(SS_HORIZON),
            "date_start": st.session_state.get(SS_DATE_START),
            "date_end": st.session_state.get(SS_DATE_END),
            "data_source": st.session_state.get(SS_DATA_SOURCE),
            "X_shape": _df_shape(st.session_state.get(SS_X)),
            "y_shape": _df_shape(st.session_state.get(SS_Y)),
            "has_cv": st.session_state.get(SS_CV) is not None,
            "has_model_results": has_results,
            "best_model": st.session_state.get(SS_BEST_MODEL_NAME)
            or cfg.get("best_model_name")
            or None,
            "selected_models": list(
                _live_or_ss("md_selected_models", SS_SELECTED_MODELS, []) or []
            ),
            "n_trials": _live_or_ss("md_n_trials", SS_N_TRIALS, None),
            "train_ratio": _live_or_ss("md_train_ratio", SS_TRAIN_RATIO, None),
        },
        "content": body,
    }


@tool
def get_current_page_snapshot(page_name: str = "") -> str:
    """Structured snapshot of the current Station UI: config, data shapes, metrics, reports, plot inventory.

    Prefer this over get_page_context when explaining what the student currently sees.
    Args:
        page_name: Optional station id (etl, feature_eng, model_design, implementation, comparison).
                   Defaults to the page where the tutor is open.
    """
    snapshot = build_current_page_snapshot(page_name)
    _log_snapshot_to_stderr(snapshot, page_name_arg=page_name)
    return _json(snapshot)


@tool
def get_page_context(page_name: str = "") -> str:
    """Return a thin snapshot of the student's current lab session / page context."""
    raw = st.session_state.get(SS_RAW_DF)
    cfg = dict(st.session_state.get(SS_WORKSPACE_CONFIG) or {})
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
        "cv_n_splits": _live_or_ss("md_cv_n_splits", SS_CV_N_SPLITS),
        "cv_train_window": _live_or_ss("md_cv_train_window", SS_CV_TRAIN_WINDOW),
        "cv_test_size": _live_or_ss("md_cv_test_size", SS_CV_TEST_SIZE),
        "cv_gap": _live_or_ss("md_cv_gap", SS_CV_GAP),
        "has_cv": st.session_state.get(SS_CV) is not None,
        "train_ratio": _live_or_ss("md_train_ratio", SS_TRAIN_RATIO),
        "n_trials": _live_or_ss("md_n_trials", SS_N_TRIALS),
        "selected_models": list(
            _live_or_ss("md_selected_models", SS_SELECTED_MODELS, []) or []
        ),
        "best_model": st.session_state.get(SS_BEST_MODEL_NAME)
        or cfg.get("best_model_name")
        or None,
        "has_model_results": st.session_state.get(SS_MODEL_RESULTS) is not None
        or bool(cfg.get("model_results")),
        "oof_metrics": st.session_state.get(SS_OOF_METRICS) or cfg.get("oof_metrics"),
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
    get_current_page_snapshot,
    get_page_context,
    ls_features,
    get_feature,
    list_models,
    list_comparison_filter_options,
    get_oof_metrics_summary,
]
