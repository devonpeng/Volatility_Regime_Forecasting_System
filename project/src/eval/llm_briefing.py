from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from project.src.utils.cache import RunCache

PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "prompts" / "eval_llm_prompts.md"


def _to_shap_3d(shap_values: Any) -> np.ndarray:
    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim == 3:
        return arr
    if isinstance(shap_values, list):
        stacked = np.stack([np.asarray(part) for part in shap_values], axis=-1)
        if stacked.ndim == 3:
            return stacked
    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def _extract_lean_shap_context(
    shap_values: Any,
    predicted_class_idx: np.ndarray,
    feature_names: list[str],
    *,
    sample_index: int = -1,
    class_names: list[str] | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    shap_3d = _to_shap_3d(shap_values)
    n_samples, n_features, n_classes = shap_3d.shape
    pred_idx = np.asarray(predicted_class_idx, dtype=int).reshape(-1)
    if pred_idx.shape[0] != n_samples:
        raise ValueError(
            f"predicted_class_idx length {pred_idx.shape[0]} != SHAP samples {n_samples}"
        )
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names length {len(feature_names)} != SHAP features {n_features}"
        )

    pred_idx = np.clip(pred_idx, 0, n_classes - 1)
    class_axis_idx = pred_idx[:, np.newaxis, np.newaxis]
    class_axis_idx = np.repeat(class_axis_idx, n_features, axis=1)
    per_sample_shap = np.take_along_axis(shap_3d, class_axis_idx, axis=2).squeeze(axis=2)

    resolved_sample_idx = sample_index if sample_index >= 0 else n_samples + sample_index
    resolved_sample_idx = int(np.clip(resolved_sample_idx, 0, n_samples - 1))
    sample_contrib = per_sample_shap[resolved_sample_idx]
    sample_pred_class = int(pred_idx[resolved_sample_idx])

    top_idx = np.argsort(np.abs(sample_contrib))[::-1][:top_k]
    top_features = {feature_names[i]: float(sample_contrib[i]) for i in top_idx}
    class_label = (
        class_names[sample_pred_class]
        if class_names is not None and sample_pred_class < len(class_names)
        else str(sample_pred_class)
    )
    return {"predicted_class": class_label, "top_features": top_features}


def _load_prompt_templates(path: Path = PROMPT_TEMPLATE_PATH) -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")
    content = path.read_text(encoding="utf-8")
    marker_gen = "## Generator Prompt"
    marker_rev = "## Reviewer Prompt"
    if marker_gen not in content or marker_rev not in content:
        raise ValueError("Prompt template markdown missing required section headers")
    gen_block = content.split(marker_gen, maxsplit=1)[1].split(marker_rev, maxsplit=1)[0]
    rev_block = content.split(marker_rev, maxsplit=1)[1]
    return gen_block.strip(), rev_block.strip()


def _call_google_llm(prompt: str) -> str:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model_name = os.getenv("LLM_MODEL", "").strip()
    if not api_key or not model_name:
        raise RuntimeError("LLM_API_KEY or LLM_MODEL is missing in environment")

    import google.generativeai as genai  # type: ignore[import-untyped]

    configure = getattr(genai, "configure", None)
    model_cls = getattr(genai, "GenerativeModel", None)
    if configure is None or model_cls is None:
        raise RuntimeError("google.generativeai SDK missing configure/GenerativeModel")

    configure(api_key=api_key)
    model = model_cls(model_name=model_name)
    response = model.generate_content(prompt)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Empty response from Google LLM")
    return str(text).strip()


def _build_mock_brief(lean_shap_context: dict[str, Any]) -> str:
    predicted_class = str(lean_shap_context.get("predicted_class", "unknown"))
    top_features: dict[str, float] = lean_shap_context.get("top_features", {})
    if not top_features:
        return (
            "[DR] No reliable SHAP feature contribution was available. "
            "[FI] The model still indicates a transition toward "
            f"{predicted_class}, but drivers cannot be ranked deterministically."
        )
    lead_feature, lead_value = next(iter(top_features.items()))
    direction = "upward" if lead_value >= 0 else "downward"
    return (
        f"[DR] {lead_feature} has SHAP contribution {lead_value:.4f} for predicted class "
        f"{predicted_class}. "
        f"[FI] The {direction} pressure from {lead_feature} appears to be the dominant "
        "driver in the current volatility regime transition."
    )


def _build_linear_proxy_context(
    X_scaled: np.ndarray,
    feature_names: list[str],
    pred_idx: np.ndarray,
    *,
    class_names: list[str] | None = None,
    model: Any | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    if model is not None and hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            coef = coef[np.newaxis, :]
        class_count = coef.shape[0]
        aligned_pred = np.clip(pred_idx, 0, class_count - 1)
        sample_contrib = X_scaled * coef[aligned_pred]
        return _extract_lean_shap_context(
            sample_contrib[..., np.newaxis],
            np.zeros_like(aligned_pred),
            feature_names,
            class_names=class_names,
            top_k=top_k,
        )
    dummy = np.zeros((X_scaled.shape[0], len(feature_names), 1), dtype=float)
    return _extract_lean_shap_context(
        dummy,
        np.zeros(X_scaled.shape[0], dtype=int),
        feature_names,
        class_names=class_names,
        top_k=top_k,
    )


def _parse_review_response(review_text: str) -> dict[str, Any]:
    cleaned = review_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        lines = cleaned.splitlines()
        if lines and lines[0].lower().startswith("json"):
            lines = lines[1:]
        cleaned = "\n".join(lines).strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    lowered = cleaned.lower()
    status = "approved" if "approved" in lowered or "pass" in lowered else "rejected"
    return {"status": status, "final_brief": cleaned}


def _generate_and_review_brief(lean_shap_context: dict[str, Any]) -> dict[str, Any]:
    generator_template, reviewer_template = _load_prompt_templates()
    context_json = json.dumps(lean_shap_context, ensure_ascii=False, indent=2)

    generator_prompt = (
        f"{generator_template}\n\n"
        "Lean SHAP Context (JSON):\n"
        f"{context_json}\n"
    )
    generator_output = _call_google_llm(generator_prompt)

    reviewer_prompt = (
        f"{reviewer_template}\n\n"
        "Lean SHAP Context (JSON):\n"
        f"{context_json}\n\n"
        "Candidate Brief (Node0 Output):\n"
        f"{generator_output}\n"
    )
    reviewer_output = _call_google_llm(reviewer_prompt)
    review_payload = _parse_review_response(reviewer_output)
    status = str(review_payload.get("status", "rejected")).lower()
    final_brief = str(review_payload.get("final_brief", generator_output))

    return {
        "generator_output": generator_output,
        "review_status": status,
        "review_raw_output": reviewer_output,
        "final_brief": final_brief,
        "source": "llm",
    }


def build_llm_briefing(run_dir: Path) -> dict[str, Any]:
    try:
        import shap
    except Exception as exc:
        return {
            "source": "mock_fallback",
            "error": f"shap import failed: {exc}",
            "lean_shap_context": {"predicted_class": "unknown", "top_features": {}},
            "final_brief": _build_mock_brief(
                {"predicted_class": "unknown", "top_features": {}}
            ),
        }

    bundle = RunCache.load_model_bundle(run_dir)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = list(bundle["feature_names"])

    X_df = pd.read_csv(run_dir / "data" / "X.csv")
    X_df = X_df[feature_names]
    X_scaled = scaler.transform(X_df.to_numpy())
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    model_name = str(bundle["model_name"])
    proba = model.predict_proba(X_scaled_df if model_name == "lgbm" else X_scaled)
    pred_idx = np.argmax(proba, axis=1).astype(int)
    class_names = ["contracting", "expanding"] if proba.shape[1] == 2 else None

    try:
        if model_name in {"xgb", "lgbm"}:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(
                X_scaled_df if model_name == "lgbm" else X_scaled
            )
        else:
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
        lean_context = _extract_lean_shap_context(
            shap_values,
            pred_idx,
            feature_names,
            class_names=class_names,
            top_k=3,
        )
    except Exception:
        lean_context = _build_linear_proxy_context(
            X_scaled,
            feature_names,
            pred_idx,
            class_names=class_names,
            model=model,
            top_k=3,
        )

    try:
        llm_result = _generate_and_review_brief(lean_context)
        llm_result["lean_shap_context"] = lean_context
        if llm_result.get("review_status") != "approved":
            llm_result["final_brief"] = _build_mock_brief(lean_context)
            llm_result["source"] = "mock_fallback"
            llm_result["error"] = "review rejected Node0 output"
        return llm_result
    except Exception as exc:
        return {
            "source": "mock_fallback",
            "error": str(exc),
            "lean_shap_context": lean_context,
            "final_brief": _build_mock_brief(lean_context),
        }
