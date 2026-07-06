"""Page 4: Evaluation & LLM Insight."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from project.src.models.evaluator import compute_oof_metrics, evaluate_oof
from project.src.models.registry import build_model
from project.src.streamlit_app.llm_helper import generate_llm_brief
from project.src.streamlit_app.plots import (
    fig_feature_importance,
    fig_model_comparison,
    fig_optuna_history,
    fig_predictions,
    fig_roc_curve,
)
from project.src.streamlit_app.session_manager import (
    SS_BEST_MODEL_NAME,
    SS_BEST_PARAMS,
    SS_CV,
    SS_FEATURE_SELECTION,
    SS_META_DF,
    SS_MODEL_RESULTS,
    SS_OOF_METRICS,
    SS_OOF_PROBA_FULL,
    SS_OOF_PROBA_VALID,
    SS_OOF_TRUE_VALID,
    SS_TOP_K,
    SS_X,
    SS_Y,
    has_model_results,
)


def _get_feature_importances(model, feature_names: list[str]) -> dict[str, float]:
    """Extract feature importances from a fitted model."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).flatten()
        imp = np.abs(coef)
    else:
        return {}
    if len(imp) != len(feature_names):
        return {}
    return dict(zip(feature_names, imp))


def _build_rule_based_report(
    model_name: str,
    best_params: dict,
    feature_importances: dict[str, float],
    oof_metrics: dict[str, float],
) -> str:
    lines = [
        f"**Model**: {model_name}",
        f"**Best parameters**: {best_params}",
        "",
    ]
    if oof_metrics:
        lines.append("**OOF Metrics**:")
        for key, val in oof_metrics.items():
            lines.append(f"- {key}: {val:.4f}")
        lines.append("")

    if feature_importances:
        sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        lines.append("**Top features**:")
        for name, score in sorted_feats[:5]:
            lines.append(f"- {name}: {score:.4f}")

    brier = oof_metrics.get("brier_score", 1.0)
    if brier < 0.25:
        lines.append("\nModel shows reasonable calibration (Brier < 0.25).")
    else:
        lines.append("\nCalibration could be improved (Brier >= 0.25).")

    return "\n".join(lines)


def render() -> None:
    st.header("Evaluation & Insight")

    if not has_model_results():
        st.warning("Complete Page 3 (Model Training) first.")
        return

    model_results = st.session_state[SS_MODEL_RESULTS]
    best_model_name = st.session_state[SS_BEST_MODEL_NAME]
    best_params = st.session_state[SS_BEST_PARAMS]
    X = st.session_state.get(SS_X)
    y = st.session_state.get(SS_Y)
    cv = st.session_state.get(SS_CV)
    meta_df = st.session_state.get(SS_META_DF)
    fs_method = st.session_state.get(SS_FEATURE_SELECTION)
    top_k = st.session_state.get(SS_TOP_K, 5)

    if X is None or y is None or cv is None:
        st.warning("Missing required data in session state.")
        return

    # ── Model comparison ───────────────────────────────────────────────
    st.subheader("Model Comparison")
    fig = fig_model_comparison(model_results)
    st.pyplot(fig)

    # ── Optuna history ─────────────────────────────────────────────────
    best_result = next(
        (r for r in model_results if r.name == best_model_name), None
    )
    if best_result is not None:
        with st.expander("Optuna Optimization History"):
            fig = fig_optuna_history(best_result.study)
            st.pyplot(fig)

    # ── OOF Evaluation ─────────────────────────────────────────────────
    st.subheader("Out-of-Fold Evaluation")
    if st.button("Compute OOF Predictions", type="primary"):
        try:
            oof_full, oof_valid, y_valid = evaluate_oof(
                best_model_name,
                best_params,
                X,
                y,
                cv,
                feature_selection=fs_method,
                top_k_features=top_k,
                nested_feature_selection=False,
            )
            metrics = compute_oof_metrics(y_valid, oof_valid)

            st.session_state[SS_OOF_PROBA_FULL] = oof_full
            st.session_state[SS_OOF_PROBA_VALID] = oof_valid
            st.session_state[SS_OOF_TRUE_VALID] = y_valid
            st.session_state[SS_OOF_METRICS] = metrics
        except Exception as exc:
            st.error(f"OOF evaluation failed: {exc}")
            return

    oof_metrics = st.session_state.get(SS_OOF_METRICS)
    oof_proba_full = st.session_state.get(SS_OOF_PROBA_FULL)
    oof_valid = st.session_state.get(SS_OOF_PROBA_VALID)
    y_valid = st.session_state.get(SS_OOF_TRUE_VALID)

    if oof_metrics is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("AUC", f"{best_result.best_auc:.4f}" if best_result else "N/A")
        col2.metric("Brier Score", f"{oof_metrics['brier_score']:.4f}")
        col3.metric("Log Loss", f"{oof_metrics['log_loss']:.4f}")

    # ── Visualizations ─────────────────────────────────────────────────
    if oof_valid is not None and y_valid is not None:
        st.subheader("ROC Curve")
        fig = fig_roc_curve(y_valid, oof_valid)
        st.pyplot(fig)

    if oof_proba_full is not None and meta_df is not None:
        st.subheader("Predictions Overlay")
        fig = fig_predictions(meta_df, oof_proba_full)
        st.pyplot(fig)

    # ── Feature Importance ─────────────────────────────────────────────
    st.subheader("Feature Importance")
    if st.button("Compute Feature Importance"):
        try:
            from sklearn.preprocessing import RobustScaler
            feature_names = list(X.columns)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X.to_numpy())
            if best_model_name == "lgbm":
                X_fit = pd.DataFrame(X_scaled, columns=feature_names)
            else:
                X_fit = X_scaled
            model = build_model(best_model_name, best_params)
            model.fit(X_fit, y.to_numpy())
            importances = _get_feature_importances(model, feature_names)
            if importances:
                st.session_state["_feature_importances"] = importances
            else:
                st.warning("Could not extract feature importances from this model.")
        except Exception as exc:
            st.error(f"Feature importance failed: {exc}")

    fi = st.session_state.get("_feature_importances")
    if fi is not None:
        names = list(fi.keys())
        values = np.array(list(fi.values()))
        fig = fig_feature_importance(names, values)
        st.pyplot(fig)

    # ── LLM Diagnostic Report ──────────────────────────────────────────
    st.subheader("LLM Diagnostic Report")
    if st.button("Generate LLM Report"):
        fi_dict = st.session_state.get("_feature_importances", {})
        metrics_dict = oof_metrics or {}
        try:
            report = generate_llm_brief(
                best_model_name,
                best_params,
                fi_dict,
                metrics_dict,
            )
            st.session_state["_llm_report"] = report
        except Exception as exc:
            st.warning(f"LLM report failed ({exc}), showing rule-based summary.")
            st.session_state["_llm_report"] = _build_rule_based_report(
                best_model_name, best_params, fi_dict, metrics_dict
            )

    report = st.session_state.get("_llm_report")
    if report is not None:
        with st.expander("Diagnostic Report", expanded=True):
            st.markdown(report)
