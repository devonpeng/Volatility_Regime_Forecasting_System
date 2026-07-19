"""Station 4: Implementation — Evaluation, Visualisation & Report."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "project").is_dir()),
    None,
)
if _REPO_ROOT is not None and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.src.models.evaluator import compute_oof_metrics, evaluate_oof
from project.src.models.registry import build_model
from project.src.streamlit_app.ai_tutor import (
    get_active_llm_creds,
    render_analyze_button,
    render_shap_section,
)
from project.src.streamlit_app.eval_ui import (
    actual_rv_aligned_to_dates,
    holdout_predictions_from_session,
    regime_bands_aligned_to_dates,
)
from project.src.streamlit_app.llm_helper import generate_llm_brief
from project.src.streamlit_app.plots import (
    CAPTION_PRED_VS_ACTUAL,
    fig_feature_importance,
    fig_model_comparison,
    fig_optuna_history,
    fig_pred_vs_actual,
    fig_roc_curve,
)
from project.src.streamlit_app.precomputed import feature_column_names
from project.src.streamlit_app.session_manager import (
    SS_BEST_MODEL_NAME,
    SS_BEST_PARAMS,
    SS_CV,
    SS_DATE_END,
    SS_DATE_START,
    SS_FEATURE_SELECTION,
    SS_FULL_DF,
    SS_HORIZON,
    SS_META_DF,
    SS_MODEL_RESULTS,
    SS_OOF_METRICS,
    SS_OOF_PROBA_FULL,
    SS_OOF_PROBA_VALID,
    SS_OOF_TRUE_VALID,
    SS_SELECTED_FEATURES,
    SS_TARGET_COL,
    SS_TOP_K,
    SS_TRAIN_RATIO,
    SS_X,
    SS_Y,
    has_model_results,
)
from project.src.streamlit_app.workspace import checkpoint_current_workspace


def _get_feature_importances(model, feature_names: list[str]) -> dict[str, float]:
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
    st.header("📊 Station 4: Implementation — Evaluation & Report")

    if not has_model_results():
        st.warning("Complete Station 3 (Model Design) first.")
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

    best_result = next(
        (r for r in model_results if r.name == best_model_name), None
    )

    # ── 4a. Model Comparison ───────────────────────────────────────────
    st.subheader("4a. Model Comparison")
    fig = fig_model_comparison(model_results)
    st.plotly_chart(fig, use_container_width=True)

    if best_result is not None:
        with st.expander("Optuna Optimization History"):
            fig = fig_optuna_history(best_result.study)
            st.plotly_chart(fig, use_container_width=True)

    # ── 4b. Out-of-Fold Evaluation ────────────────────────────────────
    st.subheader("4b. Out-of-Fold Evaluation")

    # Only show the button if OOF hasn't been computed yet
    oof_metrics = st.session_state.get(SS_OOF_METRICS)
    oof_proba_full = st.session_state.get(SS_OOF_PROBA_FULL)
    oof_valid = st.session_state.get(SS_OOF_PROBA_VALID)
    y_valid = st.session_state.get(SS_OOF_TRUE_VALID)

    if oof_metrics is None and st.button("Compute OOF Predictions", type="primary"):
        try:
            oof_full, oof_v, y_v = evaluate_oof(
                best_model_name,
                best_params,
                X,
                y,
                cv,
                feature_selection=fs_method,
                top_k_features=top_k,
                nested_feature_selection=False,
            )
            metrics = compute_oof_metrics(y_v, oof_v)

            st.session_state[SS_OOF_PROBA_FULL] = oof_full
            st.session_state[SS_OOF_PROBA_VALID] = oof_v
            st.session_state[SS_OOF_TRUE_VALID] = y_v
            st.session_state[SS_OOF_METRICS] = metrics
            oof_metrics = metrics
            oof_proba_full = oof_full
            oof_valid = oof_v
            y_valid = y_v
            checkpoint_current_workspace("oof evaluation")
            st.rerun()
        except Exception as exc:
            st.error(f"OOF evaluation failed: {exc}")

    # KPI Cards (with baselines)
    if oof_metrics is not None:
        auc_val = best_result.best_auc if best_result else None
        prior = float(y.mean()) if y is not None else 0.5
        brier_baseline = prior * (1.0 - prior)
        brier_model = oof_metrics.get('brier_score', 0.0)
        brier_improve = brier_baseline - brier_model

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "OOF AUC",
                f"{auc_val:.4f}" if auc_val else "N/A",
                help="Out-of-fold ROC-AUC: values closer to 1 indicate stronger ranking ability. Baseline = 0.5 (random classifier).",
            )
            st.caption(f"baseline = 0.5")
        with col2:
            st.metric(
                "Brier Score",
                f"{brier_model:.4f}",
                help=f"Brier score: mean squared probability error. Lower is better. Prior-probability baseline = {brier_baseline:.4f}.",
            )
            st.caption(f"baseline = {brier_baseline:.4f}")
        with col3:
            st.metric("Log Loss", f"{oof_metrics.get('log_loss', 'N/A'):.4f}",
                      help="Cross-entropy loss. Lower is better.")

        st.divider()

        # ── 4c. ROC Curve ──────────────────────────────────────────
        st.subheader("4c. ROC Curve")
        if oof_valid is not None and y_valid is not None:
            fig = fig_roc_curve(y_valid, oof_valid)
            st.plotly_chart(fig, use_container_width=True)

        # ── 4d. Predictions vs Actual ───────────────────────────────
        st.subheader("4d. Predictions vs Actual")
        pred_mode = st.radio(
            "Prediction mode",
            options=["eval", "test"],
            format_func=lambda m: "Eval (OOF)" if m == "eval" else "Test (holdout)",
            horizontal=True,
            index=0,
            key="impl_pred_mode",
        )
        train_ratio = float(st.session_state.get(SS_TRAIN_RATIO, 0.8))

        try:
            full_df = st.session_state.get(SS_FULL_DF)
            horizon = int(st.session_state.get(SS_HORIZON, 5))
            date_start = st.session_state.get(SS_DATE_START)
            date_end = st.session_state.get(SS_DATE_END)
            if pred_mode == "eval":
                if oof_proba_full is not None and meta_df is not None:
                    plot_df = meta_df.copy()
                    plot_df["oof_proba"] = oof_proba_full
                    valid = plot_df.dropna(subset=["oof_proba"])
                    dates = valid["Date"] if "Date" in valid.columns else valid.index
                    y_plot = valid["y"].to_numpy() if "y" in valid.columns else np.asarray(y_valid)
                    proba_plot = valid["oof_proba"].to_numpy()
                    title = "Eval (OOF): Predictions vs Actual"
                    actual_rv = actual_rv_aligned_to_dates(
                        dates, horizon=horizon, full_df=full_df
                    )
                    regime_labels, cluster_order = regime_bands_aligned_to_dates(
                        dates,
                        full_df=full_df,
                        start_date=date_start,
                        end_date=date_end,
                    )
                    fig_labels, fig_proba = fig_pred_vs_actual(
                        dates,
                        y_plot,
                        proba_plot,
                        title=title,
                        actual_rv=actual_rv,
                        regime_labels=regime_labels,
                        cluster_order=cluster_order,
                    )
                    st.plotly_chart(fig_labels, use_container_width=True)
                    st.plotly_chart(fig_proba, use_container_width=True)
                    st.caption(CAPTION_PRED_VS_ACTUAL)
                else:
                    st.info("OOF predictions not available yet.")
            else:
                dates, y_test, test_proba = holdout_predictions_from_session(
                    X=X,
                    y=y,
                    meta_df=meta_df,
                    model_name=best_model_name,
                    best_params=best_params,
                    train_ratio=train_ratio,
                )
                title = f"Test holdout (train_ratio={train_ratio:.2f}): Predictions vs Actual"
                actual_rv = actual_rv_aligned_to_dates(
                    dates, horizon=horizon, full_df=full_df
                )
                regime_labels, cluster_order = regime_bands_aligned_to_dates(
                    dates,
                    full_df=full_df,
                    start_date=date_start,
                    end_date=date_end,
                )
                fig_labels, fig_proba = fig_pred_vs_actual(
                    dates,
                    y_test,
                    test_proba,
                    title=title,
                    actual_rv=actual_rv,
                    regime_labels=regime_labels,
                    cluster_order=cluster_order,
                )
                st.plotly_chart(fig_labels, use_container_width=True)
                st.plotly_chart(fig_proba, use_container_width=True)
                st.caption(CAPTION_PRED_VS_ACTUAL)
        except Exception as exc:
            st.warning(f"Pred vs actual unavailable: {exc}")

        # Confusion summary on OOF (when available)
        if oof_proba_full is not None and meta_df is not None:
            valid = meta_df.copy()
            valid["oof_proba"] = oof_proba_full
            valid = valid.dropna(subset=["oof_proba"])
            valid["_correct"] = (
                ((valid["y"] == 1) & (valid["oof_proba"] >= 0.5)) |
                ((valid["y"] == 0) & (valid["oof_proba"] < 0.5))
            )
            n_correct = valid["_correct"].sum()
            n_total = len(valid)
            fp_count = ((valid["y"] == 0) & (valid["oof_proba"] >= 0.5)).sum()
            fn_count = ((valid["y"] == 1) & (valid["oof_proba"] < 0.5)).sum()

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "Accuracy",
                f"{n_correct:,} / {n_total:,}",
                delta=f"{n_correct / n_total * 100:.1f}%",
            )
            col_b.metric("False Positives (FP)", f"{fp_count:,}")
            col_c.metric("False Negatives (FN)", f"{fn_count:,}")

            csv_cols = ["Date", "y", "oof_proba"]
            target_col = st.session_state.get(SS_TARGET_COL) or ""
            if target_col and target_col in valid.columns:
                csv_cols.append(target_col)
            csv_cols = [c for c in csv_cols if c in valid.columns]
            csv_df = valid[csv_cols].copy()
            csv_str = csv_df.to_csv(index=False)
            st.download_button(
                "📥 Export OOF Predictions as CSV",
                data=csv_str,
                file_name="oof_predictions.csv",
                mime="text/csv",
            )

    # ── 4f. Feature Importance ─────────────────────────────────────────
    st.subheader("4f. Feature Importance (All Features)")
    if st.button("Compute Feature Importance", type="primary"):
        try:
            from sklearn.preprocessing import RobustScaler
            full_data = st.session_state.get("_full_data")
            if full_data is None:
                st.warning("Full dataset not available. Re-run Build Dataset in Station 2.")
            else:
                allowed = set(feature_column_names())
                all_fi_cols = [c for c in full_data.columns if c in allowed]
                if not all_fi_cols:
                    st.warning(
                        "No registry features found in the dataset. "
                        "Re-run Build Dataset in Station 2."
                    )
                else:
                    feature_names = all_fi_cols
                    fi_frame = (
                        full_data[all_fi_cols + ["y"]]
                        .replace([np.inf, -np.inf], np.nan)
                        .dropna()
                    )
                    n_dropped = len(full_data) - len(fi_frame)
                    if fi_frame.empty:
                        st.warning(
                            "No complete rows left after dropping NaN/Inf. "
                            "Cannot compute feature importance."
                        )
                    else:
                        if n_dropped:
                            st.caption(
                                f"Dropped {n_dropped:,} row(s) with NaN/Inf before fitting "
                                f"({len(fi_frame):,} remaining)."
                            )
                        y_fi = fi_frame["y"].to_numpy()
                        scaler = RobustScaler()
                        X_scaled = scaler.fit_transform(fi_frame[all_fi_cols].to_numpy())
                        if best_model_name == "lgbm":
                            X_fit = pd.DataFrame(X_scaled, columns=feature_names)
                        else:
                            X_fit = X_scaled
                        model = build_model(best_model_name, best_params)
                        model.fit(X_fit, y_fi)
                        importances = _get_feature_importances(model, feature_names)
                        if importances:
                            st.session_state["_feature_importances"] = importances
                        else:
                            st.warning(
                                "Could not extract feature importances from this model."
                            )
        except Exception as exc:
            st.error(f"Feature importance failed: {exc}")

    fi = st.session_state.get("_feature_importances")
    if fi is not None:
        names = list(fi.keys())
        values = np.array(list(fi.values()))
        selected_set = set(st.session_state.get(SS_SELECTED_FEATURES, []))
        fig = fig_feature_importance(names, values, selected_features=selected_set)
        st.plotly_chart(fig, use_container_width=True)

        # Best practice guidance
        st.info(
            "💡 **Best Practice:** Review the feature ranking above. "
            "High-importance features are the most influential drivers. "
            "Consider returning to **Station 2 (Feature Engineering)** to refine your "
            "feature set — keep the top performers and remove low-importance ones, "
            "then re-run the full pipeline for better results."
        )

        # CSV export
        fi_df = pd.DataFrame([{"feature": k, "importance": v} for k, v in sorted(
            fi.items(), key=lambda x: x[1], reverse=True)])
        fi_csv = fi_df.to_csv(index=False)
        st.download_button(
            "📥 Export Feature Importance as CSV",
            data=fi_csv,
            file_name="feature_importance.csv",
            mime="text/csv",
        )

    # ── SHAP ────────────────────────────────────────────────────────────
    if oof_metrics is not None and X is not None and best_model_name:
        try:
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X.to_numpy())
            X_fit = (
                pd.DataFrame(X_scaled, columns=X.columns)
                if best_model_name == "lgbm"
                else X_scaled
            )
            shap_model = build_model(best_model_name, best_params)
            shap_model.fit(X_fit, y.to_numpy())
            render_shap_section(
                shap_model,
                pd.DataFrame(X_scaled, columns=X.columns),
                best_model_name,
            )
        except Exception as exc:
            st.caption(f"SHAP section unavailable: {exc}")

    # ── 4g. LLM Diagnostic Report ─────────────────────────────────────
    st.subheader("4g. AI Analysis & Diagnostic Report")
    analyze_ctx = {
        "model_name": best_model_name,
        "best_params": best_params,
        "oof_metrics": oof_metrics or {},
        "feature_importances": st.session_state.get("_feature_importances") or {},
    }
    render_analyze_button(analyze_ctx, key="station4_ai")

    if st.button("Generate LLM Report", type="primary"):
        fi_dict = st.session_state.get("_feature_importances", {})
        metrics_dict = oof_metrics or {}
        with st.spinner("Calling LLM…"):
            try:
                api_key, llm_model = get_active_llm_creds()
                report = generate_llm_brief(
                    best_model_name,
                    best_params,
                    fi_dict,
                    metrics_dict,
                    api_key=api_key,
                    llm_model=llm_model,
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
