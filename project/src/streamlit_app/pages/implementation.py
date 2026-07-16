"""Station 4: Implementation — Evaluation, Visualisation & Report."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "project").is_dir()),
    None,
)
if _REPO_ROOT is not None and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.src.models.evaluator import compute_oof_metrics, evaluate_oof
from project.src.models.registry import build_model
from project.src.streamlit_app.llm_helper import generate_llm_brief
from project.src.streamlit_app.plots import (
    fig_feature_importance,
    fig_model_comparison,
    fig_optuna_history,
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
    SS_SELECTED_FEATURES,
    SS_TARGET_COL,
    SS_TOP_K,
    SS_TRAIN_RATIO,
    SS_X,
    SS_X_TEST,
    SS_X_TRAIN,
    SS_Y,
    SS_Y_TRAIN,
    has_model_results,
)


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

        # ── 4d. Actual Volatility (Full Range) ───────────────────────
        st.subheader("4d. Actual Volatility (Full Time Range)")

        target_col = st.session_state.get(SS_TARGET_COL, "")

        if meta_df is not None:
            plot_df = meta_df.copy()
            if "Date" in plot_df.columns and not pd.api.types.is_datetime64_any_dtype(plot_df["Date"]):
                plot_df["Date"] = pd.to_datetime(plot_df["Date"])
            plot_df = plot_df.sort_values("Date").reset_index(drop=True)

            fig_actual = go.Figure()
            if target_col and target_col in plot_df.columns:
                fig_actual.add_trace(go.Scatter(
                    x=plot_df["Date"], y=plot_df[target_col].values,
                    mode="lines", name="Actual RV (Annualized)",
                    line=dict(color="#2c7bb6", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>Actual RV: %{y:.4f}<extra></extra>",
                ))
            rv_max = float(plot_df[target_col].max())
            y_max = float(np.ceil(rv_max * 1.15 / 0.05) * 0.05)
            fig_actual.update_layout(
                template="plotly_white",
                title=dict(text="Annualized Realized Volatility", x=0.5, xanchor="center"),
                xaxis=dict(title="Date"),
                yaxis=dict(title="Volatility", dtick=0.1, range=[0, y_max]),
                width=1000, height=350,
                margin=dict(l=60, r=30, t=40, b=40),
            )
            st.plotly_chart(fig_actual, use_container_width=True)

        # ── 4e. Predicted Probability (Test Set) ────────────────────
        st.subheader("4e. Model Predicted Probability (Test Set)")

        X_train = st.session_state.get(SS_X_TRAIN)
        X_test = st.session_state.get(SS_X_TEST)
        test_proba_arr = None
        test_dates_arr = None

        if X_train is not None and X_test is not None and best_model_name:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train.to_numpy())
            X_test_scaled = scaler.transform(X_test.to_numpy())
            if best_model_name == "lgbm":
                model_test = build_model(best_model_name, best_params)
                model_test.fit(pd.DataFrame(X_train_scaled, columns=X_train.columns),
                               st.session_state[SS_Y_TRAIN].to_numpy())
            else:
                model_test = build_model(best_model_name, best_params)
                model_test.fit(X_train_scaled, st.session_state[SS_Y_TRAIN].to_numpy())
            test_proba_arr = model_test.predict_proba(
                pd.DataFrame(X_test_scaled, columns=X_test.columns)
                if best_model_name == "lgbm" else X_test_scaled
            )[:, 1]
            train_ratio = st.session_state.get(SS_TRAIN_RATIO, 0.8)
            split_idx = int(len(meta_df) * train_ratio) if meta_df is not None else 0
            test_dates_arr = meta_df.iloc[split_idx:]["Date"].values if meta_df is not None else None

        if test_proba_arr is not None and test_dates_arr is not None:
            test_dt = pd.to_datetime(pd.Series(test_dates_arr))
            fig_pred = go.Figure()
            fig_pred.add_hline(y=0.5, line=dict(color="#888888", width=1, dash="dot"),
                               annotation_text="Threshold 0.5", annotation_position="right")
            fig_pred.add_trace(go.Scatter(
                x=test_dt, y=test_proba_arr,
                mode="lines", name="Predicted Probability",
                line=dict(color="#d73027", width=2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>Prob: %{y:.4f}<extra></extra>",
            ))
            fig_pred.update_layout(
                template="plotly_white",
                title=dict(text="Predicted High-Volatility Probability (Test Set)", x=0.5, xanchor="center"),
                xaxis=dict(title="Date"),
                yaxis=dict(title="Probability", range=[-0.05, 1.05], dtick=0.1),
                width=1000, height=350,
                margin=dict(l=60, r=30, t=40, b=40),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Confusion matrix on OOF (when available)
            if oof_proba_full is not None:
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
                col_a.metric("Accuracy", f"{n_correct:,} / {n_total:,}",
                             delta=f"{n_correct / n_total * 100:.1f}%")
                col_b.metric("False Positives (FP)", f"{fp_count:,}")
                col_c.metric("False Negatives (FN)", f"{fn_count:,}")

                # CSV export
                csv_cols = ["Date", "y", "oof_proba"]
                if target_col and target_col in valid.columns:
                    csv_cols.append(target_col)
                csv_df = valid[csv_cols].copy() if "Date" in valid.columns else valid[csv_cols].copy()
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
                all_fi_cols = [c for c in full_data.columns if c not in ("Date", "Close", "y") and not c.startswith("rv_fwd")]
                feature_names = all_fi_cols
                y_fi = full_data["y"].values
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(full_data[all_fi_cols].values)
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
                    st.warning("Could not extract feature importances from this model.")
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

    # ── 4g. LLM Diagnostic Report ─────────────────────────────────────
    st.subheader("4g. LLM Diagnostic Report")
    if st.button("Generate LLM Report", type="primary"):
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
