"""Plotly-based chart functions for the Streamlit workbench.

Each function returns a ``plotly.graph_objects.Figure``.
Callers use ``st.plotly_chart(fig, use_container_width=True)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import auc, roc_curve

_COLOUR_PROBA = "#2c7bb6"
_COLOUR_DIAGONAL = "#888888"
_COLOUR_TRUE = "#1a9641"
_COLOUR_FP = "#d7191c"
_COLOUR_FN = "#fdae61"
_COLOUR_BAR = "#0571b0"
_COLOUR_ACTUAL_RV = "#d73027"
_TEMPLATE = "plotly_white"

_PLOTLY_CONFIG = {
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        "hoverClosestCartesian", "hoverCompareCartesian",
        "toggleSpikelines", "sendDataToCloud",
    ],
    "displaylogo": False,
}


def fig_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> go.Figure:
    """ROC curve with AUC annotation and diagonal reference line."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"AUC = {roc_auc:.4f}",
        line=dict(color=_COLOUR_PROBA, width=2.5),
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random (AUC=0.5)",
        line=dict(color=_COLOUR_DIAGONAL, width=1.5, dash="dash"),
        hoverinfo="skip",
    ))
    fig.add_annotation(
        x=0.75, y=0.25,
        text=f"<b>AUC = {roc_auc:.4f}</b>",
        showarrow=False, font=dict(size=14, color=_COLOUR_PROBA),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=_COLOUR_PROBA, borderwidth=1.5, borderpad=6,
    )
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text="ROC Curve (OOF Predictions)", x=0.5, xanchor="center", font=dict(size=16)),
        xaxis=dict(title="False Positive Rate (FPR)", range=[0, 1], dtick=0.1),
        yaxis=dict(title="True Positive Rate (TPR)", range=[0, 1], dtick=0.1),
        width=700, height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=50, b=60),
    )
    return fig


def fig_predictions(
    meta_df: pd.DataFrame,
    *,
    highlight_errors: bool = True,
    actual_rv_col: str | None = None,
    test_proba: np.ndarray | None = None,
    test_dates: pd.Index | None = None,
) -> go.Figure:
    """Multi-axis time series: actual RV (full range) + predicted prob (test set only).

    Parameters
    ----------
    meta_df : pd.DataFrame
        Full dataset with Date, y columns. If *actual_rv_col* is given, that
        column is plotted as a blue solid line over the entire date range.
    actual_rv_col : str, optional
        Column in *meta_df* holding the actual forward RV (annualised).
    test_proba : np.ndarray, optional
        Predicted probabilities for the test set (red dashed line).
        Requires *test_dates* of the same length.
    test_dates : pd.Index, optional
        Date index for *test_proba* values. Must align 1:1.
    highlight_errors : bool
        Colour-code ground-truth scatter by correct / FP / FN.

    When shown, the blue Actual RV line covers all dates; the red dashed
    predicted-probability line only covers the test period (no overlap).
    """
    # ── Full-range actual RV line ──────────────────────────────────────
    plot_df = meta_df.copy()
    if "Date" in plot_df.columns and not pd.api.types.is_datetime64_any_dtype(plot_df["Date"]):
        plot_df["Date"] = pd.to_datetime(plot_df["Date"])
    plot_df = plot_df.sort_values("Date").reset_index(drop=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Threshold line
    fig.add_hline(
        y=0.5, line=dict(color=_COLOUR_DIAGONAL, width=1, dash="dot"),
        secondary_y=True,
        annotation_text="Threshold 0.5",
        annotation_position="right",
        annotation_font=dict(size=10, color=_COLOUR_DIAGONAL),
    )

    # Actual RV line (blue solid, full range)
    if actual_rv_col and actual_rv_col in plot_df.columns:
        rv_vals = plot_df[actual_rv_col].values
        fig.add_trace(
            go.Scatter(
                x=plot_df["Date"], y=rv_vals,
                mode="lines",
                name="Actual RV (Annualized)",
                line=dict(color=_COLOUR_PROBA, width=1.8),
                hovertemplate="%{x|%Y-%m-%d}<br>Actual RV: %{y:.4f}<extra></extra>",
            ),
            secondary_y=False,
        )

    # ── Test-set predicted probability (red dashed, test period only) ──
    if test_proba is not None and test_dates is not None:
        test_dt = pd.to_datetime(pd.Series(test_dates))
        fig.add_trace(
            go.Scatter(
                x=test_dt, y=np.asarray(test_proba),
                mode="lines",
                name="Predicted Probability (Test)",
                line=dict(color=_COLOUR_ACTUAL_RV, width=1.8, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>Predicted Prob: %{y:.4f}<extra></extra>",
            ),
            secondary_y=True,
        )

    # ── Ground-truth label scatter ────────────────────────────────
    if "y" in plot_df.columns:
        plot_df = plot_df.dropna(subset=["y"])
        if highlight_errors:
            plot_df["_class"] = "correct"
            plot_df.loc[(plot_df["y"] == 1) & (plot_df["oof_proba"] < 0.5) if "oof_proba" in plot_df.columns else [], "_class"] = "fn"
            plot_df.loc[(plot_df["y"] == 0) & (plot_df["oof_proba"] >= 0.5) if "oof_proba" in plot_df.columns else [], "_class"] = "fp"
        else:
            plot_df["_class"] = "correct"

        colour_map = {"correct": _COLOUR_TRUE, "fp": _COLOUR_FP, "fn": _COLOUR_FN}
        label_map = {"correct": "Correct", "fp": "False Positive (FP)", "fn": "False Negative (FN)"}
        marker_map = {"correct": "circle", "fp": "x", "fn": "diamond"}
        for cls in ("correct", "fp", "fn"):
            subset = plot_df[plot_df["_class"] == cls]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["Date"], y=subset["y"],
                    mode="markers", name=label_map[cls],
                    marker=dict(symbol=marker_map[cls], size=7,
                                color=colour_map[cls],
                                line=dict(width=0.8, color="rgba(0,0,0,0.3)")),
                    hovertemplate=(
                        "%{x|%Y-%m-%d}<br>True Label: %{y}<br>"
                        f"Class: {label_map[cls]}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(
            text="Actual Volatility & Predicted Probability",
            x=0.5, xanchor="center", font=dict(size=16),
        ),
        width=1000, height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=80, t=50, b=40),
    )
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title_text="Actual RV / True Label", secondary_y=False)
    fig.update_yaxes(title_text="Predicted Probability", range=[-0.05, 1.05], dtick=0.1, secondary_y=True)
    return fig


def fig_model_comparison(model_results) -> go.Figure:
    """Bar chart comparing best CV AUC across models."""
    names = [r.name for r in model_results]
    aucs = [r.best_auc for r in model_results]
    colours = [_COLOUR_BAR, "#ff7f0e", "#2ca02c"][: len(names)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names, y=aucs,
        marker=dict(color=colours),
        text=[f"{s:.4f}" for s in aucs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>AUC: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text="Model Comparison", x=0.5, xanchor="center", font=dict(size=16)),
        yaxis=dict(title="Best CV AUC", range=[0, 1]),
        xaxis=dict(title="Model"),
        width=800, height=450,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def fig_optuna_history(study) -> go.Figure:
    """Scatter + line plot of Optuna trial values over time."""
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    trial_nums = [t.number for t in trials if t.value is not None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trial_nums, y=values,
        mode="lines+markers",
        name="CV AUC",
        line=dict(color=_COLOUR_PROBA, width=1.5),
        marker=dict(size=5, color=_COLOUR_PROBA),
        hovertemplate="Trial %{x}<br>AUC: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text="Optuna Optimization History", x=0.5, xanchor="center", font=dict(size=16)),
        xaxis=dict(title="Trial"),
        yaxis=dict(title="CV AUC"),
        width=900, height=400,
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def fig_cv_splits(cv, X) -> go.Figure:
    """Time series CV split visualisation (train / gap / test bands)."""
    n_samples = len(X) if isinstance(X, (pd.DataFrame, pd.Series)) else X.shape[0]
    data_points = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        gap_start = train_idx[-1] + 1
        gap_end = test_idx[0] - 1
        if gap_end >= gap_start:
            gap_idx = np.arange(gap_start, gap_end + 1)
        else:
            gap_idx = np.array([], dtype=int)
        for i in train_idx:
            data_points.append((i, fold_idx, "Train"))
        for i in gap_idx:
            data_points.append((i, fold_idx, "Gap"))
        for i in test_idx:
            data_points.append((i, fold_idx, "Test"))

    df_plot = pd.DataFrame(data_points, columns=["Index", "Fold", "Set"])
    colour_map = {"Train": _COLOUR_PROBA, "Gap": "#ff7f0e", "Test": "#2ca02c"}

    fig = go.Figure()
    for set_name in ("Train", "Gap", "Test"):
        subset = df_plot[df_plot["Set"] == set_name]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["Index"], y=subset["Fold"],
            mode="markers",
            name=set_name,
            marker=dict(symbol="line-ns-open", size=10, color=colour_map[set_name]),
            hovertemplate="Index: %{x}<br>Fold: %{y}<br>Set: " + set_name + "<extra></extra>",
        ))

    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text="Time Series CV Splits (with purge gap)", x=0.5, xanchor="center", font=dict(size=16)),
        xaxis=dict(title="Sample Index (Time)"),
        yaxis=dict(title="Fold", dtick=1),
        width=950, height=250,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=50, b=40),
    )
    return fig


def fig_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    selected_features: set[str] | None = None,
) -> go.Figure:
    """Horizontal bar chart of feature importance (descending).

    Features in *selected_features* are coloured blue; others are grey.
    """
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = importances[idx]
    bar_colours = [
        _COLOUR_BAR if f in (selected_features or set()) else "rgba(5,113,176,0.25)"
        for f in sorted_names
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sorted_imp, y=sorted_names,
        orientation="h",
        marker=dict(color=bar_colours),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text="Feature Importance", x=0.5, xanchor="center", font=dict(size=16)),
        xaxis=dict(title="Importance"),
        yaxis=dict(autorange="reversed"),
        width=700, height=max(350, len(sorted_names) * 22 + 80),
        margin=dict(l=120, r=30, t=50, b=60),
    )
    return fig


def fig_feature_correlation(df: pd.DataFrame) -> go.Figure:
    """Heatmap of feature correlations."""
    corr = df.corr().round(2)
    features = corr.columns.tolist()
    z = corr.values

    fig = go.Figure(data=go.Heatmap(
        z=z, x=features, y=features,
        colorscale="RdBu_r", zmid=0,
        text=z, texttemplate="%{text}",
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b>: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text="Feature Correlation Heatmap", x=0.5, xanchor="center", font=dict(size=16)),
        width=800, height=700,
        xaxis=dict(tickangle=45),
        margin=dict(l=80, r=30, t=50, b=120),
    )
    return fig
