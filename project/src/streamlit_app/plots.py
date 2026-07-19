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


_RIGHT_LEGEND = dict(
    orientation="v",
    yanchor="top",
    y=1,
    xanchor="left",
    x=1.02,
)

# English captions rendered under charts via ``st.caption``.
CAPTION_BUBBLE = (
    "AUC higher = better ranking of high-vol days. "
    "Brier / LogLoss higher = worse probability quality. "
    "Bubble size = top_k (number of selected features)."
)
CAPTION_RADAR = (
    "Radar axes for Brier and LogLoss are inverted so outward is better. "
    "AUC higher (farther out) = better discrimination."
)
CAPTION_CM = (
    "Cell counts of true vs predicted labels. "
    "Off-diagonal cells are errors (false positives / false negatives)."
)
CAPTION_REGIME_TIMELINE = (
    "Higher marker level = higher macro indicator (e.g. VIX). "
    "Cluster_1 = highest-mean / highest-risk regime."
)
CAPTION_CLUSTER_SCATTER = (
    "Same regime meaning as the timeline: Cluster_1 is highest-risk. "
    "Markers show macro level vs time index."
)
CAPTION_PRED_VS_ACTUAL = (
    "Top: y-position = actual label (0/1); marker colour = predicted class "
    "(red = prob < 0.5, green = prob ≥ 0.5). "
    "Bottom: predicted probability (and normalized forward RV). "
    "Redder background bands = higher-risk macro regime."
)


def fig_bubble_scatter(df: pd.DataFrame) -> go.Figure:
    """Bubble chart: X=brier, Y=auc, color=log_loss, size=top_k."""
    plot_df = df.copy()
    need = ["brier", "auc"]
    for col in need:
        if col not in plot_df.columns:
            fig = go.Figure()
            fig.update_layout(
                title="Bubble chart unavailable (missing brier/auc)",
                template=_TEMPLATE,
            )
            return fig

    x = pd.to_numeric(plot_df["brier"], errors="coerce")
    y = pd.to_numeric(plot_df["auc"], errors="coerce")
    color = (
        pd.to_numeric(plot_df["log_loss"], errors="coerce")
        if "log_loss" in plot_df.columns
        else pd.Series(np.zeros(len(plot_df)))
    )
    size_raw = (
        pd.to_numeric(plot_df["top_k"], errors="coerce")
        if "top_k" in plot_df.columns
        else pd.Series(np.full(len(plot_df), 10.0))
    )
    size_raw = size_raw.fillna(size_raw.median() if size_raw.notna().any() else 10.0)
    smin, smax = float(size_raw.min()), float(size_raw.max())
    if smax > smin:
        sizes = 8.0 + 28.0 * (size_raw - smin) / (smax - smin)
    else:
        sizes = pd.Series(np.full(len(plot_df), 16.0))

    labels = []
    for i, row in plot_df.iterrows():
        eid = row.get("experiment_id") or row.get("model") or i
        labels.append(str(eid)[:48])

    hover = (
        "id: %{customdata[0]}<br>"
        "model: %{customdata[1]}<br>"
        "top_k: %{customdata[2]}<br>"
        "brier: %{x:.4f}<br>"
        "auc: %{y:.4f}<br>"
        "log_loss: %{marker.color:.4f}<extra></extra>"
    )
    custom = np.column_stack(
        [
            labels,
            plot_df["model"].astype(str).tolist()
            if "model" in plot_df.columns
            else ["—"] * len(plot_df),
            size_raw.astype(str).tolist(),
        ]
    )

    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Experiments",
            marker=dict(
                size=sizes,
                color=color,
                colorscale="Viridis_r",
                showscale=True,
                colorbar=dict(title="log_loss", x=1.18),
                line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
                opacity=0.85,
            ),
            customdata=custom,
            hovertemplate=hover,
        )
    )
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text="Experiment bubble chart", x=0.5, xanchor="center"),
        xaxis_title="Brier (lower is better)",
        yaxis_title="AUC (higher is better)",
        height=440,
        showlegend=True,
        legend=_RIGHT_LEGEND,
        margin=dict(l=60, r=140, t=50, b=50),
    )
    return fig


def fig_metrics_radar(rows: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Radar chart for selected experiments (inverts brier/log_loss to 'higher=better')."""
    fig = go.Figure()
    for _, row in rows.iterrows():
        values = []
        for m in metrics:
            v = float(row.get(m, 0.0))
            if m in ("brier", "log_loss"):
                v = 1.0 / (1.0 + max(v, 0.0))
            values.append(v)
        label = str(row.get("experiment_id") or row.get("model") or _)
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=metrics + metrics[:1],
                fill="toself",
                name=label[:40],
            )
        )
    fig.update_layout(
        template=_TEMPLATE,
        title="Metrics radar (brier/log_loss inverted)",
        polar=dict(radialaxis=dict(visible=True)),
        height=420,
        showlegend=True,
        legend=_RIGHT_LEGEND,
        margin=dict(l=50, r=140, t=50, b=40),
    )
    return fig


_CLUSTER_PALETTE = [
    "#e53935", "#1e88e5", "#43a047", "#fb8c00",
    "#8e24aa", "#00897b", "#6d4c41", "#546e7a",
]

_REGIME_BAND_COLORS = [
    "rgba(229, 57, 53, 0.18)",   # Cluster_1 high risk
    "rgba(67, 160, 71, 0.12)",   # lower risk
    "rgba(30, 136, 229, 0.10)",
    "rgba(251, 140, 0, 0.10)",
]


def _cluster_colors(n: int) -> list[str]:
    if n <= 0:
        return []
    return [_CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)] for i in range(n)]


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, np.nan, dtype=float)
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        out = np.full_like(arr, 0.5, dtype=float)
        out[~finite] = np.nan
        return out
    out = (arr - lo) / (hi - lo)
    out[~finite] = np.nan
    return out


def _regime_vrect_shapes(
    x: pd.Series,
    regimes: np.ndarray,
    cluster_order: list | None,
    *,
    yref: str = "y",
    y0: float = 0.0,
    y1: float = 1.0,
) -> list[dict]:
    """Build vertical band shapes for a single cartesian axes pair."""
    regimes = np.asarray(regimes, dtype=object)
    n = min(len(x), len(regimes))
    if n == 0:
        return []
    x = x.iloc[:n]
    regimes = regimes[:n]
    order = list(cluster_order) if cluster_order else sorted(set(regimes.tolist()), key=str)
    color_map = {
        cname: _REGIME_BAND_COLORS[i % len(_REGIME_BAND_COLORS)]
        for i, cname in enumerate(order)
    }

    shapes: list[dict] = []
    i = 0
    while i < n:
        cname = regimes[i]
        j = i + 1
        while j < n and regimes[j] == cname:
            j += 1
        x0 = x.iloc[i]
        x1 = x.iloc[j - 1]
        fill = color_map.get(str(cname), "rgba(128,128,128,0.08)")
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref=yref,
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor=fill,
                line=dict(width=0),
                layer="below",
            )
        )
        i = j
    return shapes


def _add_regime_legend(
    fig: go.Figure,
    regimes: np.ndarray,
    cluster_order: list | None,
) -> None:
    order = list(cluster_order) if cluster_order else sorted(set(regimes.tolist()), key=str)
    for i, cname in enumerate(order):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{cname} regime",
                marker=dict(
                    size=10,
                    color=_CLUSTER_PALETTE[i % len(_CLUSTER_PALETTE)],
                    symbol="square",
                ),
                hoverinfo="skip",
            )
        )


def fig_pred_vs_actual(
    dates,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    title: str = "Predictions vs Actual Labels",
    actual_rv: np.ndarray | None = None,
    regime_labels: np.ndarray | None = None,
    cluster_order: list | None = None,
) -> tuple[go.Figure, go.Figure]:
    """Return two independent figures: actual labels, then predicted prob (+ RV).

    Split charts avoid Plotly ``make_subplots`` marker rendering issues seen in
    Streamlit while keeping shared calendar / regime context.
    """
    x = pd.to_datetime(pd.Series(dates))
    y = np.asarray(y_true, dtype=float)
    proba = np.asarray(y_proba, dtype=float)
    n = min(len(x), len(y), len(proba))
    x, y, proba = x.iloc[:n], y[:n], proba[:n]
    x_list = x.tolist()

    regimes = None
    if regime_labels is not None:
        raw = np.asarray(regime_labels, dtype=object)
        if len(raw) >= n:
            regimes = raw[:n]

    # ── Top: actual binary labels ──────────────────────────────────────
    fig_labels = go.Figure()
    if regimes is not None:
        fig_labels.update_layout(
            shapes=_regime_vrect_shapes(
                x, regimes, cluster_order, yref="y", y0=-0.1, y1=1.1
            )
        )

    label_styles = (
        (False, "Pred = 0 (prob < 0.5)", "#d7191c"),
        (True, "Pred = 1 (prob ≥ 0.5)", "#1a9641"),
    )
    pred_pos = np.isfinite(proba) & (proba >= 0.5)
    for is_pos, name, color in label_styles:
        mask = np.isfinite(y) & np.isfinite(proba) & (pred_pos if is_pos else ~pred_pos)
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        fig_labels.add_trace(
            go.Scatter(
                x=[x_list[i] for i in idx],
                y=y[mask].tolist(),
                mode="markers",
                name=name,
                marker=dict(
                    size=9,
                    color=color,
                    line=dict(width=0.4, color="rgba(0,0,0,0.25)"),
                ),
                customdata=np.asarray(proba[mask], dtype=float),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Actual label: %{y}"
                    "<br>Pred. prob: %{customdata:.3f}<extra></extra>"
                ),
                cliponaxis=False,
            )
        )

    if regimes is not None:
        _add_regime_legend(fig_labels, regimes, cluster_order)

    fig_labels.update_yaxes(
        title_text="Actual Label",
        tickvals=[0, 1],
        range=[-0.1, 1.1],
    )
    fig_labels.update_xaxes(showticklabels=False)
    fig_labels.update_layout(
        template=_TEMPLATE,
        title=dict(
            text=f"{title} — Actual Labels",
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        height=260,
        showlegend=True,
        legend=_RIGHT_LEGEND,
        margin=dict(l=60, r=150, t=50, b=10),
    )

    # ── Bottom: predicted probability (+ optional RV) ──────────────────
    fig_proba = go.Figure()
    if regimes is not None:
        fig_proba.update_layout(
            shapes=_regime_vrect_shapes(
                x, regimes, cluster_order, yref="y", y0=-0.05, y1=1.05
            )
        )

    fig_proba.add_trace(
        go.Scatter(
            x=x_list,
            y=proba.tolist(),
            mode="lines",
            name="Predicted prob",
            line=dict(color="#9467bd", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>Prob: %{y:.4f}<extra></extra>",
        )
    )

    if actual_rv is not None:
        rv = np.asarray(actual_rv, dtype=float)
        if len(rv) >= n:
            rv_norm = _minmax_normalize(rv[:n])
            fig_proba.add_trace(
                go.Scatter(
                    x=x_list,
                    y=rv_norm.tolist(),
                    mode="lines",
                    name="Actual RV (normalized)",
                    line=dict(color=_COLOUR_ACTUAL_RV, width=1.2),
                    hovertemplate="%{x|%Y-%m-%d}<br>RV (norm): %{y:.3f}<extra></extra>",
                )
            )

    fig_proba.add_hline(
        y=0.5,
        line=dict(color=_COLOUR_DIAGONAL, width=1, dash="dash"),
    )
    if regimes is not None:
        _add_regime_legend(fig_proba, regimes, cluster_order)

    fig_proba.update_yaxes(
        title_text="Prob / RV (norm)",
        range=[-0.05, 1.05],
        dtick=0.25,
    )
    fig_proba.update_xaxes(title_text="Date")
    fig_proba.update_layout(
        template=_TEMPLATE,
        title=dict(
            text=f"{title} — Predicted Probability",
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        height=340,
        showlegend=True,
        legend=_RIGHT_LEGEND,
        margin=dict(l=60, r=150, t=50, b=40),
    )
    return fig_labels, fig_proba


def fig_confusion_matrices(
    report: dict,
    split: dict | None = None,
) -> go.Figure:
    """Overall + per-cluster confusion matrix heatmaps (Plotly port of eval CM)."""
    from sklearn.metrics import confusion_matrix as sk_cm

    split = split or report.get("regime_split") or {}
    gcm = split.get("group_confusion_matrices") or {}
    cluster_keys = [k for k in gcm if k != "Overall"]
    # Prefer ordered keys when available
    try:
        from project.src.eval.regime import ordered_group_keys

        ordered = [k for k in ordered_group_keys(gcm) if k != "Overall"]
        if ordered:
            cluster_keys = ordered
    except Exception:
        pass

    panels: list[tuple[str, np.ndarray, int]] = []
    overall = gcm.get("Overall")
    if overall is not None:
        panels.append(
            (
                f"Overall ({overall.get('n_samples', '?')})",
                np.asarray(overall["confusion_matrix"], dtype=int),
                int(overall.get("n_samples", 0)),
            )
        )
    else:
        y_true = np.asarray(report.get("_y_valid", []), dtype=int)
        y_pred = (np.asarray(report.get("_model_proba", [])) >= 0.5).astype(int)
        if len(y_true) and len(y_pred) == len(y_true):
            cm = sk_cm(y_true, y_pred, labels=[0, 1])
            panels.append((f"Overall ({len(y_true)})", cm, len(y_true)))

    means = split.get("cluster_means") or {}
    for name in cluster_keys:
        stats = gcm[name]
        mean_txt = f", mean={means[name]:.2f}" if name in means else ""
        panels.append(
            (
                f"{name} ({stats.get('n_samples', '?')}{mean_txt})",
                np.asarray(stats["confusion_matrix"], dtype=int),
                int(stats.get("n_samples", 0)),
            )
        )

    if not panels:
        fig = go.Figure()
        fig.update_layout(title="No confusion matrices available", template=_TEMPLATE)
        return fig

    n = len(panels)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[p[0] for p in panels],
        horizontal_spacing=0.08,
    )
    for i, (title, cm, _) in enumerate(panels, start=1):
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=["Pred 0", "Pred 1"],
                y=["True 0", "True 1"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=(i == n),
                hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
            ),
            row=1,
            col=i,
        )
    display = split.get("method_display") or split.get("method") or "regime"
    n_regimes = split.get("n_regimes", report.get("n_regimes", "?"))
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(
            text=(
                f"Confusion Matrices — {display} (n={n_regimes}) | "
                f"Model: {report.get('best_model_name')} | "
                f"Horizon={report.get('horizon')}D"
            ),
            x=0.5,
            xanchor="center",
        ),
        height=360,
        margin=dict(l=50, r=40, t=80, b=40),
    )
    return fig


def fig_macro_regime_timeline(
    report: dict,
    split: dict | None = None,
) -> go.Figure:
    """Macro indicator over OOF time, coloured by regime cluster."""
    split = split or report.get("regime_split") or {}
    regimes = np.asarray(split.get("_Regime", []), dtype=object)
    indicator = split.get("macro_indicator", "macro")
    raw_key = f"raw_{indicator}"
    macro = split.get(raw_key)
    if macro is None:
        for k, v in split.items():
            if str(k).startswith("raw_") and isinstance(v, list):
                macro = v
                break
    macro = np.asarray(macro if macro is not None else [], dtype=float)
    n = len(macro)
    if n == 0 or len(regimes) != n:
        fig = go.Figure()
        fig.update_layout(title="No macro regime timeline available", template=_TEMPLATE)
        return fig

    date_list = report.get("_dates", list(range(n)))
    if len(date_list) != n:
        date_list = list(range(n))
    try:
        x = pd.to_datetime(pd.Series(date_list))
    except Exception:
        x = pd.Series(date_list)

    cluster_order = split.get("cluster_order") or sorted(set(regimes.tolist()), key=str)
    colors = _cluster_colors(len(cluster_order))
    means = split.get("cluster_means") or {}

    fig = go.Figure()
    for i, cname in enumerate(cluster_order):
        mask = regimes == cname
        mean_txt = f", μ={means[cname]:.2f}" if cname in means else ""
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=macro[mask],
                mode="markers",
                name=f"{cname} (n={int(mask.sum())}{mean_txt})",
                marker=dict(size=6, color=colors[i], opacity=0.75),
                hovertemplate="%{x}<br>Macro: %{y:.3f}<extra></extra>",
            )
        )

    display = split.get("method_display") or split.get("method") or "regime"
    n_regimes = split.get("n_regimes", report.get("n_regimes", "?"))
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(
            text=f"Macro Regime Timeline — {display} (n={n_regimes}) | Indicator={indicator}",
            x=0.5,
            xanchor="center",
        ),
        xaxis_title="Date",
        yaxis_title="Macro Indicator Value",
        height=380,
        showlegend=True,
        legend=_RIGHT_LEGEND,
        margin=dict(l=50, r=150, t=70, b=40),
    )
    return fig


def fig_cluster_scatter(
    report: dict,
    split: dict | None = None,
) -> go.Figure:
    """(time_index, macro) scatter coloured by cluster."""
    split = split or report.get("regime_split") or {}
    regimes = np.asarray(split.get("_Regime", []), dtype=object)
    indicator = split.get("macro_indicator", "macro")
    raw_key = f"raw_{indicator}"
    macro = split.get(raw_key)
    if macro is None:
        for k, v in split.items():
            if str(k).startswith("raw_") and isinstance(v, list):
                macro = v
                break
    macro = np.asarray(macro if macro is not None else [], dtype=float)
    n = len(macro)
    if n == 0 or len(regimes) != n:
        fig = go.Figure()
        fig.update_layout(title="No cluster scatter available", template=_TEMPLATE)
        return fig

    cluster_order = split.get("cluster_order") or sorted(set(regimes.tolist()), key=str)
    colors = _cluster_colors(len(cluster_order))
    means = split.get("cluster_means") or {}
    x = np.arange(n, dtype=float)

    fig = go.Figure()
    for i, cname in enumerate(cluster_order):
        mask = regimes == cname
        mean_txt = f" μ={means[cname]:.2f}" if cname in means else ""
        fig.add_trace(
            go.Scatter(
                x=x[mask],
                y=macro[mask],
                mode="markers",
                name=f"{cname}{mean_txt}",
                marker=dict(
                    size=7,
                    color=colors[i],
                    opacity=0.8,
                    line=dict(width=0.3, color="black"),
                ),
            )
        )
        if cname in means and mask.any():
            fig.add_trace(
                go.Scatter(
                    x=[float(x[mask].mean())],
                    y=[float(means[cname])],
                    mode="markers",
                    name=f"{cname} centroid",
                    marker=dict(symbol="x", size=12, color="black", line=dict(width=2)),
                    showlegend=False,
                )
            )

    display = split.get("method_display") or split.get("method") or "regime"
    n_regimes = split.get("n_regimes", report.get("n_regimes", "?"))
    fig.update_layout(
        template=_TEMPLATE,
        title=dict(text=f"Cluster Scatter — {display} (n={n_regimes})", x=0.5, xanchor="center"),
        xaxis_title="Time index",
        yaxis_title="Macro indicator",
        height=400,
        showlegend=True,
        legend=_RIGHT_LEGEND,
        margin=dict(l=50, r=150, t=70, b=40),
    )
    return fig


def dm_baselines_dataframe(report: dict) -> pd.DataFrame:
    """Flatten Diebold-Mariano baseline comparison rows for Streamlit tables."""
    rows = []
    for bl in report.get("baselines_compared") or []:
        metrics = bl.get("metrics") or {}
        dm_b = bl.get("dm_brier") or {}
        dm_l = bl.get("dm_log_loss") or {}
        rows.append(
            {
                "baseline": bl.get("name"),
                "brier": metrics.get("brier"),
                "log_loss": metrics.get("log_loss"),
                "dm_brier_stat": dm_b.get("dm_stat"),
                "dm_brier_pvalue": dm_b.get("p_value_two_sided"),
                "dm_logloss_stat": dm_l.get("dm_stat"),
                "dm_logloss_pvalue": dm_l.get("p_value_two_sided"),
            }
        )
    if not rows:
        dm_b = report.get("dm_brier") or {}
        dm_l = report.get("dm_log_loss") or {}
        rows.append(
            {
                "baseline": report.get("baseline"),
                "brier": (report.get("baseline_metrics") or {}).get("brier"),
                "log_loss": (report.get("baseline_metrics") or {}).get("log_loss"),
                "dm_brier_stat": dm_b.get("dm_stat"),
                "dm_brier_pvalue": dm_b.get("p_value_two_sided"),
                "dm_logloss_stat": dm_l.get("dm_stat"),
                "dm_logloss_pvalue": dm_l.get("p_value_two_sided"),
            }
        )
    return pd.DataFrame(rows)
