"""Thin wrappers around src.plots that return matplotlib Figure objects."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve


def fig_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (OOF Predictions)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_predictions(
    meta_df: pd.DataFrame,
    oof_proba_full: np.ndarray,
) -> plt.Figure:
    plot_df = meta_df.copy()
    plot_df["pred_proba"] = oof_proba_full
    plot_df = plot_df.dropna(subset=["pred_proba"])
    if "Date" in plot_df.columns:
        x = plot_df["Date"]
    else:
        x = np.arange(len(plot_df))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].scatter(x, plot_df["y"], c=plot_df["pred_proba"], cmap="RdYlGn", s=18, alpha=0.8)
    axes[0].set_ylabel("Actual Label")
    axes[0].set_title("Actual High-Vol Labels (colored by predicted probability)")
    axes[0].set_yticks([0, 1])
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, plot_df["pred_proba"], color="#9467bd", linewidth=1.2)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Predicted Prob")
    axes[1].set_xlabel("Date")
    axes[1].set_title("OOF Predicted Probability of High RV")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_model_comparison(model_results) -> plt.Figure:
    names = [r.name for r in model_results]
    aucs = [r.best_auc for r in model_results]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, aucs, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(names)])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Best CV AUC")
    ax.set_title("Model Comparison")
    ax.grid(axis="y", alpha=0.3)
    for bar, score in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    return fig


def fig_optuna_history(study) -> plt.Figure:
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    trial_nums = [t.number for t in trials if t.value is not None]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trial_nums, values, marker="o", linestyle="-", color="#1f77b4", markersize=4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV AUC")
    ax.set_title("Optuna Optimization History")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_cv_splits(cv, X) -> plt.Figure:
    n_samples = len(X) if isinstance(X, (pd.DataFrame, pd.Series)) else X.shape[0]
    data_points = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        gap_start = train_idx[-1] + 1
        gap_end = test_idx[0] - 1
        gap_idx = np.arange(gap_start, gap_end + 1) if gap_end >= gap_start else []
        data_points.extend([(i, fold_idx, "Train") for i in train_idx])
        data_points.extend([(i, fold_idx, "Gap") for i in gap_idx])
        data_points.extend([(i, fold_idx, "Test") for i in test_idx])

    df_plot = pd.DataFrame(data_points, columns=["Index", "Fold", "Set"])
    fig, ax = plt.subplots(figsize=(12, 1.5 * df_plot["Fold"].nunique()))
    sns.scatterplot(
        data=df_plot, x="Index", y="Fold", hue="Set",
        palette={"Train": "#1f77b4", "Gap": "#ff7f0e", "Test": "#2ca02c"},
        marker="|", s=400, linewidth=2, ax=ax,
    )
    ax.set_title("Time Series CV Splits (with purge gap)")
    ax.set_xlabel("Sample Index (Time)")
    ax.set_ylabel("Fold")
    ax.legend(title="Set", loc="upper right")
    fig.tight_layout()
    return fig


def fig_feature_importance(feature_names: list[str], importances: np.ndarray) -> plt.Figure:
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = importances[idx]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(sorted_names))))
    ax.barh(sorted_names[::-1], sorted_imp[::-1], color="#2ca02c")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_feature_correlation(df: pd.DataFrame) -> plt.Figure:
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        cbar=True, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig
