from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.features.selection import FeatureSelectionResult
from project.src.metrics.baseline import prior_probability_baseline

DEFAULT_COST_MATRIX = np.array([[0, 1], [1, 0]], dtype=float)


def plot_feature_selection(
    result: FeatureSelectionResult,
    output_path: Path,
) -> None:
    scores = result.scores.sort_values(ascending=True)
    colors = ["#2ca02c" if f in result.selected_features else "#cccccc" for f in scores.index]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(scores))))
    ax.barh(scores.index, scores.values, color=colors)
    ax.set_xlabel("Score")
    ax.set_title(f"Feature Selection ({result.method})")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_price_and_rv(
    meta_df: pd.DataFrame,
    target_col: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    if "Date" in meta_df.columns:
        x = meta_df["Date"]
    else:
        x = np.arange(len(meta_df))

    axes[0].plot(x, meta_df["Close"], color="#1f77b4", linewidth=1.2)
    axes[0].set_title("Close Price")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, meta_df[target_col], color="#d62728", linewidth=1.2)
    axes[1].set_title(f"Forward RV ({target_col})")
    axes[1].set_ylabel("Annualized Vol")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_cv_splits(
    cv: TimeSeriesSplitWithPurge,
    X: pd.DataFrame,
    output_path: Path,
) -> None:
    n_samples = len(X)
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
        data=df_plot,
        x="Index",
        y="Fold",
        hue="Set",
        palette={"Train": "#1f77b4", "Gap": "#ff7f0e", "Test": "#2ca02c"},
        marker="|",
        s=400,
        linewidth=2,
        ax=ax,
    )
    ax.set_title("Time Series CV Splits (with purge gap)")
    ax.set_xlabel("Sample Index (Time)")
    ax.set_ylabel("Fold")
    ax.legend(title="Set", loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_model_comparison(model_results, output_path: Path) -> None:
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

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path) -> None:
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
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_predictions(
    meta_df: pd.DataFrame,
    oof_proba_full: np.ndarray,
    output_path: Path,
) -> None:
    plot_df = meta_df.copy()
    plot_df["pred_proba"] = oof_proba_full
    plot_df = plot_df.dropna(subset=["pred_proba"])

    if "Date" in plot_df.columns:
        x = plot_df["Date"]
    else:
        x = np.arange(len(plot_df))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].scatter(
        x,
        plot_df["y"],
        c=plot_df["pred_proba"],
        cmap="RdYlGn",
        s=18,
        alpha=0.8,
    )
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

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _weighted_confusion_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_matrix: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, float]:
    y_pred = (y_proba >= threshold).astype(int)
    y = np.asarray(y_true, dtype=int)
    counts = np.zeros((2, 2), dtype=float)
    for actual in (0, 1):
        for pred in (0, 1):
            counts[actual, pred] = np.sum((y == actual) & (y_pred == pred))
    weighted = cost_matrix * counts
    return weighted, float(weighted.sum())


def plot_asymmetric_loss_matrix(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    *,
    cost_matrix: np.ndarray | None = None,
    threshold: float = 0.5,
    baseline_proba: np.ndarray | None = None,
) -> None:
    cost = np.asarray(cost_matrix if cost_matrix is not None else DEFAULT_COST_MATRIX, dtype=float)
    model_weighted, model_total = _weighted_confusion_matrix(y_true, y_proba, cost, threshold)

    baseline_total: float | None = None
    if baseline_proba is not None:
        _, baseline_total = _weighted_confusion_matrix(
            y_true, baseline_proba, cost, threshold
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    sns.heatmap(
        cost,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[0],
    )
    axes[0].set_title("Asymmetric Cost Matrix")

    sns.heatmap(
        model_weighted,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=axes[1],
    )
    title = f"Model Weighted Confusion (total cost = {model_total:.0f})"
    if baseline_total is not None:
        title += f"\nBaseline total cost = {baseline_total:.0f}"
    axes[1].set_title(title)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_optuna_history(study, output_path: Path) -> None:
    trials = study.trials
    values = [t.value for t in trials if t.value is not None]
    trial_nums = [t.number for t in trials if t.value is not None]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trial_nums, values, marker="o", linestyle="-", color="#1f77b4", markersize=4)
    ax.set_xlabel("Trial")
    ax.set_ylabel("CV AUC")
    ax.set_title("Optuna Optimization History")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

def plot_feature_correlation(df: pd.DataFrame, output_path: Path) -> None:
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)