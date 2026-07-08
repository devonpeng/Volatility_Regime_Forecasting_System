from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from project.src.eval.regime import METHOD_DISPLAY_NAMES, ordered_group_keys


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
        return True
    except Exception:
        return False


def _method_display(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method, method)


def _resolve_splits(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    splits = report.get("regime_splits") or {}
    if splits:
        return splits
    primary = report.get("regime_split")
    if primary:
        key = primary.get("method") or report.get("primary_regime_method") or "regime"
        return {key: primary}
    return {}


def _macro_values(split: dict[str, Any]) -> np.ndarray:
    indicator = split.get("macro_indicator", "")
    raw_key = f"raw_{indicator}"
    vals = split.get(raw_key)
    if vals is None:
        # Fallback: any raw_* key
        for k, v in split.items():
            if k.startswith("raw_") and isinstance(v, list):
                vals = v
                break
    if vals is None:
        return np.array([])
    return np.asarray(vals, dtype=float)


def _cluster_palette(n: int) -> list[str]:
    # Distinct, readable colors for up to ~8 clusters; cycle if more.
    base = [
        "#e53935",  # Cluster_1 (highest mean) — red
        "#1e88e5",  # blue
        "#43a047",  # green
        "#fb8c00",  # orange
        "#8e24aa",  # purple
        "#00897b",  # teal
        "#6d4c41",  # brown
        "#546e7a",  # blue-grey
    ]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        out.append(base[i % len(base)])
    return out


def plot_confusion_matrices_for_split(
    report: dict[str, Any],
    split: dict[str, Any],
    save_dir: Path,
) -> Path | None:
    """Overall + per-cluster CM heatmaps for one regime method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as cf

    method = split.get("method", "unknown")
    n_regimes = int(split.get("n_regimes", report.get("n_regimes", 2)))
    display = split.get("method_display") or _method_display(method)
    gcm = split.get("group_confusion_matrices", {})

    # Prefer Overall from gcm; else compute from report arrays
    cluster_keys = [k for k in ordered_group_keys(gcm) if k != "Overall"]
    n_panels = 1 + len(cluster_keys)
    fig_w = max(8, 5 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    overall_stats = gcm.get("Overall")
    if overall_stats is not None:
        cm_all = np.array(overall_stats["confusion_matrix"])
        n_overall = overall_stats["n_samples"]
    else:
        y_true_all = np.array(report["_y_valid"], dtype=int)
        y_pred_all = (np.array(report["_model_proba"]) >= 0.5).astype(int)
        cm_all = cf(y_true_all, y_pred_all, labels=[0, 1])
        n_overall = int(report.get("n_oof", len(y_true_all)))

    ax = axes[ax_idx]
    ax_idx += 1
    sns.heatmap(
        cm_all,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax.set_title(f"Overall ({n_overall} samples)", fontsize=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    means = split.get("cluster_means") or {}
    colors = _cluster_palette(max(len(cluster_keys), 1))
    for i, name in enumerate(cluster_keys):
        stats = gcm[name]
        ax = axes[ax_idx]
        ax_idx += 1
        cm_r = np.array(stats["confusion_matrix"])
        sns.heatmap(
            cm_r,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        mean_txt = f", mean={means[name]:.2f}" if name in means else ""
        ax.set_title(
            f"{name} ({stats['n_samples']}{mean_txt})",
            fontsize=12,
            color=colors[i % len(colors)],
            fontweight="bold",
        )
        ax.set_xlabel("Predicted label")

    fig.suptitle(
        f"Confusion Matrices — {display} (n={n_regimes}) | "
        f"Model: {report.get('best_model_name')} | Horizon={report.get('horizon')}D",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    fp = save_dir / f"confusion_matrices_{method}_n{n_regimes}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_macro_regime_timeline(
    report: dict[str, Any],
    split: dict[str, Any],
    save_dir: Path,
) -> Path | None:
    """Colour the macro indicator on the time axis by cluster assignment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method = split.get("method", "unknown")
    n_regimes = int(split.get("n_regimes", report.get("n_regimes", 2)))
    display = split.get("method_display") or _method_display(method)
    regimes = np.array(split.get("_Regime", []), dtype=object)
    macro = _macro_values(split)
    n = len(macro)
    if n == 0 or len(regimes) != n:
        return None

    date_list = report.get("_dates", list(range(n)))
    if len(date_list) != n:
        date_list = list(range(n))

    cluster_order = split.get("cluster_order") or sorted(
        set(regimes.tolist()), key=lambda x: str(x)
    )
    means = split.get("cluster_means") or {}
    colors = _cluster_palette(len(cluster_order))
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(cluster_order)}

    fig, ax = plt.subplots(figsize=(18, 4.5))
    x = np.arange(n)
    for i, cname in enumerate(cluster_order):
        mask = regimes == cname
        mean_txt = f", μ={means[cname]:.2f}" if cname in means else ""
        ax.scatter(
            x[mask],
            macro[mask],
            c=color_map[cname],
            s=18,
            alpha=0.75,
            label=f"{cname} (n={mask.sum()}{mean_txt})",
        )

    # Vertical separators when regime changes
    for i in range(1, n):
        if regimes[i] != regimes[i - 1]:
            ax.axvline(i - 0.5, color="black", ls="--", lw=0.5, alpha=0.25)

    tick_pos = np.linspace(0, n - 1, min(10, n))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        [str(date_list[int(t)])[:10] for t in tick_pos],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    indicator = split.get("macro_indicator", "macro")
    ax.set_title(
        f"Macro Regime Timeline — {display} (n={n_regimes}) | Indicator={indicator}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Time (OOF order)")
    ax.set_ylabel("Macro Indicator Value")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    fp = save_dir / f"macro_regime_timeline_{method}_n{n_regimes}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_cluster_scatter(
    report: dict[str, Any],
    split: dict[str, Any],
    save_dir: Path,
) -> Path | None:
    """Sklearn-style scatter of (time_index, macro_value) coloured by cluster."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    method = split.get("method", "unknown")
    n_regimes = int(split.get("n_regimes", report.get("n_regimes", 2)))
    display = split.get("method_display") or _method_display(method)
    regimes = np.array(split.get("_Regime", []), dtype=object)
    macro = _macro_values(split)
    n = len(macro)
    if n == 0 or len(regimes) != n:
        return None

    cluster_order = split.get("cluster_order") or sorted(
        set(regimes.tolist()), key=lambda x: str(x)
    )
    means = split.get("cluster_means") or {}
    colors = _cluster_palette(len(cluster_order))
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(cluster_order)}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n, dtype=float)
    for cname in cluster_order:
        mask = regimes == cname
        mean_txt = f" μ={means[cname]:.2f}" if cname in means else ""
        ax.scatter(
            x[mask],
            macro[mask],
            s=22,
            c=color_map[cname],
            alpha=0.8,
            edgecolors="k",
            linewidths=0.15,
            label=f"{cname}{mean_txt}",
        )
        if cname in means and mask.any():
            ax.scatter(
                [x[mask].mean()],
                [means[cname]],
                marker="x",
                s=120,
                c="black",
                linewidths=2,
                zorder=5,
            )

    ax.set_title(f"{display} (n={n_regimes})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Macro indicator")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    fp = save_dir / f"cluster_scatter_{method}_n{n_regimes}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


def plot_cluster_comparison(
    report: dict[str, Any],
    splits: dict[str, dict[str, Any]],
    save_dir: Path,
) -> Path | None:
    """Multi-panel sklearn-style comparison across regime methods."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not splits:
        return None

    n_regimes = int(report.get("n_regimes", 2))
    methods = list(splits.keys())
    n_methods = len(methods)
    ncols = min(3, n_methods)
    nrows = int(np.ceil(n_methods / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for i, method in enumerate(methods):
        ax = axes_flat[i]
        split = splits[method]
        regimes = np.array(split.get("_Regime", []), dtype=object)
        macro = _macro_values(split)
        n = len(macro)
        if n == 0 or len(regimes) != n:
            ax.set_visible(False)
            continue
        cluster_order = split.get("cluster_order") or sorted(
            set(regimes.tolist()), key=lambda x: str(x)
        )
        colors = _cluster_palette(len(cluster_order))
        color_map = {c: colors[j % len(colors)] for j, c in enumerate(cluster_order)}
        means = split.get("cluster_means") or {}
        x = np.arange(n, dtype=float)
        for cname in cluster_order:
            mask = regimes == cname
            ax.scatter(
                x[mask],
                macro[mask],
                s=12,
                c=color_map[cname],
                alpha=0.75,
                edgecolors="none",
            )
            if cname in means and mask.any():
                ax.scatter(
                    [x[mask].mean()],
                    [means[cname]],
                    marker="x",
                    s=80,
                    c="black",
                    linewidths=1.5,
                    zorder=5,
                )
        display = split.get("method_display") or _method_display(method)
        ax.set_title(f"{display} (n={n_regimes})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Macro indicator")

    for j in range(n_methods, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Cluster Comparison — Macro Indicator (n={n_regimes})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = save_dir / f"cluster_comparison_n{n_regimes}.png"
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


def save_eval_plots(report: dict[str, Any], save_dir: Path | str) -> list[Path]:
    """Write all regime plots for the report under save_dir."""
    if not _has_matplotlib():
        print("\n  Skipped plotting — install matplotlib+seaborn first.")
        return []

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    splits = _resolve_splits(report)
    if not splits:
        print("\n  Skipped plotting — no regime_split in report.")
        return []

    saved: list[Path] = []
    try:
        for method, split in splits.items():
            for fn in (
                plot_confusion_matrices_for_split,
                plot_macro_regime_timeline,
                plot_cluster_scatter,
            ):
                fp = fn(report, split, save_dir)
                if fp is not None:
                    saved.append(fp)
                    print(f"  -> Saved {fp.name} to {fp}")
        if len(splits) > 1:
            fp = plot_cluster_comparison(report, splits, save_dir)
            if fp is not None:
                saved.append(fp)
                print(f"  -> Saved {fp.name} to {fp}")
    except Exception as exc:
        print(f"\n  ! Plotting error: {exc}")
        import traceback

        traceback.print_exc()
    return saved
