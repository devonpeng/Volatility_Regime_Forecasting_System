from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.src.eval.reporting import print_report, run_dm_evaluation


def _has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        matplotlib.use("Agg")
        import seaborn  # noqa: F401
        return True
    except Exception:
        return False


def plot_confusion_matrices(report, save_dir=None):
    """Plot overall + per-regime confusion-matrix heatmaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    gcm = report.get("regime_split", {}).get("group_confusion_matrices", {})

    regimes_in_gcm = [k for k in gcm if k != "Overall"]
    n_panels = 1 + len(regimes_in_gcm)

    fig_w = max(8, 6 * n_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0

    # Overall CM
    y_true_all = np.array(report["_y_valid"], dtype=int)
    y_pred_all = (np.array(report["_model_proba"]) >= 0.5).astype(int)
    from sklearn.metrics import confusion_matrix as cf
    cm_all = cf(y_true_all, y_pred_all, labels=[0, 1])

    ax = axes[ax_idx]; ax_idx += 1
    sns.heatmap(cm_all, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    ax.set_title(f"Overall ({report['n_oof']} samples)", fontsize=13)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    prec = cm_all[1, 1] / max(cm_all[1, :].sum(), 1)
    rec  = cm_all[1, 1] / max(cm_all[:, 1].sum(), 1)
    ax.text(0.5, -0.28, f"Prec={prec:.2f}  Rec={rec:.2f}", ha="center",
            transform=ax.transAxes, fontsize=11, bbox=dict(fc="#fff3e0"))

    # Per-regime CMs
    regime_colors = {"High_VIX": "#ef5350", "Low_VIX": "#42a5f5"}
    for name, stats in gcm.items():
        if name == "Overall":
            continue
        ax = axes[ax_idx]; ax_idx += 1
        cm_r = np.array(stats["confusion_matrix"])
        sns.heatmap(cm_r, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        col = regime_colors.get(name, "#66bb6a")
        ax.set_title(f"{name}  ({stats['n_samples']} samples)",
                     fontsize=13, color=col, fontweight="bold")
        ax.set_xlabel("Predicted label")
        prec_r = cm_r[1, 1] / max(cm_r[1, :].sum(), 1)
        rec_r  = cm_r[1, 1] / max(cm_r[:, 1].sum(), 1)
        ax.text(0.5, -0.28, f"Prec={prec_r:.2f}  Rec={rec_r:.2f}", ha="center",
                transform=ax.transAxes, fontsize=11, bbox=dict(fc="#fff3e0"))

    fig.suptitle(
        f"Confusion Matrices — Model: {report['best_model_name']} | "
        f"Horizon={report['horizon']}D | Threshold=0.5",
        fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_dir is not None:
        fp = save_dir / "confusion_matrices.png"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"\n  -> Saved confusion-matrix plot to {fp}")
    plt.close(fig)


def plot_vix_ma30_clusters(report, save_dir=None):
    """Cluster macro indicator via KMeans(k=2) and plot with coloured markers + band separators."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans

    regime_info = report.get("regime_split", {})
    threshold_value = float(regime_info.get("threshold_value", 25.0))
    indicator_name = regime_info.get("macro_indicator", "")
    raw_key = "raw_" + indicator_name
    macro_vals_raw = np.asarray(regime_info.get(raw_key, []))
    n = int(report.get("n_oof", len(macro_vals_raw)))

    # Dates from the OOF DataFrame (saved by reporting.py)
    date_list = report.get("_dates", list(range(n)))
    if len(date_list) != n:
        date_list = list(range(n))

    # Try to fetch VIX MA30 from yfinance
    use_local = False
    if len(macro_vals_raw) == 0 or len(macro_vals_raw) < 2:
        use_local = True
    elif bool(regime_info.get("used_fallback", False)):
        use_local = True

    if not use_local:
        from project.src.eval.regime import fetch_vix_ma_series
        start_d = str(date_list[0])[:10]
        end_d   = str(date_list[-1])[:10]
        vix_ma = fetch_vix_ma_series(start_d, end_d, ma_window=30)
        vix_map = {d.strftime("%Y-%m-%d"): float(v)
                   for d, v in zip(vix_ma.index.date, vix_ma.values)}
        vix_vals = np.array([vix_map.get(dd[:10], np.nan) for dd in date_list])
        vix_source = "^VIX"
    else:
        vix_vals = np.array(macro_vals_raw, dtype=float)
        vix_source = "(fallback)"

    # KMeans clustering
    clusters = np.full(n, -1, dtype=int)
    if (~np.isnan(vix_vals)).sum() >= 2:
        mask = ~np.isnan(vix_vals)
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        fitted = km.fit(vix_vals[mask].reshape(-1, 1))
        tmp = np.full(n, -1)
        tmp[mask] = fitted.labels_
        clusters = tmp

    centers = km.cluster_centers_.flatten()
    hi_center = int(np.argmax(centers))
    cat = ["High_Vol" if c == hi_center else "Low_Vol" for c in clusters]
    lo_mask = np.array([c != hi_center for c in cat])
    hi_mask = ~lo_mask

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.scatter(np.arange(n)[lo_mask], vix_vals[lo_mask], c="blue", alpha=0.7, s=40,
               label=f"Low_Vol ({lo_mask.sum()})")
    ax.scatter(np.arange(n)[hi_mask], vix_vals[hi_mask], c="red", alpha=0.7, s=40,
               label=f"High_Vol ({hi_mask.sum()})")

    for i in range(1, n):
        if cat[i] != cat[i - 1]:
            ax.axvline(i - 0.5, color="black", ls="--", lw=0.8, alpha=0.3)

    # Nice x-axis with date tickers
    ax.set_xticks(np.linspace(0, n - 1, min(10, n)))
    ax.set_xticklabels(
        [str(date_list[int(t)])[:10] for t in np.linspace(0, n - 1, min(10, n))],
        rotation=45, ha="right", fontsize=9)

    ax.set_title(
        f"VIX MA30 Clusters (KMeans k=2)\nThreshold={threshold_value:.4f} | "
        f"N={n}  |  Indicator={vix_source}",
        fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (OOF order)")
    ax.set_ylabel("Macro Indicator Value")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if save_dir is not None:
        fp = save_dir / "vix_ma30_clusters.png"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"  -> Saved VIX-MA30 cluster plot to {fp}")
    plt.close(fig)


# ──────────────────────── public API ────────────────────────


def evaluate(run_dir, *, baselines=None, shap_max_samples=None, save_dir=None):
    path = Path(run_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")

    report = run_dm_evaluation(path, baselines=baselines,
                               shap_max_samples=shap_max_samples)
    print_report(report)

    # Attach metadata for downstream plotting
    report["_run_dir"] = str(path)

    # Save plots when requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if _has_matplotlib():
            try:
                plot_confusion_matrices(report, save_dir=save_dir)
                plot_vix_ma30_clusters(report, save_dir=save_dir)
            except Exception as exc:
                print(f"\n  ! Plotting error: {exc}")
                import traceback; traceback.print_exc()
        else:
            print("\n  Skipped plotting — install matplotlib+seaborn first.")

    return report


def _print_report(report: dict[str, object]) -> None:
    print_report(report)


def _parse_baselines(value: str | None) -> list[str]:
    VALID = {"prior", "rv_persist", "garch"}
    if value is None:
        return None
    items = [
        name.strip()
        for part in value.replace(",", " ").split()
        for name in [part]
        if name
    ]
    if not items:
        raise ValueError("--baselines must be non-empty")
    bad = set(items) - VALID
    if bad:
        raise ValueError(f"Unknown baseline(s): {bad}. Valid: {VALID}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OOF predictions vs one or more baselines with Diebold-Mamar test"
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to a training run directory under cache/runs/",
    )
    parser.add_argument(
        "--baselines", type=str, default=None,
        help=(
            "Comma- or space-separated list of baselines. "
            "Valid: prior, rv_persist, garch."
        ),
    )
    parser.add_argument(
        "--shap-max-samples", type=str, default="100",
        help="SHAP background samples (default=100, <=0 = all).",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Save dir for PNG plots (confusion_matrices.png, vix_ma30_clusters.png).",
    )
    args = parser.parse_args()

    baselines = _parse_baselines(args.baselines)

    raw_shap = args.shap_max_samples
    try:
        shap_max = int(raw_shap)
    except (ValueError, TypeError):
        shap_max = 100
    if shap_max <= 0:
        shap_max = None

    evaluate(args.run_dir, baselines=baselines, shap_max_samples=shap_max,
             save_dir=args.save_dir)


if __name__ == "__main__":
    main()
