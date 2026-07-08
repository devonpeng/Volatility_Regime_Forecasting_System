from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.src.eval.plots import save_eval_plots
from project.src.eval.regime import REGIME_METHOD_NAMES
from project.src.eval.reporting import print_report, run_dm_evaluation

__all__ = [
    "evaluate",
    "main",
    "run_dm_evaluation",
    "print_report",
    "save_eval_plots",
    "_parse_baselines",
]


def evaluate(
    run_dir,
    *,
    baselines=None,
    shap_max_samples=None,
    save_dir=None,
    regime_method: str = "kmeans",
    n_regimes: int = 2,
    clusterer: Any | None = None,
):
    path = Path(run_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")

    report = run_dm_evaluation(
        path,
        baselines=baselines,
        shap_max_samples=shap_max_samples,
        regime_method=regime_method,
        n_regimes=n_regimes,
        clusterer=clusterer,
    )
    print_report(report)

    report["_run_dir"] = str(path)

    # Default plots next to training plots
    out_dir = Path(save_dir) if save_dir is not None else (path / "plots")
    print(f"\n=== Saving eval plots to {out_dir} ===")
    save_eval_plots(report, out_dir)

    return report


def _print_report(report: dict[str, object]) -> None:
    print_report(report)


def _parse_baselines(value: str | None) -> list[str] | None:
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
    method_choices = list(REGIME_METHOD_NAMES) + ["all"]
    parser = argparse.ArgumentParser(
        description="Evaluate OOF predictions vs baselines with Diebold-Mariano tests "
        "and regime-cluster confusion matrices / plots."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a training run directory under cache/runs/",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default=None,
        help=(
            "Comma- or space-separated list of baselines. "
            "Valid: prior, rv_persist, garch."
        ),
    )
    parser.add_argument(
        "--shap-max-samples",
        type=str,
        default="100",
        help="SHAP background samples (default=100, <=0 = all).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "Directory for PNG plots "
            "(default: {run_dir}/plots/). "
            "Writes confusion_matrices_{method}_n{N}.png, "
            "macro_regime_timeline_{method}_n{N}.png, "
            "cluster_scatter_{method}_n{N}.png "
            "(and cluster_comparison_n{N}.png when --regime-method all)."
        ),
    )
    parser.add_argument(
        "--regime-method",
        type=str,
        default="kmeans",
        choices=method_choices,
        help=(
            "Macro-indicator clustering method for regime CMs / plots. "
            f"Choices: {', '.join(method_choices)}. "
            "Default: kmeans. Use 'all' to run every method."
        ),
    )
    parser.add_argument(
        "--n-regimes",
        type=int,
        default=2,
        help="Number of macro clusters / quantile bins (default: 2, must be >= 2).",
    )
    args = parser.parse_args()

    if args.n_regimes < 2:
        raise SystemExit("--n-regimes must be >= 2")

    baselines = _parse_baselines(args.baselines)

    raw_shap = args.shap_max_samples
    try:
        shap_max = int(raw_shap)
    except (ValueError, TypeError):
        shap_max = 100
    if shap_max <= 0:
        shap_max = None

    evaluate(
        args.run_dir,
        baselines=baselines,
        shap_max_samples=shap_max,
        save_dir=args.save_dir,
        regime_method=args.regime_method,
        n_regimes=args.n_regimes,
    )


if __name__ == "__main__":
    main()
