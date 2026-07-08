from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.src.evaluation.reporting import print_report, run_dm_evaluation


def _print_report(report: dict[str, object]) -> None:
    """Backward-compatible wrapper for existing imports."""
    print_report(report)


def evaluate(
    run_dir: str | Path,
    *,
    baselines: list[str] | None = None,
    shap_max_samples: int | None = None,
) -> dict[str, object]:
    """Run DM evaluation on a training run directory and print the report."""
    path = Path(run_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")

    report = run_dm_evaluation(
        path,
        baselines=baselines,
        shap_max_samples=shap_max_samples,
    )
    print_report(report)
    return report


def _parse_baselines(value: str | None) -> list[str]:
    """Parse --baselines into a list. Accepts 'a,b,c' or 'a b c'.

    Valid names: prior, rv_persist, garch.
    If not provided, returns None so reporting.py uses its defaults.
    """
    VALID = {"prior", "rv_persist", "garch"}
    if value is None:
        return None
    # Split on commas or whitespace
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
            "Comma- or space-separated list of baselines to compare against. "
            "Valid names: prior, rv_persist, garch. Default (omit) runs all available."
        ),
    )
    parser.add_argument(
        "--shap-max-samples",
        type=str,
        default="100",
        help=(
            "Number of background samples for SHAP LinearExplainer (default=100). "
            "Set <= 0 to use all samples."
        ),
    )
    args = parser.parse_args()

    baselines = _parse_baselines(args.baselines)

    # For --shap-max-samples: <=0 means "use all", otherwise clamp
    raw_shap = args.shap_max_samples
    try:
        shap_max = int(raw_shap)
    except (ValueError, TypeError):
        shap_max = 100
    if shap_max <= 0:
        shap_max = None  # signal "all samples"

    evaluate(args.run_dir, baselines=baselines, shap_max_samples=shap_max)


if __name__ == "__main__":
    main()
