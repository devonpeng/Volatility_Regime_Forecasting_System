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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OOF predictions vs prior baseline with Diebold-Mariano test"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a training run directory under cache/runs/",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    report = run_dm_evaluation(run_dir)
    _print_report(report)


if __name__ == "__main__":
    main()
