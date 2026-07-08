from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.src.models.registry import MODEL_REGISTRY
from project.train import main as train_main
from project.eval import evaluate, _parse_baselines  # noqa: F401

import warnings

warnings.filterwarnings(
    "ignore", 
    message="X does not have valid feature names, but.*was fitted with feature names"
)

# Edit these values to configure a full train → eval run.
# Defaults match train.py CLI.
CONFIG = {
    "train": {
        "symbol": "SPY",
        "start": "2010-01-01",
        "end": "2025-12-31",
        "models": list(MODEL_REGISTRY.keys()),
        "n_trials": 30,
        "synthetic": False,
        "horizon": 5,
        "label_method": "expanding",
        "feature_selection": "lasso",
        "nested_feature_selection": False,
        "top_k_features": 12,
        "n_splits": 20,
        "train_window": 252 * 3,
        "test_size": 252 * 1,
        "gap_size": 10,
        "output_dir": None,
        "force_rebuild_features": True,
    },
    "eval": {
        "baselines": None,      # None = all available; e.g. ["prior", "rv_persist"]
        "shap_max_samples": -1, # LinearExplainer background sample size
    }, 
}


def main() -> None:
    load_dotenv()

    args = argparse.Namespace(**CONFIG["train"])

    result = train_main(args)
    if result.run_dir is None:
        raise RuntimeError("Training finished without a run_dir; cannot run eval")

    # Pass eval knobs to evaluation
    baselines = CONFIG["eval"]["baselines"]
    shap_max = CONFIG["eval"]["shap_max_samples"]
    if shap_max is not None and shap_max <= 0:
        shap_max = None
    evaluate(result.run_dir, baselines=baselines, shap_max_samples=shap_max)


if __name__ == "__main__":
    main()
