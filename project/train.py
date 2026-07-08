import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.src.features.selection import SELECTOR_REGISTRY
from project.src.models.registry import MODEL_REGISTRY
from project.src.pipline import RVPipeline


def main(args: argparse.Namespace):
    pipeline = RVPipeline(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        horizon=args.horizon,
        models=args.models,
        n_trials=args.n_trials,
        use_synthetic=args.synthetic,
        label_method=args.label_method,
        feature_selection=args.feature_selection,
        nested_feature_selection=args.nested_feature_selection,
        top_k_features=args.top_k_features,
        n_splits=args.n_splits,
        train_window_size=args.train_window,
        test_size=args.test_size,
        gap_size=args.gap_size,
        output_dir=args.output_dir,
        force_rebuild_features=args.force_rebuild_features,
        early_stop_patience=args.early_stopping,
    )

    result = pipeline.run()

    print("\n=== RV Prediction Pipeline Complete ===")
    print(f"Data source : {result.data_source}")
    print(f"Samples     : {result.n_samples}")
    print(f"Features    : {result.n_features}")
    if result.feature_selection_method:
        print(f"Selection   : {result.feature_selection_method} -> {result.selected_features}")
    print(f"Best model  : {result.best_model_name} (AUC={result.best_auc:.4f})")
    if result.oof_metrics:
        print(
            f"OOF metrics : Brier={result.oof_metrics['brier_score']:.4f}, "
            f"Log-Loss={result.oof_metrics['log_loss']:.4f}"
        )
    print("\nModel ranking:")
    for model_result in sorted(result.model_results, key=lambda r: r.best_auc, reverse=True):
        print(f"  {model_result.name:10s}  AUC={model_result.best_auc:.4f}")
    print(f"\nRun dir     : {result.run_dir}")
    print(f"Plots dir     : {result.plot_dir}")
    return result


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="RV prediction pipeline")
    parser.add_argument("--symbol", type=str, default="SPY")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()),
        default=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--label-method",
        choices=["median", "expanding"],
        default="median",
    )
    parser.add_argument(
        "--feature-selection",
        choices=[*SELECTOR_REGISTRY.keys(), "none"],
        default=None,
        help="Feature selection method; omit to skip selection",
    )
    parser.add_argument(
        "--nested-feature-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable per-fold nested feature selection in CV/OOF. "
            "When enabled, global pre-selection is skipped."
        ),
    )
    parser.add_argument("--top-k-features", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=252)
    parser.add_argument("--test-size", type=int, default=21)
    parser.add_argument("--gap-size", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional parent directory for run folders (default: cache/runs)",
    )
    parser.add_argument(
        "--force-rebuild-features",
        action="store_true",
        help="Ignore shared dataset cache and rebuild/save full features",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=5,
        help="Trial-level early stopping patience: terminate a trial if "
             "no fold improves over best for this many consecutive folds (default 5).",
    )
    args = parser.parse_args()
    if args.feature_selection == "none":
        args.feature_selection = None
    main(args)
