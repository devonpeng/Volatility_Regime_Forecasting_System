from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from project.src.data.load_data import get_data_with_fallback
from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.features.core import build_dataset, get_feature_steps
from project.src.features.selection import (
    FeatureSelectionResult,
    SELECTOR_REGISTRY,
    select_features,
)
from project.src.models import ModelResult, compute_oof_metrics, evaluate_oof, tune_model, tune_models
from project.src.metrics.baseline import prior_probability_baseline
from project.src.models.registry import MODEL_REGISTRY
from project.src.utils.artifacts import save_run_artifacts
from project.src.utils.cache import (
    RunCache,
    dataset_cache_key,
    get_runs_dir,
    load_cached_dataset,
    save_cached_dataset,
)


@dataclass
class PipelineResult:
    data_source: str
    n_samples: int
    n_features: int = 0
    selected_features: list[str] = field(default_factory=list)
    feature_selection_method: str | None = None
    model_results: list[ModelResult] = field(default_factory=list)
    best_model_name: str = ""
    best_auc: float = 0.0
    oof_proba: np.ndarray | None = None
    oof_true: np.ndarray | None = None
    oof_metrics: dict[str, float] = field(default_factory=dict)
    meta_df: pd.DataFrame | None = None
    run_dir: Path | None = None
    plot_dir: Path | None = None
    feature_selection_result: FeatureSelectionResult | None = None


class RVPipeline:
    def __init__(
        self,
        symbol: str = "SPY",
        start_date: str = "2018-01-01",
        end_date: str = "2024-12-31",
        horizon: int = 5,
        models: list[str] | None = None,
        n_trials: int = 30,
        use_synthetic: bool = False,
        label_method: str = "median",
        feature_selection: str | None = None,
        nested_feature_selection: bool = False,
        top_k_features: int = 5,
        n_splits: int = 5,
        train_window_size: int = 252,
        test_size: int = 21,
        gap_size: int = 5,
        output_dir: str | None = None,
        random_state: int = 42,
        force_rebuild_features: bool = False,
    ):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.horizon = horizon
        self.models = models or list(MODEL_REGISTRY.keys())
        self.n_trials = n_trials
        self.use_synthetic = use_synthetic
        self.label_method = label_method
        self.feature_selection = feature_selection
        self.nested_feature_selection = nested_feature_selection
        self.top_k_features = top_k_features
        self.n_splits = n_splits
        self.train_window_size = train_window_size
        self.test_size = test_size
        self.gap_size = gap_size
        self.runs_parent = Path(output_dir) if output_dir else None
        self.random_state = random_state
        self.force_rebuild_features = force_rebuild_features

        self.run_cache: RunCache | None = None
        self.output_dir: Path | None = None

        self.raw_df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.meta_df: pd.DataFrame | None = None
        self.cv: TimeSeriesSplitWithPurge | None = None
        self.model_results: list[ModelResult] = []
        self.best_model_name: str = ""
        self.best_params: dict = {}
        self.oof_proba: np.ndarray | None = None
        self.oof_true: np.ndarray | None = None
        self.oof_proba_full: np.ndarray | None = None
        self.oof_metrics: dict[str, float] = {}
        self.feature_selection_result: FeatureSelectionResult | None = None
        self.selected_features: list[str] = []
        self._dataset_cache_key: str | None = None
        self._loaded_from_dataset_cache: bool = False

    def _pipeline_config(self) -> dict:
        return {
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "horizon": self.horizon,
            "models": self.models,
            "n_trials": self.n_trials,
            "use_synthetic": self.use_synthetic,
            "label_method": self.label_method,
            "feature_selection": self.feature_selection,
            "nested_feature_selection": self.nested_feature_selection,
            "top_k_features": self.top_k_features,
            "random_state": self.random_state,
            "force_rebuild_features": self.force_rebuild_features,
        }

    def _feature_step_names(self) -> list[str]:
        return [step.name for step in get_feature_steps()]

    def _compute_dataset_cache_key(self) -> str:
        return dataset_cache_key(
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            horizon=self.horizon,
            label_method=self.label_method,
            use_synthetic=self.use_synthetic,
            feature_names=self._feature_step_names(),
        )

    def _cv_config(self) -> dict:
        return {
            "n_splits": self.n_splits,
            "train_window_size": self.train_window_size,
            "test_size": self.test_size,
            "gap_size": self.gap_size,
        }

    def load_data(self) -> pd.DataFrame:
        self.raw_df = get_data_with_fallback(
            self.symbol,
            self.start_date,
            self.end_date,
            use_synthetic=self.use_synthetic,
        )
        if self.raw_df.shape[0] == 0:
            raise ValueError(f"No data found for {self.symbol} from {self.start_date} to {self.end_date}")
        width, height = self.raw_df.shape
        if height < 255:
            print("GARCH warning: fewer than 255 features may lead to unstable results or throw an error.")
        return self.raw_df

    def build_features(self) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        cache_key = self._compute_dataset_cache_key()
        self._dataset_cache_key = cache_key

        if not self.force_rebuild_features:
            cached = load_cached_dataset(cache_key)
            if cached is not None:
                self.raw_df, self.X, self.y, self.meta_df = cached
                self._loaded_from_dataset_cache = True
                print(f"Dataset cache hit: {cache_key[:12]}... (skipping load_data/build_features)")
                return self.X, self.y, self.meta_df

        if self.raw_df is None:
            self.load_data()
        assert self.raw_df is not None
        self.X, self.y, self.meta_df = build_dataset(
            self.raw_df,
            horizon=self.horizon,
            label_method=self.label_method,
        )
        self._loaded_from_dataset_cache = False

        feature_names = list(self.X.columns)
        save_cached_dataset(
            cache_key,
            symbol=self.symbol,
            start_date=self.start_date,
            end_date=self.end_date,
            horizon=self.horizon,
            label_method=self.label_method,
            use_synthetic=self.use_synthetic,
            feature_names=feature_names,
            raw_df=self.raw_df,
            X=self.X,
            y=self.y,
            meta_df=self.meta_df,
        )
        action = "rebuilt" if self.force_rebuild_features else "miss"
        print(f"Dataset cache {action}: saved datasets/{cache_key[:12]}...")
        return self.X, self.y, self.meta_df

    def run_feature_selection(self) -> FeatureSelectionResult | None:
        """Select features using tree top-k, lasso, or t-test."""
        # Store full X before feature selection
        self._X_unselected = self.X.copy()
        if self.feature_selection is None:
            return None
        if self.nested_feature_selection:
            print(
                "Feature selection: nested mode enabled, "
                "skip global pre-selection and fit selector within each fold."
            )
            return None

        if self.X is None or self.y is None:
            self.build_features()
        assert self.X is not None and self.y is not None

        if self.feature_selection not in SELECTOR_REGISTRY:
            raise ValueError(
                f"Unknown feature selection: {self.feature_selection}. "
                f"Choose from {list(SELECTOR_REGISTRY)}"
            )

        result = select_features(
            self.X,
            self.y,
            method=self.feature_selection,
            top_k=self.top_k_features,
            random_state=self.random_state,
        )
        self.feature_selection_result = result
        self.selected_features = result.selected_features
        # Keep full X (pre-selection) for baselines that need it
        if hasattr(self, '_X_pre_selection') and self._X_pre_selection is not None:
            pass  # already stored
        else:
            self._X_pre_selection = self._X_unselected.copy() if hasattr(self, '_X_unselected') and self._X_unselected is not None else None
        self.X = result.X_selected
        print(
            f"Feature selection ({self.feature_selection}): "
            f"kept {len(self.selected_features)} / {len(result.scores)} features"
        )
        print(f"  Selected: {', '.join(self.selected_features)}")
        return result

    def create_cv(self) -> TimeSeriesSplitWithPurge:
        if self.X is None:
            self.build_features()
        assert self.X is not None
        self.cv = TimeSeriesSplitWithPurge(
            n_splits=self.n_splits,
            train_window_size=self.train_window_size,
            test_size=self.test_size,
            gap_size=self.gap_size,
        )
        return self.cv

    def tune_model(self, name: str) -> ModelResult:
        if self.X is None or self.y is None:
            self.build_features()
        if self.cv is None:
            self.create_cv()

        assert self.X is not None and self.y is not None and self.cv is not None
        result = tune_model(
            name,
            self.X,
            self.y,
            self.cv,
            n_trials=self.n_trials,
        )
        self.model_results.append(result)
        return result

    def evaluate_best(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.best_model_name:
            raise RuntimeError("Call run() before evaluate_best()")

        assert self.X is not None and self.y is not None and self.cv is not None
        oof_full, oof_valid, oof_true = evaluate_oof(
            self.best_model_name,
            self.best_params,
            self.X,
            self.y,
            self.cv,
            random_state=self.random_state,
            feature_selection=self.feature_selection,
            top_k_features=self.top_k_features,
            nested_feature_selection=self.nested_feature_selection,
        )
        self.oof_proba_full = oof_full
        self.oof_proba = oof_valid
        self.oof_true = oof_true
        self.oof_metrics = compute_oof_metrics(oof_true, oof_valid)
        print(
            "OOF metrics: "
            f"Brier={self.oof_metrics['brier_score']:.4f}, "
            f"Log-Loss={self.oof_metrics['log_loss']:.4f}"
        )
        return self.oof_true, self.oof_proba

    def _init_run_cache(self) -> RunCache:
        parent = self.runs_parent or get_runs_dir()
        self.run_cache = RunCache.create(
            self.best_model_name,
            self.feature_selection,
            parent=parent,
        )
        self.output_dir = self.run_cache.plots_dir
        return self.run_cache

    def plot_all(self) -> Path:
        from project.src import plots as plot_module

        assert self.meta_df is not None and self.X is not None
        assert self.output_dir is not None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        target_col = f"rv_fwd_{self.horizon}d"
        plot_module.plot_price_and_rv(
            self.meta_df,
            target_col=target_col,
            output_path=self.output_dir / "price_and_rv.png",
        )

        if self.feature_selection_result is not None:
            plot_module.plot_feature_selection(
                self.feature_selection_result,
                output_path=self.output_dir / "feature_selection.png",
            )

        if self.cv is not None:
            plot_module.plot_cv_splits(
                self.cv,
                self.X,
                output_path=self.output_dir / "cv_splits.png",
            )

        if self.model_results:
            plot_module.plot_model_comparison(
                self.model_results,
                output_path=self.output_dir / "model_comparison.png",
            )
        # Plot a corr heatmap of the features used in the best model
        if self.selected_features and self.X is not None:
            plot_module.plot_feature_correlation(
                self.X[self.selected_features],
                output_path=self.output_dir / "feature_correlation.png",
            )
        
        best_result = next(
            (r for r in self.model_results if r.name == self.best_model_name),
            None,
        )
        if best_result is not None:
            plot_module.plot_optuna_history(
                best_result.study,
                output_path=self.output_dir / "optuna_history.png",
                model_name=best_result.name,
            )
        for model_result in self.model_results:
            plot_module.plot_optuna_history(
                model_result.study,
                output_path=self.output_dir / f"optuna_history_{model_result.name}.png",
                model_name=model_result.name,
            )

        if self.oof_proba_full is not None and self.oof_true is not None and self.oof_proba is not None:
            plot_module.plot_roc_curve(
                self.oof_true,
                self.oof_proba,
                output_path=self.output_dir / "roc_curve.png",
            )
            plot_module.plot_predictions(
                self.meta_df,
                self.oof_proba_full,
                output_path=self.output_dir / "predictions.png",
            )
            baseline_proba = prior_probability_baseline(self.oof_true)
            plot_module.plot_asymmetric_loss_matrix(
                self.oof_true,
                self.oof_proba,
                output_path=self.output_dir / "asymmetric_loss_matrix.png",
                baseline_proba=baseline_proba,
            )

        return self.output_dir

    def save_artifacts(self) -> Path:
        assert (
            self.run_cache is not None
            and self.raw_df is not None
            and self.X is not None
            and self.y is not None
            and self.meta_df is not None
            and self.oof_proba_full is not None
        )
        return save_run_artifacts(
            self.run_cache,
            pipeline_config=self._pipeline_config(),
            cv_config=self._cv_config(),
            data_source=self.raw_df.attrs.get("source", "unknown"),
            raw_df=self.raw_df,
            X=self.X,
            X_all=getattr(self, '_X_unselected', None),
            y=self.y,
            meta_df=self.meta_df,
            model_results=self.model_results,
            best_model_name=self.best_model_name,
            best_params=self.best_params,
            best_auc=max(r.best_auc for r in self.model_results),
            oof_metrics=self.oof_metrics,
            oof_proba_full=self.oof_proba_full,
            feature_selection_result=self.feature_selection_result,
            selected_features=self.selected_features,
            random_state=self.random_state,
        )

    def run(self) -> PipelineResult:
        print(f"Running pipeline for {self.symbol} from {self.start_date} to {self.end_date} with horizon {self.horizon}")
        print(f"Using models: {self.models}")
        print(f"Using feature selection: {self.feature_selection}")
        print(f"Using nested feature selection: {self.nested_feature_selection}")
        print(f"Using top k features: {self.top_k_features}")
        print(f"Using n splits: {self.n_splits}")
        print(f"Using train window size: {self.train_window_size}")
        print(f"Using test size: {self.test_size}")
        print(f"########################################################")
        print("Building features (with shared dataset cache)...\n")
        self.build_features()
        if self._loaded_from_dataset_cache:
            print("Features loaded from dataset cache")
        else:
            print("Features built successfully")
        print("Running feature selection...\n")
        self.run_feature_selection()
        print(f"Feature selection completed successfully")
        print("Creating cv...\n")
        self.create_cv()
        print("Tuning models...")

        assert self.raw_df is not None and self.X is not None and self.y is not None
        data_source = self.raw_df.attrs.get("source", "unknown")

        assert self.cv is not None
        self.model_results = tune_models(
            self.models,
            self.X,
            self.y,
            self.cv,
            n_trials=self.n_trials,
            feature_selection=self.feature_selection,
            top_k_features=self.top_k_features,
            nested_feature_selection=self.nested_feature_selection,
            random_state=self.random_state,
        )

        best = max(self.model_results, key=lambda r: r.best_auc)
        self.best_model_name = best.name
        self.best_params = best.best_params
        self.evaluate_best()

        self._init_run_cache()
        plot_dir = self.plot_all()
        run_dir = self.save_artifacts()

        from project.src.eval.reporting import run_dm_evaluation

        run_dm_evaluation(run_dir, save_report=True)

        result = PipelineResult(
            data_source=data_source,
            n_samples=len(self.X),
            n_features=len(self.X.columns),
            selected_features=self.selected_features,
            feature_selection_method=self.feature_selection,
            model_results=self.model_results,
            best_model_name=self.best_model_name,
            best_auc=best.best_auc,
            oof_proba=self.oof_proba,
            oof_true=self.oof_true,
            oof_metrics=self.oof_metrics,
            meta_df=self.meta_df,
            feature_selection_result=self.feature_selection_result,
            run_dir=run_dir,
            plot_dir=plot_dir,
        )
        print(f"Run artifacts saved to: {run_dir}")
        return result
