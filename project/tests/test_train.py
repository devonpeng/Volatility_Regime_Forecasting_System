"""Unit tests for project.train.main (RVPipeline mocked)."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from project.train import main


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        symbol="SPY",
        start="2018-01-01",
        end="2024-12-31",
        horizon=5,
        models=["logistic"],
        n_trials=3,
        synthetic=True,
        label_method="median",
        feature_selection=None,  # CLI maps "none" → None before main()
        nested_feature_selection=False,
        top_k_features=5,
        n_splits=3,
        train_window=252,
        test_size=21,
        gap_size=5,
        output_dir="/tmp/runs",
        force_rebuild_features=False,
        early_stopping=5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_main_wires_pipeline_kwargs_and_returns_run_result():
    fake_result = SimpleNamespace(
        data_source="synthetic",
        n_samples=100,
        n_features=8,
        feature_selection_method=None,
        selected_features=None,
        best_model_name="logistic",
        best_auc=0.75,
        oof_metrics=None,
        model_results=[SimpleNamespace(name="logistic", best_auc=0.75)],
        run_dir="/tmp/runs/run1",
        plot_dir="/tmp/runs/run1/plots",
    )
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = fake_result

    args = _make_args(
        symbol="QQQ",
        feature_selection=None,
        nested_feature_selection=True,
        train_window=100,
        test_size=10,
        gap_size=2,
    )

    with patch("project.train.RVPipeline", return_value=mock_pipeline) as ctor:
        out = main(args)

    ctor.assert_called_once_with(
        symbol="QQQ",
        start_date="2018-01-01",
        end_date="2024-12-31",
        horizon=5,
        models=["logistic"],
        n_trials=3,
        use_synthetic=True,
        label_method="median",
        feature_selection=None,
        nested_feature_selection=True,
        top_k_features=5,
        n_splits=3,
        train_window_size=100,
        test_size=10,
        gap_size=2,
        output_dir="/tmp/runs",
        force_rebuild_features=False,
        early_stop_patience=5,
    )
    mock_pipeline.run.assert_called_once_with()
    assert out is fake_result


def test_main_passes_feature_selection_method():
    fake_result = SimpleNamespace(
        data_source="synthetic",
        n_samples=10,
        n_features=3,
        feature_selection_method="ttest",
        selected_features=["a"],
        best_model_name="logistic",
        best_auc=0.6,
        oof_metrics={"brier_score": 0.2, "log_loss": 0.5},
        model_results=[SimpleNamespace(name="logistic", best_auc=0.6)],
        run_dir="/tmp/r",
        plot_dir="/tmp/r/plots",
    )
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = fake_result

    with patch("project.train.RVPipeline", return_value=mock_pipeline) as ctor:
        main(_make_args(feature_selection="ttest", top_k_features=3))

    assert ctor.call_args.kwargs["feature_selection"] == "ttest"
    assert ctor.call_args.kwargs["top_k_features"] == 3


def test_main_cli_none_semantics_via_namespace():
    """After CLI mapping, feature_selection='none' becomes None before main()."""
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = SimpleNamespace(
        data_source="x",
        n_samples=1,
        n_features=1,
        feature_selection_method=None,
        selected_features=None,
        best_model_name="logistic",
        best_auc=0.5,
        oof_metrics=None,
        model_results=[SimpleNamespace(name="logistic", best_auc=0.5)],
        run_dir="/tmp/r",
        plot_dir="/tmp/r/p",
    )

    with patch("project.train.RVPipeline", return_value=mock_pipeline) as ctor:
        main(_make_args(feature_selection=None))

    assert ctor.call_args.kwargs["feature_selection"] is None
