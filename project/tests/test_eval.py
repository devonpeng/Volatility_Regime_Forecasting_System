"""Unit tests for project.eval."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from project.eval import _parse_baselines, evaluate


def test_parse_baselines_none_and_list():
    assert _parse_baselines(None) is None
    assert _parse_baselines("prior, garch") == ["prior", "garch"]
    assert _parse_baselines("rv_persist prior") == ["rv_persist", "prior"]


def test_parse_baselines_invalid_and_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _parse_baselines("   ")
    with pytest.raises(ValueError, match="Unknown baseline"):
        _parse_baselines("prior, not_real")


def test_evaluate_missing_dir(tmp_path: Path):
    missing = tmp_path / "no_such_run"
    with pytest.raises(FileNotFoundError, match="Run directory not found"):
        evaluate(missing)


def test_evaluate_happy_path_mocks(tmp_path: Path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    fake_report = {"ok": True}

    with (
        patch("project.eval.run_dm_evaluation", return_value=fake_report) as dm,
        patch("project.eval.print_report") as printer,
        patch("project.eval.save_eval_plots") as saver,
    ):
        out = evaluate(run_dir, baselines=["prior"], shap_max_samples=50)

    dm.assert_called_once()
    assert dm.call_args.args[0] == run_dir.resolve()
    assert dm.call_args.kwargs["baselines"] == ["prior"]
    assert dm.call_args.kwargs["shap_max_samples"] == 50
    printer.assert_called_once_with(fake_report)
    saver.assert_called_once()
    assert saver.call_args.args[1] == run_dir / "plots"
    assert out["_run_dir"] == str(run_dir.resolve())
    assert out["ok"] is True


def test_evaluate_custom_save_dir(tmp_path: Path):
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    save_dir = tmp_path / "custom_plots"

    with (
        patch("project.eval.run_dm_evaluation", return_value={}),
        patch("project.eval.print_report"),
        patch("project.eval.save_eval_plots") as saver,
    ):
        evaluate(run_dir, save_dir=save_dir)

    assert saver.call_args.args[1] == save_dir
