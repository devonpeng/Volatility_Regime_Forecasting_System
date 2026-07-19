#!/usr/bin/env python3
"""Run SPY offline experiment grid from configs/grid_recipe.yaml via RVPipeline.

Persists each experiment under ``data/precomputed/runs/{experiment_id}/`` and
writes ``data/precomputed/grid_results.parquet`` (including a ``run_dir`` column).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "project").is_dir()),
    None,
)
if _REPO_ROOT is not None and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from project.src.pipline import RVPipeline
from project.src.streamlit_app.precomputed import (
    PROJECT_ROOT,
    extract_xy_from_wide,
    grid_results_path,
    load_precomputed_df,
    precomputed_runs_root,
    relative_run_dir,
    run_dir_for_experiment,
    slice_by_dates,
)

RECIPE_PATH = PROJECT_ROOT / "configs" / "grid_recipe.yaml"


def _fs(method: str | None) -> str | None:
    if method in (None, "", "none", "None"):
        return None
    return method


def run_one(recipe: dict, exp: dict) -> dict:
    defaults = recipe.get("defaults") or {}
    symbol = recipe.get("symbol", "SPY")
    horizon = int(exp["horizon"])
    date_key = exp["date_range"]
    dr = recipe["date_ranges"][date_key]
    cv = recipe["cv_profiles"][exp["cv"]]
    model = exp["model"]
    fs = _fs(exp.get("feature_selection"))
    top_k = int(exp.get("top_k", 12))
    label_method = defaults.get("label_method", "expanding")
    n_trials = int(exp.get("n_trials", defaults.get("n_trials", 15)))
    exp_id = str(exp.get("id", ""))

    wide = load_precomputed_df(symbol, horizon)
    sliced = slice_by_dates(wide, dr["start"], dr["end"])
    X, y, meta_df, _ = extract_xy_from_wide(
        sliced,
        horizon=horizon,
        label_method=label_method,
        selected_features=None,
    )

    pipe = RVPipeline(
        symbol=symbol,
        start_date=dr["start"],
        end_date=dr["end"],
        horizon=horizon,
        models=[model],
        n_trials=n_trials,
        label_method=label_method,
        feature_selection=fs,
        nested_feature_selection=bool(defaults.get("nested_feature_selection", False)),
        top_k_features=top_k,
        n_splits=int(cv["n_splits"]),
        train_window_size=int(cv["train_window"]),
        test_size=int(cv["test_size"]),
        gap_size=int(cv["gap"]),
        early_stop_patience=int(defaults.get("early_stop_patience", 5)),
        output_dir=str(precomputed_runs_root()),
    )
    pipe.persist_artifacts = False
    pipe.run_id = exp_id
    pipe.inject_dataset(X, y, meta_df, raw_df=sliced)
    result = pipe.run(persist=False)

    run_path = run_dir_for_experiment(exp_id)
    if run_path.exists():
        shutil.rmtree(run_path)
    pipe.persist_run_artifacts(run_path)

    sel_feats = list(pipe.selected_features or X.columns)
    metrics = result.oof_metrics or {}
    row = {
        "experiment_id": exp_id,
        "symbol": symbol,
        "horizon": horizon,
        "date_range": date_key,
        "start": dr["start"],
        "end": dr["end"],
        "feature_selection": exp.get("feature_selection", "none"),
        "top_k": top_k,
        "model": model,
        "cv_profile": exp["cv"],
        "label_method": label_method,
        "n_trials": n_trials,
        "auc": float(result.best_auc),
        "brier": float(metrics.get("brier_score", float("nan"))),
        "log_loss": float(metrics.get("log_loss", float("nan"))),
        "best_params": json.dumps(pipe.best_params or {}),
        "selected_features": json.dumps(sel_feats),
        "source": "grid",
        "run_dir": relative_run_dir(run_path),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=RECIPE_PATH)
    parser.add_argument("--limit", type=int, default=0, help="Run only first N experiments")
    args = parser.parse_args()

    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    experiments = list(recipe.get("experiments") or [])
    if args.limit > 0:
        experiments = experiments[: args.limit]

    precomputed_runs_root().mkdir(parents=True, exist_ok=True)

    rows = []
    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] {exp.get('id')} ...")
        try:
            rows.append(run_one(recipe, exp))
        except Exception as exc:
            print(f"  FAILED: {exc}")
            rows.append(
                {
                    "experiment_id": exp.get("id", ""),
                    "symbol": recipe.get("symbol", "SPY"),
                    "horizon": exp.get("horizon"),
                    "date_range": exp.get("date_range"),
                    "feature_selection": exp.get("feature_selection"),
                    "top_k": exp.get("top_k"),
                    "model": exp.get("model"),
                    "cv_profile": exp.get("cv"),
                    "auc": float("nan"),
                    "brier": float("nan"),
                    "log_loss": float("nan"),
                    "best_params": "{}",
                    "selected_features": "[]",
                    "source": "grid",
                    "run_dir": "",
                    "error": str(exc),
                }
            )

    out = grid_results_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"Wrote {len(rows)} rows → {out}")
    print(f"Run artifacts under → {precomputed_runs_root()}")


if __name__ == "__main__":
    main()
