from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.src.metrics import (
    brier_score,
    diebold_mariano_test,
    log_loss_score,
    per_observation_brier,
    per_observation_log_loss,
    prior_probability_baseline,
)


def _load_run_data(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, int, pd.DataFrame, str]:
    y_path = run_dir / "data" / "y.csv"
    oof_path = run_dir / "data" / "oof_predictions.csv"
    pipeline_path = run_dir / "config" / "pipeline.json"

    if not y_path.exists() or not oof_path.exists():
        raise FileNotFoundError(
            f"Missing y.csv or oof_predictions.csv under {run_dir / 'data'}"
        )

    y = pd.read_csv(y_path)["y"].to_numpy()
    oof_df = pd.read_csv(oof_path)
    if "oof_proba" not in oof_df.columns:
        raise ValueError("oof_predictions.csv must contain column 'oof_proba'")

    valid = ~oof_df["oof_proba"].isna()
    y_valid = y[valid.to_numpy()]
    model_proba = oof_df.loc[valid, "oof_proba"].to_numpy(dtype=float)

    horizon = 5
    best_model_name = "unknown"
    if pipeline_path.exists():
        with pipeline_path.open(encoding="utf-8") as f:
            pipeline_config = json.load(f)
        horizon = int(pipeline_config.get("horizon", 5))
        best_model_name = str(pipeline_config.get("best_model_name", "unknown"))

    valid_df = oof_df.loc[valid].reset_index(drop=True)
    valid_df["y"] = y_valid
    valid_df["model_proba"] = model_proba

    return y_valid, model_proba, horizon, valid_df, best_model_name


def _build_macro_regime_labels(
    oof_valid_df: pd.DataFrame,
    *,
    ma_window: int = 30,
) -> tuple[pd.Series, float]:
    """Create High_VIX / Low_VIX regime labels from historical volatility proxy.

    We use rolling volatility from Close (sqrt(252) annualized), then a moving
    average as the macro indicator (similar to VIX MA30 idea).
    """
    if "Close" not in oof_valid_df.columns:
        raise ValueError("oof_predictions.csv must contain column 'Close' for regime split")

    close = oof_valid_df["Close"].astype(float)
    returns = close.pct_change()
    # Historical vol proxy (uses only past and current close)
    hist_vol_30 = returns.rolling(window=30, min_periods=10).std() * np.sqrt(252.0)
    macro_indicator = hist_vol_30.rolling(window=ma_window, min_periods=10).mean()
    macro_indicator = macro_indicator.bfill().ffill()

    threshold = float(macro_indicator.median())
    regimes = pd.Series(
        np.where(macro_indicator >= threshold, "High_VIX", "Low_VIX"),
        index=oof_valid_df.index,
        name="Regime",
    )
    return regimes, threshold


def _group_confusion_matrices(
    df: pd.DataFrame,
    *,
    group_col: str,
    y_col: str = "y",
    proba_col: str = "model_proba",
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Reusable grouped confusion statistics (similar spirit to MetricFrame)."""
    out: dict[str, dict[str, Any]] = {}
    for group_value, part in df.groupby(group_col):
        y_true = part[y_col].astype(int).to_numpy()
        y_pred = (part[proba_col].astype(float).to_numpy() >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        out[str(group_value)] = {
            "n_samples": int(len(part)),
            "threshold": threshold,
            "confusion_matrix": cm.tolist(),  # [[tn, fp], [fn, tp]]
        }
    return out


def run_dm_evaluation(
    run_dir: Path,
    *,
    save_report: bool = True,
) -> dict[str, Any]:
    y_valid, model_proba, horizon, oof_valid_df, best_model_name = _load_run_data(run_dir)
    baseline_proba = prior_probability_baseline(y_valid)

    model_metrics = {
        "brier": brier_score(y_valid, model_proba),
        "log_loss": log_loss_score(y_valid, model_proba),
    }
    baseline_metrics = {
        "brier": brier_score(y_valid, baseline_proba),
        "log_loss": log_loss_score(y_valid, baseline_proba),
    }

    brier_diff = per_observation_brier(y_valid, baseline_proba) - per_observation_brier(
        y_valid, model_proba
    )
    log_loss_diff = per_observation_log_loss(
        y_valid, baseline_proba
    ) - per_observation_log_loss(y_valid, model_proba)

    dm_brier = diebold_mariano_test(brier_diff, h=horizon)
    dm_log_loss = diebold_mariano_test(log_loss_diff, h=horizon)

    report = {
        "baseline": "expanding_prior_probability",
        "n_oof": int(len(y_valid)),
        "horizon": horizon,
        "best_model_name": best_model_name,
        "model": model_metrics,
        "baseline_metrics": baseline_metrics,
        "dm_brier": {
            "dm_stat": dm_brier.dm_stat,
            "p_value_one_sided": dm_brier.p_value_one_sided,
            "p_value_two_sided": dm_brier.p_value_two_sided,
            "mean_loss_diff": dm_brier.mean_diff,
        },
        "dm_log_loss": {
            "dm_stat": dm_log_loss.dm_stat,
            "p_value_one_sided": dm_log_loss.p_value_one_sided,
            "p_value_two_sided": dm_log_loss.p_value_two_sided,
            "mean_loss_diff": dm_log_loss.mean_diff,
        },
    }

    regimes, regime_threshold = _build_macro_regime_labels(oof_valid_df, ma_window=30)
    oof_valid_df["Regime"] = regimes
    xgb_group_cms = _group_confusion_matrices(
        oof_valid_df,
        group_col="Regime",
        y_col="y",
        proba_col="model_proba",
        threshold=0.5,
    )
    report["regime_split"] = {
        "macro_indicator": "rolling_volatility_ma30_from_close",
        "threshold_rule": "median_of_macro_indicator",
        "threshold_value": regime_threshold,
        "group_confusion_matrices": xgb_group_cms,
    }

    if save_report:
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        report_path = config_dir / "dm_test_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        report["report_path"] = str(report_path)

    return report


def _print_report(report: dict[str, Any]) -> None:
    print("\n=== Diebold-Mariano Test (Model vs Prior Baseline) ===")
    print(f"Baseline    : {report['baseline']}")
    print(f"OOF samples : {report['n_oof']}")
    print(
        f"Model       : Brier={report['model']['brier']:.4f}, "
        f"Log-Loss={report['model']['log_loss']:.4f}"
    )
    print(
        f"Baseline    : Brier={report['baseline_metrics']['brier']:.4f}, "
        f"Log-Loss={report['baseline_metrics']['log_loss']:.4f}"
    )

    for name, key in [("Brier", "dm_brier"), ("Log-Loss", "dm_log_loss")]:
        dm = report[key]
        print(f"\nDM ({name}):")
        print(f"  DM stat           : {dm['dm_stat']:.4f}")
        print(f"  Mean loss diff    : {dm['mean_loss_diff']:.4f}")
        print(f"  p-value (one-sided): {dm['p_value_one_sided']:.4f}")

    dm_brier = report["dm_brier"]
    if dm_brier["p_value_one_sided"] < 0.05 and dm_brier["mean_loss_diff"] > 0:
        print(
            "\nConclusion: Model significantly outperforms the prior baseline on Brier "
            "score (DM test, p < 0.05); improvement is unlikely due to chance alone."
        )
    else:
        print(
            "\nConclusion: No statistically significant Brier advantage over the prior "
            "baseline at the 5% level (one-sided DM test)."
        )

    if "report_path" in report:
        print(f"\nReport saved to: {report['report_path']}")
    print("\n=== Regime-based Confusion Matrices ===")
    if str(report.get("best_model_name", "")).lower() != "xgb":
        print(
            "Warning: this run's best model is not xgb. "
            "The matrices below are for the model stored in this run."
        )
    split = report.get("regime_split", {})
    print(
        f"Macro indicator: {split.get('macro_indicator')} "
        f"(threshold={split.get('threshold_value', 0.0):.6f})"
    )
    for regime, stats in split.get("group_confusion_matrices", {}).items():
        print(f"\n{regime}: n={stats['n_samples']}")
        print(f"  confusion_matrix [[tn, fp], [fn, tp]] = {stats['confusion_matrix']}")


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
