from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from project.src.eval.data import load_run_data
from project.src.eval.llm_briefing import build_llm_briefing
from project.src.eval.regime import build_macro_regime_labels, group_confusion_matrices
from project.src.metrics import (
    brier_score,
    diebold_mariano_test,
    log_loss_score,
    per_observation_brier,
    per_observation_log_loss,
    prior_probability_baseline,
)


def run_dm_evaluation(
    run_dir: Path,
    *,
    save_report: bool = True,
) -> dict[str, Any]:
    y_valid, model_proba, horizon, oof_valid_df, best_model_name, pipeline_config = load_run_data(
        run_dir
    )
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

    report: dict[str, Any] = {
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

    regimes, regime_threshold, regime_indicator_name = build_macro_regime_labels(
        oof_valid_df,
        pipeline_config,
        ma_window=30,
    )
    oof_valid_df["Regime"] = regimes
    report["regime_split"] = {
        "macro_indicator": regime_indicator_name,
        "threshold_rule": "median_of_macro_indicator",
        "threshold_value": regime_threshold,
        "group_confusion_matrices": group_confusion_matrices(
            oof_valid_df,
            group_col="Regime",
            y_col="y",
            proba_col="model_proba",
            threshold=0.5,
        ),
    }
    report["llm_briefing"] = build_llm_briefing(run_dir)

    if save_report:
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        report_path = config_dir / "dm_test_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        report["report_path"] = str(report_path)

    return report


def print_report(report: dict[str, Any]) -> None:
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
    split = report.get("regime_split", {})
    print(
        f"Macro indicator: {split.get('macro_indicator')} "
        f"(threshold={split.get('threshold_value', 0.0):.6f})"
    )
    for regime, stats in split.get("group_confusion_matrices", {}).items():
        print(f"\n{regime}: n={stats['n_samples']}")
        print(f"  confusion_matrix [[tn, fp], [fn, tp]] = {stats['confusion_matrix']}")

    llm = report.get("llm_briefing", {})
    if llm:
        print("\n=== LLM Briefing ===")
        print(f"Source       : {llm.get('source', 'unknown')}")
        if llm.get("error"):
            print(f"LLM status   : fallback triggered ({llm['error']})")
        print(f"Final brief  : {llm.get('final_brief', '')}")
