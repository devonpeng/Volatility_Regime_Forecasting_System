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
    rv_hist_persistence_baseline,
)


def run_dm_evaluation(
    run_dir: Path,
    *,
    save_report: bool = True,
) -> dict[str, Any]:
    y_valid, model_proba, horizon, oof_valid_df, best_model_name, pipeline_config = load_run_data(
        run_dir
    )
    model_metrics = {
        "brier": brier_score(y_valid, model_proba),
        "log_loss": log_loss_score(y_valid, model_proba),
    }

    def _evaluate_baseline(
        baseline_name: str,
        baseline_proba: Any,
    ) -> dict[str, Any]:
        brier_diff = per_observation_brier(y_valid, baseline_proba) - per_observation_brier(
            y_valid, model_proba
        )
        log_loss_diff = per_observation_log_loss(
            y_valid, baseline_proba
        ) - per_observation_log_loss(y_valid, model_proba)
        dm_brier = diebold_mariano_test(brier_diff, h=horizon, use_hac=True, hac_lag=horizon - 1)
        dm_log_loss = diebold_mariano_test(
            log_loss_diff, h=horizon, use_hac=True, hac_lag=horizon - 1
        )
        return {
            "name": baseline_name,
            "metrics": {
                "brier": brier_score(y_valid, baseline_proba),
                "log_loss": log_loss_score(y_valid, baseline_proba),
            },
            "dm_brier": {
                "dm_stat": dm_brier.dm_stat,
                "p_value_one_sided": dm_brier.p_value_one_sided,
                "p_value_two_sided": dm_brier.p_value_two_sided,
                "mean_loss_diff": dm_brier.mean_diff,
                "variance_estimator": dm_brier.variance_estimator,
                "hac_lag": dm_brier.hac_lag,
            },
            "dm_log_loss": {
                "dm_stat": dm_log_loss.dm_stat,
                "p_value_one_sided": dm_log_loss.p_value_one_sided,
                "p_value_two_sided": dm_log_loss.p_value_two_sided,
                "mean_loss_diff": dm_log_loss.mean_diff,
                "variance_estimator": dm_log_loss.variance_estimator,
                "hac_lag": dm_log_loss.hac_lag,
            },
        }

    baseline_proba = prior_probability_baseline(y_valid)
    prior_eval = _evaluate_baseline("expanding_prior_probability", baseline_proba)
    baselines: list[dict[str, Any]] = [prior_eval]
    baseline_notes: list[str] = []

    if "rv_hist_5d" in oof_valid_df.columns:
        rv_baseline_proba = rv_hist_persistence_baseline(oof_valid_df["rv_hist_5d"].to_numpy(), y_valid)
        baselines.append(_evaluate_baseline("rv_hist_5d_state_persistence", rv_baseline_proba))
    else:
        baseline_notes.append(
            "rv_hist_5d_state_persistence skipped: rv_hist_5d column not found in OOF evaluation data."
        )

    report: dict[str, Any] = {
        "baseline": prior_eval["name"],
        "dm_variance_settings": {
            "variance_estimator": "newey_west",
            "hac_lag_rule": "horizon_minus_1",
            "hac_lag_value": max(0, horizon - 1),
            "overlap_note": "For horizon=h, adjacent targets overlap by h-1 observations.",
        },
        "n_oof": int(len(y_valid)),
        "horizon": horizon,
        "best_model_name": best_model_name,
        "model": model_metrics,
        "baseline_metrics": prior_eval["metrics"],
        "dm_brier": prior_eval["dm_brier"],
        "dm_log_loss": prior_eval["dm_log_loss"],
        "baselines_compared": baselines,
        "baseline_notes": baseline_notes,
    }

    regimes, regime_threshold, regime_indicator_name, regime_indicator_meta = build_macro_regime_labels(
        oof_valid_df,
        pipeline_config,
        ma_window=30,
    )
    oof_valid_df["Regime"] = regimes
    report["regime_split"] = {
        "macro_indicator": regime_indicator_name,
        "macro_indicator_details": regime_indicator_meta,
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
    dm_settings = report.get("dm_variance_settings", {})
    if dm_settings:
        print(
            "DM variance : "
            f"{dm_settings.get('variance_estimator', 'unknown')} "
            f"(hac_lag={dm_settings.get('hac_lag_value', 0)})"
        )
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
        print(f"  Variance estimator: {dm.get('variance_estimator', 'unknown')}")
        print(f"  HAC lag           : {dm.get('hac_lag', 0)}")
        print(f"  p-value (one-sided): {dm['p_value_one_sided']:.4f}")

    for comp in report.get("baselines_compared", [])[1:]:
        print("\n=== Additional Baseline Comparison ===")
        print(f"Baseline    : {comp['name']}")
        print(
            f"Baseline    : Brier={comp['metrics']['brier']:.4f}, "
            f"Log-Loss={comp['metrics']['log_loss']:.4f}"
        )
        for name, key in [("Brier", "dm_brier"), ("Log-Loss", "dm_log_loss")]:
            dm = comp[key]
            print(f"\nDM ({name}):")
            print(f"  DM stat           : {dm['dm_stat']:.4f}")
            print(f"  Mean loss diff    : {dm['mean_loss_diff']:.4f}")
            print(f"  Variance estimator: {dm.get('variance_estimator', 'unknown')}")
            print(f"  HAC lag           : {dm.get('hac_lag', 0)}")
            print(f"  p-value (one-sided): {dm['p_value_one_sided']:.4f}")
    for note in report.get("baseline_notes", []):
        print(f"\nBaseline note: {note}")

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
    macro_meta = split.get("macro_indicator_details", {})
    if macro_meta.get("used_fallback"):
        print(
            "Macro indicator note: fallback was used because external ^VIX data was unavailable "
            f"or unalignable ({macro_meta.get('reason', 'unknown reason')})."
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
