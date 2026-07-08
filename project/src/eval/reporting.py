from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.eval.data import load_run_data
from project.src.eval.llm_briefing import build_llm_briefing
from project.src.eval.regime import build_macro_regime_labels, group_confusion_matrices
from project.src.metrics import (
    brier_score,
    diebold_mariano_test,
    garch_calibrated_baseline_oof,
    log_loss_score,
    per_observation_brier,
    per_observation_log_loss,
    prior_probability_baseline,
    rv_hist_persistence_baseline,
)


# Mapping: user-facing name → (function, arg_key_in_df_or_config, fallback_note)
_BASELINE_SPECS = {
    "prior": ("prior_probability_baseline", None, None),
    "rv_persist": ("rv_hist_persistence_baseline", "rv_fwd_5d", "rv_fwd_5d_state_persistence skipped: rv_fwd_5d column not found in OOF evaluation data."),
    "garch": ("garch_calibrated_baseline_oof", "garch_vol_1d", "garch_vol_1d_logistic_calibration skipped: garch_vol_1d / CV artifacts not found in run data."),
}


_BASELINE_DESCRIPTIONS = {
    "prior": (
        "Prior Probability Baseline\n"
        "  - Prediction: expanding historical mean of labels up to time t (no look-ahead).\n"
        "  - For observation t, p_t = mean(y[:t]). The first sample uses mean(y) as fallback."
    ),
    "rv_persist": (
        "RV Persistence Baseline\n"
        "  - Uses today's realized volatility histogram high/low state (rv_hist_5d) vs. an\n"
        "    expanding median threshold computed from past observations (no look-ahead).\n"
        "  - Predicts p_t = 1 when today's rv_hist_5d[t] >= expanding median of rv_hist_5d[:t],\n"
        "    else p_t = 0. First sample falls back to unconditional mean(y)."
    ),
    "garch": (
        "GARCH-Calibrated Baseline\n"
        "  - Calibrates a single GARCH volatility forecast (garch_vol_1d) into binary probabilities\n"
        "    via out-of-fold logistic regression with StandardScaler + LogisticRegression(max_iter=1000).\n"
        "  - TimeSeriesSplitWithPurge CV is used for OOF predictions; min 50 training samples and\n"
        "    at least 2 label classes required per fold. Fallback: mean(y_train) when conditions unmet."
    ),
}


def _run_single_baseline(
    baseline_name: str,
    compute_proba_fn,
    y_valid,
    oof_valid_df,
    pipeline_config,
) -> tuple[Any | None, str | None]:
    """Return (baseline_proba_or_None, skip_note_or_None)."""
    spec = _BASELINE_SPECS.get(baseline_name)
    if spec is None:
        return None, f"Unknown baseline '{baseline_name}'"

    fn_name, col_or_feat, skip_note = spec

    try:
        fn = getattr(__import__("project.src.metrics", fromlist=[fn_name]), fn_name)
    except ImportError:
        return None, f"Baseline function '{fn_name}' not found in metrics module."

    if baseline_name == "prior":
        proba = fn(y_valid)
        return proba, None

    if baseline_name == "rv_persist":
        if col_or_feat not in oof_valid_df.columns:
            return None, skip_note
        return fn(oof_valid_df[col_or_feat].to_numpy(), y_valid), None

    if baseline_name == "garch":
        x_full = pipeline_config.get("_x_full")
        y_full = pipeline_config.get("_y_full")
        oof_valid_mask = pipeline_config.get("_oof_valid_mask")
        cv_config = pipeline_config.get("_cv_config") or {}
        if (
            x_full is None
            or y_full is None
            or oof_valid_mask is None
            or col_or_feat not in x_full.columns
            or not cv_config
        ):
            return None, skip_note

        cv = TimeSeriesSplitWithPurge(
            n_splits=int(cv_config.get("n_splits", 5)),
            train_window_size=int(cv_config.get("train_window_size", 252)),
            test_size=int(cv_config.get("test_size", 21)),
            gap_size=int(cv_config.get("gap_size", 5)),
        )
        garch_oof = fn(x_full, y_full, cv, feature_name=col_or_feat)
        garch_proba = garch_oof.to_numpy(dtype=float)[oof_valid_mask]
        if np.isnan(garch_proba).any():
            fallback = float(np.nanmean(y_valid)) if len(y_valid) else 0.5
            garch_proba = np.where(np.isnan(garch_proba), fallback, garch_proba)
        return garch_proba, None

    return None, f"Unsupported baseline '{baseline_name}'"


def _evaluate_baseline(
    baseline_name: str,
    baseline_proba,
    y_valid,
    model_proba,
    horizon,
) -> dict[str, Any]:
    brier_diff = per_observation_brier(y_valid, baseline_proba) - per_observation_brier(
        y_valid, model_proba
    )
    log_loss_diff = (
        per_observation_log_loss(y_valid, baseline_proba)
        - per_observation_log_loss(y_valid, model_proba)
    )
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


def run_dm_evaluation(
    run_dir: Path,
    *,
    save_report: bool = True,
    baselines: list[str] | None = None,
    shap_max_samples: int | None = None,
) -> dict[str, Any]:
    y_valid, model_proba, horizon, oof_valid_df, best_model_name, pipeline_config = load_run_data(
        run_dir
    )
    model_metrics = {
        "brier": brier_score(y_valid, model_proba),
        "log_loss": log_loss_score(y_valid, model_proba),
    }

    # ── resolve which baselines to run ──────────────────────────────
    VALID = {"prior", "rv_persist", "garch"}
    if baselines is None:
        requested = list(VALID)  # all three
    else:
        requested = baselines

    baselines_out: list[dict[str, Any]] = []
    baseline_notes: list[str] = []

    for bl_name in requested:
        proba, note = _run_single_baseline(bl_name, None, y_valid, oof_valid_df, pipeline_config)
        if proba is None:
            baseline_notes.append(note or f"{bl_name}: skipped")
            continue
        try:
            eval_result = _evaluate_baseline(bl_name, proba, y_valid, model_proba, horizon)
            baselines_out.append(eval_result)
        except Exception as exc:
            baseline_notes.append(f"{bl_name}: evaluation error: {exc}")

    # Build top-level report fields (keep prior_eval-like keys for compat)
    prior_entry = next((b for b in baselines_out if b["name"] == "prior"), None)
    if prior_entry is None:
        prior_entry = baselines_out[0] if baselines_out else {"name": "none", "metrics": {"brier": 0.0, "log_loss": 0.0}}

    report: dict[str, Any] = {
        "baseline": prior_entry["name"],
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
        "baseline_metrics": prior_entry["metrics"],
        "dm_brier": prior_entry["dm_brier"],
        "dm_log_loss": prior_entry["dm_log_loss"],
        "baselines_compared": baselines_out,
        "baseline_notes": baseline_notes,
        # ── fields for programmatic access / plotting (not serialised) ──
        "_y_valid": y_valid.tolist(),
        "_model_proba": model_proba.tolist(),
    }

    regimes, regime_threshold, regime_indicator_name, regime_indicator_meta, macro_indicator_raw = build_macro_regime_labels(
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
    # Store regime vector & raw VIX series for downstream plotting (JSON-serialisable)
    report["regime_split"]["_Regime"] = regimes.tolist()
    report["regime_split"].update({k: v for k, v in regime_indicator_meta.items() if isinstance(v, (str, int, float, bool))})
    report["regime_split"]["raw_" + regime_indicator_name] = macro_indicator_raw.tolist()
    # Save Date column for downstream plotting
    if "Date" in oof_valid_df:
        report["_dates"] = oof_valid_df["Date"].tolist()
    report["llm_briefing"] = build_llm_briefing(run_dir, shap_max_samples=shap_max_samples)

    if save_report:
        config_dir = run_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        report_path = config_dir / "dm_test_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        report["report_path"] = str(report_path)

    return report


def print_report(report: dict[str, Any]) -> None:
    print("\n=== Diebold-Mariano Test (Model vs Baseline) ===")
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
    print()
    for comp in report.get("baselines_compared", []):
        bl_name = comp["name"]
        # Print baseline description
        if bl_name in _BASELINE_DESCRIPTIONS:
            print(_BASELINE_DESCRIPTIONS[bl_name].lstrip())
        for name, key in [("Brier", "dm_brier"), ("Log-Loss", "dm_log_loss")]:
            dm = comp[key]
            print(f"\nDM ({name}) — {bl_name}:")
            print(f"  DM stat           : {dm['dm_stat']:.4f}")
            print(f"  Mean loss diff    : {dm['mean_loss_diff']:.4f}")
            print(f"  Variance estimator: {dm.get('variance_estimator', 'unknown')}")
            print(f"  HAC lag           : {dm.get('hac_lag', 0)}")
            print(f"  p-value (one-sided): {dm['p_value_one_sided']:.4f}")
    for note in report.get("baseline_notes", []):
        print(f"\nBaseline note: {note}")

    # Conclusion based on first baseline entry
    dm_brier = report.get("dm_brier", {})
    if dm_brier.get("p_value_one_sided", 1) < 0.05 and dm_brier.get("mean_loss_diff", 0) > 0:
        print(
            "\nConclusion: Model significantly outperforms the first baseline on Brier "
            "score (DM test, p < 0.05); improvement is unlikely due to chance alone."
        )
    else:
        print(
            "\nConclusion: No statistically significant advantage over the first "
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
