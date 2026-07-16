from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from project.src.data.spliter import TimeSeriesSplitWithPurge
from project.src.eval.data import load_run_data
from project.src.eval.llm_briefing import build_llm_briefing
from project.src.eval.regime import (
    REGIME_METHOD_NAMES,
    build_macro_indicator,
    build_macro_regime_labels,
    group_confusion_matrices,
    ordered_group_keys,
)
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
    "rv_persist": ("rv_hist_persistence_baseline", "rv_hist_5d", "rv_hist_5d_state_persistence skipped: rv_hist_5d column not found in OOF evaluation data."),
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


def _build_one_regime_split(
    oof_valid_df,
    pipeline_config: dict[str, Any],
    *,
    method: str,
    n_regimes: int,
    clusterer: Any | None = None,
    macro_indicator=None,
    indicator_name: str | None = None,
    indicator_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit one clustering method and return a serialisable regime_split dict."""
    if macro_indicator is None or indicator_name is None or indicator_meta is None:
        regimes, indicator_name, indicator_meta, macro_indicator, cluster_meta = (
            build_macro_regime_labels(
                oof_valid_df,
                pipeline_config,
                ma_window=30,
                method=method,
                n_regimes=n_regimes,
                clusterer=clusterer,
            )
        )
    else:
        from project.src.eval.regime import assign_regime_clusters

        regimes, cluster_meta = assign_regime_clusters(
            macro_indicator.to_numpy(dtype=float),
            method=method,
            n_regimes=n_regimes,
            clusterer=clusterer,
            index=oof_valid_df.index,
        )

    df = oof_valid_df.copy()
    df["Regime"] = regimes
    gcm = group_confusion_matrices(
        df,
        group_col="Regime",
        y_col="y",
        proba_col="model_proba",
        threshold=0.5,
    )
    # Overall CM alongside per-cluster matrices
    overall_cm = group_confusion_matrices(
        df.assign(Regime="Overall"),
        group_col="Regime",
        y_col="y",
        proba_col="model_proba",
        threshold=0.5,
    ).get("Overall")
    if overall_cm is not None:
        ordered_gcm: dict[str, Any] = {"Overall": overall_cm}
        for key in ordered_group_keys(gcm):
            ordered_gcm[key] = gcm[key]
        gcm = ordered_gcm

    split: dict[str, Any] = {
        "macro_indicator": indicator_name,
        "macro_indicator_details": indicator_meta,
        "method": cluster_meta.get("method", method),
        "method_display": cluster_meta.get("method_display", method),
        "n_regimes": int(cluster_meta.get("n_regimes", n_regimes)),
        "threshold_rule": cluster_meta.get("threshold_rule", "cluster_mean_descending"),
        "cluster_order": cluster_meta.get("cluster_order", []),
        "cluster_means": cluster_meta.get("cluster_means", {}),
        "group_confusion_matrices": gcm,
        "_Regime": regimes.tolist(),
        "raw_" + indicator_name: macro_indicator.tolist(),
    }
    # Flatten a few scalar indicator flags for convenience
    for key, value in indicator_meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            split[key] = value
    # Optional algo-specific extras (edges, raw means)
    for key in ("quantile_edges", "raw_means", "raw_centers", "clusterer_type"):
        if key in cluster_meta:
            split[key] = cluster_meta[key]
    return split


def run_dm_evaluation(
    run_dir: Path,
    *,
    save_report: bool = True,
    baselines: list[str] | None = None,
    shap_max_samples: int | None = None,
    regime_method: str = "kmeans",
    n_regimes: int = 2,
    clusterer: Any | None = None,
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

    if n_regimes < 2:
        raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")

    method_key = (regime_method or "kmeans").strip().lower()
    if clusterer is not None:
        methods_to_run = ["custom"]
    elif method_key == "all":
        methods_to_run = list(REGIME_METHOD_NAMES)
    elif method_key in REGIME_METHOD_NAMES:
        methods_to_run = [method_key]
    else:
        raise ValueError(
            f"Unknown regime_method '{regime_method}'. "
            f"Valid: {list(REGIME_METHOD_NAMES) + ['all']}"
        )

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
        "regime_method": method_key if clusterer is None else "custom",
        "n_regimes": int(n_regimes),
        # ── fields for programmatic access / plotting ──
        "_y_valid": y_valid.tolist(),
        "_model_proba": model_proba.tolist(),
    }

    # Build macro once; reuse across methods when running `all`
    macro_indicator, indicator_name, indicator_meta = build_macro_indicator(
        oof_valid_df, pipeline_config, ma_window=30
    )

    regime_splits: dict[str, Any] = {}
    for m in methods_to_run:
        split = _build_one_regime_split(
            oof_valid_df,
            pipeline_config,
            method="kmeans" if m == "custom" else m,
            n_regimes=n_regimes,
            clusterer=clusterer if m == "custom" else None,
            macro_indicator=macro_indicator,
            indicator_name=indicator_name,
            indicator_meta=indicator_meta,
        )
        # When custom, method inside split is already "custom"
        store_key = split.get("method", m)
        regime_splits[store_key] = split

    report["regime_splits"] = regime_splits
    # Backward-compatible primary split (first method; prefer kmeans if present)
    primary_key = (
        "kmeans"
        if "kmeans" in regime_splits
        else next(iter(regime_splits))
    )
    report["regime_split"] = regime_splits[primary_key]
    report["primary_regime_method"] = primary_key

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

    regime_splits = report.get("regime_splits") or {}
    if not regime_splits and report.get("regime_split"):
        primary = report.get("primary_regime_method") or report["regime_split"].get("method", "regime")
        regime_splits = {primary: report["regime_split"]}

    for method_key, split in regime_splits.items():
        display = split.get("method_display", method_key)
        n_reg = split.get("n_regimes", report.get("n_regimes", "?"))
        print(f"\n=== Regime-based Confusion Matrices — {display} (n={n_reg}) ===")
        print(f"Macro indicator: {split.get('macro_indicator')}")
        print(f"Method         : {display} | rule={split.get('threshold_rule')}")
        means = split.get("cluster_means") or {}
        if means:
            means_str = ", ".join(f"{k}={v:.4f}" for k, v in means.items())
            print(f"Cluster means  : {means_str}")
        macro_meta = split.get("macro_indicator_details", {})
        if macro_meta.get("used_fallback"):
            print(
                "Macro indicator note: fallback was used because external ^VIX data was unavailable "
                f"or unalignable ({macro_meta.get('reason', 'unknown reason')})."
            )
        gcm = split.get("group_confusion_matrices", {})
        for regime in ordered_group_keys(gcm):
            stats = gcm[regime]
            mean_note = ""
            if regime in means:
                mean_note = f", mean={means[regime]:.4f}"
            print(f"\n{regime}: n={stats['n_samples']}{mean_note}")
            print(f"  confusion_matrix [[tn, fp], [fn, tp]] = {stats['confusion_matrix']}")

    llm = report.get("llm_briefing", {})
    if llm:
        print("\n=== LLM Briefing ===")
        print(f"Source       : {llm.get('source', 'unknown')}")
        if llm.get("error"):
            print(f"LLM status   : fallback triggered ({llm['error']})")
        print(f"Final brief  : {llm.get('final_brief', '')}")
