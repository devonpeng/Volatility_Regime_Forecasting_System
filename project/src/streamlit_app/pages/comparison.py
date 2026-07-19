"""Station 5: Comparison — Quick Browse + Custom Experiment + Eval Deep-dive."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
import yaml

from project.src.pipline import RVPipeline
from project.src.streamlit_app.ai_tutor import render_analyze_button
from project.src.streamlit_app.eval_ui import (
    actual_rv_aligned_to_dates,
    ensure_run_dir,
    holdout_predictions_from_run,
    load_eval_report,
    oof_series_from_report,
    regime_bands_aligned_to_dates,
)
from project.src.streamlit_app.plots import (
    CAPTION_BUBBLE,
    CAPTION_CLUSTER_SCATTER,
    CAPTION_CM,
    CAPTION_PRED_VS_ACTUAL,
    CAPTION_RADAR,
    CAPTION_REGIME_TIMELINE,
    dm_baselines_dataframe,
    fig_bubble_scatter,
    fig_cluster_scatter,
    fig_confusion_matrices,
    fig_macro_regime_timeline,
    fig_metrics_radar,
    fig_pred_vs_actual,
)
from project.src.streamlit_app.precomputed import (
    PROJECT_ROOT,
    date_range_presets,
    extract_xy_from_wide,
    grid_results_path,
    relative_run_dir,
    run_dir_for_experiment,
    slice_by_dates,
)
from project.src.streamlit_app.session_manager import (
    SS_CUSTOM_EXPERIMENTS,
    SS_FULL_DF,
    SS_HORIZON,
    SS_LABEL_METHOD,
    SS_SYMBOL,
    SS_TRAIN_RATIO,
    is_parquet_workspace,
)
from project.src.streamlit_app.workspace import checkpoint_current_workspace

VIEW_PATH = PROJECT_ROOT / "configs" / "comparison_view.yaml"


def _load_view_cfg() -> dict:
    if VIEW_PATH.is_file():
        return yaml.safe_load(VIEW_PATH.read_text(encoding="utf-8")) or {}
    return {}


def _load_grid_df() -> pd.DataFrame:
    path = grid_results_path()
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _combine_browse(grid: pd.DataFrame, custom: list[dict]) -> pd.DataFrame:
    frames = []
    if not grid.empty:
        g = grid.copy()
        if "source" not in g.columns:
            g["source"] = "grid"
        frames.append(g)
    if custom:
        frames.append(pd.DataFrame(custom))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _render_deep_dive(
    row: dict,
    full_df: pd.DataFrame,
    *,
    mode: str,
    key_suffix: str,
) -> None:
    """Render one experiment's full eval deep-dive (no shared controls)."""
    exp_id = str(row.get("experiment_id") or key_suffix)
    st.markdown(f"#### Deep-dive: `{exp_id}`")
    train_ratio = float(st.session_state.get(SS_TRAIN_RATIO, 0.8))
    horizon = int(row.get("horizon") or st.session_state.get(SS_HORIZON, 5))

    cache_key = f"_dd_report_{exp_id}"
    run_cache_key = f"_dd_run_{exp_id}"

    report = st.session_state.get(cache_key)
    run_dir_str = st.session_state.get(run_cache_key)
    if report is None or not run_dir_str:
        st.info("Click **Load / Refresh Deep-dive** above to compute this report.")
        return

    from pathlib import Path

    run_dir = Path(run_dir_str)

    model_m = report.get("model") or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Model Brier", f"{model_m.get('brier', float('nan')):.4f}")
    c2.metric("Model Log-Loss", f"{model_m.get('log_loss', float('nan')):.4f}")
    c3.metric("OOF n", f"{report.get('n_oof', '—')}")

    st.markdown("**Diebold–Mariano vs baselines**")
    dm_df = dm_baselines_dataframe(report)
    st.dataframe(dm_df, use_container_width=True)
    notes = report.get("baseline_notes") or []
    if notes:
        with st.expander(f"Baseline notes ({exp_id})"):
            for n in notes:
                st.caption(str(n))

    split = report.get("regime_split") or {}
    st.plotly_chart(
        fig_confusion_matrices(report, split),
        use_container_width=True,
        key=f"dd_cm_{key_suffix}",
    )
    st.caption(CAPTION_CM)
    st.plotly_chart(
        fig_macro_regime_timeline(report, split),
        use_container_width=True,
        key=f"dd_regime_{key_suffix}",
    )
    st.caption(CAPTION_REGIME_TIMELINE)
    st.plotly_chart(
        fig_cluster_scatter(report, split),
        use_container_width=True,
        key=f"dd_cluster_{key_suffix}",
    )
    st.caption(CAPTION_CLUSTER_SCATTER)

    st.markdown("**Predictions vs Actual**")
    try:
        date_start = str(row.get("start") or "")
        date_end = str(row.get("end") or "")
        if mode == "eval":
            dates, y, proba = oof_series_from_report(report)
            title = f"Eval (OOF): Predictions vs Actual — {exp_id}"
            regimes = np.asarray(split.get("_Regime", []), dtype=object)
            if len(regimes) == len(proba) and len(regimes) > 0:
                regime_labels = regimes
                cluster_order = split.get("cluster_order")
            else:
                regime_labels, cluster_order = regime_bands_aligned_to_dates(
                    dates,
                    full_df=full_df,
                    start_date=date_start,
                    end_date=date_end,
                )
        else:
            dates, y, proba = holdout_predictions_from_run(
                run_dir, train_ratio=train_ratio
            )
            title = (
                f"Test holdout (train_ratio={train_ratio:.2f}): "
                f"Predictions vs Actual — {exp_id}"
            )
            regime_labels, cluster_order = regime_bands_aligned_to_dates(
                dates,
                full_df=full_df,
                start_date=date_start,
                end_date=date_end,
            )

        actual_rv = actual_rv_aligned_to_dates(
            dates, horizon=horizon, full_df=full_df
        )
        fig_labels, fig_proba = fig_pred_vs_actual(
            dates,
            y,
            proba,
            title=title,
            actual_rv=actual_rv,
            regime_labels=regime_labels,
            cluster_order=cluster_order,
        )
        st.plotly_chart(
            fig_labels,
            use_container_width=True,
            key=f"dd_pred_labels_{key_suffix}",
        )
        st.plotly_chart(
            fig_proba,
            use_container_width=True,
            key=f"dd_pred_proba_{key_suffix}",
        )
        st.caption(CAPTION_PRED_VS_ACTUAL)
    except Exception as exc:
        st.warning(f"Pred vs actual unavailable: {exc}")


def render() -> None:
    st.header("Station 5: Comparison")

    if not is_parquet_workspace():
        st.warning(
            "Comparison requires a Parquet-backed teaching-pack workspace. "
            "Use Station 1 → Load Teaching Pack (avoid the live escape hatch)."
        )
        return

    full_df = st.session_state.get(SS_FULL_DF)
    if full_df is None:
        st.warning("Load a teaching pack in Station 1 first.")
        return

    view = _load_view_cfg()
    max_compare = int(view.get("max_compare", 5))
    presets = date_range_presets()

    # ── Quick Browse ──────────────────────────────────────────────────
    st.subheader("Quick Browse")
    grid = _load_grid_df()
    custom = list(st.session_state.get(SS_CUSTOM_EXPERIMENTS) or [])
    browse = _combine_browse(grid, custom)

    if browse.empty:
        st.info(
            "No grid results yet. Generate with:\n\n"
            "`uv run python scripts/precompute_grid.py`\n\n"
            "You can still run a Custom Experiment below."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        def _opts(col: str) -> list:
            vals = sorted({str(v) for v in browse[col].dropna().unique()}) if col in browse.columns else []
            return ["all", *vals]

        with c1:
            f_sym = st.selectbox("Symbol", _opts("symbol"), key="qb_sym")
        with c2:
            f_hrz = st.selectbox("Horizon", _opts("horizon"), key="qb_hrz")
        with c3:
            f_sel = st.selectbox("Selection", _opts("feature_selection"), key="qb_sel")
        with c4:
            f_model = st.selectbox("Model", _opts("model"), key="qb_model")
        with c5:
            f_dates = st.selectbox("Date Range", _opts("date_range"), key="qb_dates")
        with c6:
            f_topk = st.selectbox("Top-K", _opts("top_k"), key="qb_topk")

        filtered = browse.copy()
        for col, val in [
            ("symbol", f_sym),
            ("horizon", f_hrz),
            ("feature_selection", f_sel),
            ("model", f_model),
            ("date_range", f_dates),
            ("top_k", f_topk),
        ]:
            if val != "all" and col in filtered.columns:
                filtered = filtered[filtered[col].astype(str) == str(val)]

        sort_cfg = view.get("default_sort") or {"column": "auc", "ascending": False}
        sort_col = sort_cfg.get("column", "auc")
        if sort_col in filtered.columns:
            filtered = filtered.sort_values(
                sort_col, ascending=bool(sort_cfg.get("ascending", False))
            )

        display_cols = [
            c["field"] for c in view.get("columns", []) if c.get("field") in filtered.columns
        ]
        if not display_cols:
            display_cols = list(filtered.columns)
        # Prefer showing source + run_dir when present
        for extra in ("source", "run_dir"):
            if extra in filtered.columns and extra not in display_cols:
                display_cols.append(extra)

        st.dataframe(filtered[display_cols], use_container_width=True)
        if "source" in filtered.columns and (filtered["source"] == "custom").any():
            st.caption("Rows with `source=custom` come from Custom Experiment runs this session.")

        id_col = "experiment_id" if "experiment_id" in filtered.columns else filtered.columns[0]
        options = filtered[id_col].astype(str).tolist()
        selected_ids = st.multiselect(
            f"Select rows for metrics compare (max {max_compare})",
            options=options,
            key="qb_compare_ids",
        )
        if len(selected_ids) > max_compare:
            st.warning(f"Using first {max_compare} of {len(selected_ids)} selections.")
            selected_ids = selected_ids[:max_compare]
        selected_rows = filtered[filtered[id_col].astype(str).isin(selected_ids)]

        if len(filtered) > 0:
            st.plotly_chart(
                fig_bubble_scatter(filtered),
                use_container_width=True,
                key="qb_bubble",
            )
            st.caption(CAPTION_BUBBLE)
        if len(selected_rows) > 0:
            radar_metrics = view.get("radar_metrics") or ["auc", "brier", "log_loss"]
            st.plotly_chart(
                fig_metrics_radar(selected_rows, radar_metrics),
                use_container_width=True,
                key="qb_radar",
            )
            st.caption(CAPTION_RADAR)
            st.session_state["compared_experiments"] = selected_rows.to_dict(orient="records")

        # Deep-dive: up to 3 experiments, each full stack
        st.subheader("Eval Deep-dive")
        max_deep = 1
        auto_id = st.session_state.pop("_dd_auto_id", None)
        if auto_id and auto_id in options:
            prev = list(st.session_state.get("qb_deep_ids") or [])
            if auto_id not in prev:
                st.session_state["qb_deep_ids"] = ([auto_id] + prev)[:max_deep]
        deep_ids = st.multiselect(
            f"Deep-dive experiments (max {max_deep})",
            options=options,
            key="qb_deep_ids",
        )
        if len(deep_ids) > max_deep:
            st.warning(f"Using first {max_deep} of {len(deep_ids)} selections.")
            deep_ids = deep_ids[:max_deep]

        mode = st.radio(
            "Prediction mode",
            options=["eval", "test"],
            format_func=lambda m: "Eval (OOF)" if m == "eval" else "Test (holdout)",
            horizontal=True,
            index=0,
            key="dd_pred_mode",
        )

        if deep_ids and st.button(
            "Load / Refresh Deep-dive", type="primary", key="dd_load_btn"
        ):
            with st.spinner("Resolving run artifacts & computing eval reports..."):
                for did in deep_ids:
                    deep_rows = filtered[filtered[id_col].astype(str) == did]
                    if deep_rows.empty:
                        deep_rows = browse[browse[id_col].astype(str) == did]
                    if deep_rows.empty:
                        st.error(f"Experiment not found: {did}")
                        continue
                    row = deep_rows.iloc[0].to_dict()
                    try:
                        run_dir = ensure_run_dir(row, full_df)
                        report = load_eval_report(run_dir)
                        st.session_state[f"_dd_run_{did}"] = str(run_dir)
                        st.session_state[f"_dd_report_{did}"] = report
                        st.success(f"Loaded {did}: {run_dir.name}")
                    except Exception as exc:
                        st.error(f"Deep-dive failed for {did}: {exc}")

        for did in deep_ids:
            deep_rows = filtered[filtered[id_col].astype(str) == did]
            if deep_rows.empty:
                deep_rows = browse[browse[id_col].astype(str) == did]
            if deep_rows.empty:
                continue
            st.markdown("---")
            _render_deep_dive(
                deep_rows.iloc[0].to_dict(),
                full_df,
                mode=mode,
                key_suffix=did,
            )

    # ── Custom Experiment ─────────────────────────────────────────────
    with st.expander("Custom Experiment", expanded=False):
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)

        with r1c1:
            st.markdown("**Time range**")
            preset_keys = list(presets.keys()) + ["custom"]
            date_choice = st.radio(
                "Preset",
                options=preset_keys,
                horizontal=True,
                key="ce_date_preset",
                label_visibility="collapsed",
            )
            dmin = pd.to_datetime(full_df["Date"]).min().date()
            dmax = pd.to_datetime(full_df["Date"]).max().date()
            if date_choice == "custom":
                dc1, dc2 = st.columns(2)
                with dc1:
                    c_start = st.date_input(
                        "Start",
                        value=max(dmin, pd.Timestamp("2010-01-01").date()),
                        min_value=dmin,
                        max_value=dmax,
                        key="ce_start",
                    )
                with dc2:
                    c_end = st.date_input(
                        "End",
                        value=min(dmax, pd.Timestamp("2024-12-31").date()),
                        min_value=dmin,
                        max_value=dmax,
                        key="ce_end",
                    )
                date_range_key = "custom"
                start_s = c_start.strftime("%Y-%m-%d")
                end_s = c_end.strftime("%Y-%m-%d")
            else:
                start_s = presets[date_choice]["start"]
                end_s = presets[date_choice]["end"]
                date_range_key = date_choice
                st.caption(f"`{date_choice}`: {start_s} → {end_s}")

        with r1c2:
            st.markdown("**Feature selection**")
            fs_method = st.selectbox(
                "Method",
                ["none", "lasso", "tree_topk", "ttest"],
                index=1,
                key="ce_fs",
            )
            top_k = st.number_input("Top-K", 3, 40, 12, key="ce_topk")
            nested = st.checkbox("Nested per-fold", value=False, key="ce_nested")

        with r2c1:
            st.markdown("**Model & tuning**")
            models = st.multiselect(
                "Models",
                ["lgbm", "xgb", "logistic"],
                default=["lgbm"],
                key="ce_models",
            )
            n_trials = st.number_input("n_trials", 5, 100, 20, key="ce_trials")
            early_stop = st.number_input("early_stop", 1, 20, 5, key="ce_early")

        with r2c2:
            st.markdown("**Cross-validation**")
            n_splits = st.number_input("n_splits", 3, 20, 12, key="ce_ns")
            train_window = st.number_input("train_window", 63, 1008, 252, key="ce_tw")
            test_size = st.number_input("test_size", 5, 126, 21, key="ce_ts")
            gap = st.number_input("gap", 0, 63, 5, key="ce_gap")

        if st.button("Run Custom Comparison", type="primary"):
            if not models:
                st.warning("Select at least one model.")
            elif start_s > end_s:
                st.warning("Invalid date range.")
            else:
                with st.spinner("Running RVPipeline..."):
                    try:
                        symbol = st.session_state.get(SS_SYMBOL, "SPY")
                        horizon = int(st.session_state.get(SS_HORIZON, 5))
                        label_method = st.session_state.get(SS_LABEL_METHOD, "expanding")
                        sliced = slice_by_dates(full_df, start_s, end_s)
                        X, y, meta_df, _ = extract_xy_from_wide(
                            sliced,
                            horizon=horizon,
                            label_method=label_method,
                        )
                        fs = None if fs_method == "none" else fs_method
                        exp_id = (
                            f"custom_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
                        )
                        pipe = RVPipeline(
                            symbol=symbol,
                            start_date=start_s,
                            end_date=end_s,
                            horizon=horizon,
                            models=list(models),
                            n_trials=int(n_trials),
                            label_method=label_method,
                            feature_selection=fs,
                            nested_feature_selection=bool(nested),
                            top_k_features=int(top_k),
                            n_splits=int(n_splits),
                            train_window_size=int(train_window),
                            test_size=int(test_size),
                            gap_size=int(gap),
                            early_stop_patience=int(early_stop),
                        )
                        pipe.persist_artifacts = False
                        pipe.run_id = exp_id
                        pipe.inject_dataset(X, y, meta_df, raw_df=sliced)
                        result = pipe.run(persist=False)

                        run_path = run_dir_for_experiment(exp_id)
                        if run_path.exists():
                            import shutil

                            shutil.rmtree(run_path)
                        pipe.persist_run_artifacts(run_path)

                        metrics = result.oof_metrics or {}
                        row = {
                            "experiment_id": exp_id,
                            "symbol": symbol,
                            "horizon": horizon,
                            "date_range": date_range_key,
                            "start": start_s,
                            "end": end_s,
                            "feature_selection": fs_method,
                            "top_k": int(top_k),
                            "model": result.best_model_name,
                            "cv_profile": "custom",
                            "n_splits": int(n_splits),
                            "train_window": int(train_window),
                            "test_size": int(test_size),
                            "gap": int(gap),
                            "auc": float(result.best_auc),
                            "brier": float(metrics.get("brier_score", float("nan"))),
                            "log_loss": float(metrics.get("log_loss", float("nan"))),
                            "best_params": json.dumps(pipe.best_params or {}),
                            "selected_features": json.dumps(
                                list(pipe.selected_features or [])
                            ),
                            "source": "custom",
                            "run_dir": relative_run_dir(run_path),
                        }
                        custom_list = list(
                            st.session_state.get(SS_CUSTOM_EXPERIMENTS) or []
                        )
                        custom_list.insert(0, row)
                        st.session_state[SS_CUSTOM_EXPERIMENTS] = custom_list
                        st.session_state["_custom_last"] = row
                        st.session_state["_dd_auto_id"] = exp_id
                        checkpoint_current_workspace("custom experiment")
                        st.success(
                            f"Done: {result.best_model_name} AUC={result.best_auc:.4f}"
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Custom run failed: {exc}")

        last = st.session_state.get("_custom_last")
        if last:
            st.markdown("**Latest custom result**")
            st.json(last)

    st.subheader("AI Analysis")
    ctx = {
        "browse_head": (browse.head(10).to_dict(orient="records") if not browse.empty else []),
        "custom": st.session_state.get(SS_CUSTOM_EXPERIMENTS) or [],
        "compared": st.session_state.get("compared_experiments") or [],
    }
    render_analyze_button(ctx, key="compare_ai")
