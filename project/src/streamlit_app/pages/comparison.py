"""Station 5: Feature Comparison — Reference demo of LASSO top-k results."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.header("📐 Feature Count Comparison (Reference)")

    st.markdown(
        "Same pipeline parameters, varying LASSO ``top_k`` — how feature count "
        "affects AUC, Brier, and Log Loss."
    )

    # ── 1. Fixed parameters ────────────────────────────────────────────
    st.subheader("1. Fixed Parameters")
    st.markdown(
        "| Parameter | Value |\n"
        "|---|---|\n"
        "| Symbol / Date Range | SPY, 2010-01-01 ~ 2025-12-31 |\n"
        "| Horizon | 5 |\n"
        "| n_trials | 200 |\n"
        "| label_method | expanding |\n"
        "| feature_selection | lasso |\n"
        "| CV Config | 24-split, train_window=1008, test_size=126, gap=10 |\n"
        "| random_state | 42 |"
    )

    # ── 2. Feature selection table ─────────────────────────────────────
    st.subheader("2. Feature Selection")
    st.markdown(
        "| Feature | LASSO Score | top_k=10 | top_k=12 | top_k=20 |\n"
        "|:---|---:|:---:|:---:|:---:|\n"
        "| `parkinson_rv` | 1.319 | ✅ | ✅ | ✅ |\n"
        "| `vix_level` | 0.827 | ✅ | ✅ | ✅ |\n"
        "| `vix_term_structure` | 0.326 | ✅ | ✅ | ✅ |\n"
        "| `kalman_trend_medium` | 0.283 | ✅ | ✅ | ✅ |\n"
        "| `ret_1d` | 0.247 | ✅ | ✅ | ✅ |\n"
        "| `kalman_price_medium` | 0.239 | ✅ | ✅ | ✅ |\n"
        "| `kalman_ret_1d_medium` | 0.193 | ✅ | ✅ | ✅ |\n"
        "| `vix_rv_spread` | 0.169 | ✅ | ✅ | ✅ |\n"
        "| `rv_ratio_5_20` | 0.140 | ✅ | ✅ | ✅ |\n"
        "| `rv_accel` | 0.132 | ✅ | ✅ | ✅ |\n"
        "| `overnight_ret` | 0.129 | — | ✅ | ✅ |\n"
        "| `vix_change_1d` | 0.102 | — | ✅ | ✅ |\n"
        "| `vvix_level` | 0.096 | — | — | ✅ |\n"
        "| `ret_kurt_20d` | 0.091 | — | — | ✅ |\n"
        "| `har_rv_forecast_1d` | 0.077 | — | — | ✅ |\n"
        "| `is_month_end` | 0.074 | — | — | ✅ |\n"
        "| `ret_skew_20d` | 0.062 | — | — | ✅ |\n"
        "| `day_of_week` | 0.039 | — | — | ✅ |\n"
        "| `volume_zscore` | 0.022 | — | — | ✅ |\n"
        "| `hl_range` | 0.018 | — | — | ✅ |"
    )

    # ── 3. Results table ───────────────────────────────────────────────
    st.subheader("3. Result Metrics")
    st.markdown(
        "| Metric | top_k=10 | top_k=12 | top_k=20 |\n"
        "|:---|:---:|:---:|:---:|\n"
        "| **AUC** | 0.76985 | **0.77360** 🏆 | 0.76517 |\n"
        "| **Brier** ↓ | **0.16225** 🏆 | 0.16389 | 0.16499 |\n"
        "| **Log Loss** ↓ | **0.49123** 🏆 | 0.49353 | 0.49807 |"
    )

    # ── 4. Interpretation ──────────────────────────────────────────────
    st.subheader("4. Interpretation")

    st.markdown("##### AUC")
    st.markdown(
        "AUC exhibits an inverted-U shape, with **12 features at the peak**. "
        "Starting from 20 features, weak-signal predictors "
        "(``vvix_level`` through ``hl_range``, score 0.018–0.096) "
        "inject noise and degrade ranking power — AUC settles at 0.7652. "
        "Removing those 8 features (down to 12) largely eliminates the noise, "
        "lifting AUC to its peak of 0.7736 (+0.0084). "
        "Further trimming the two marginal features "
        "(``overnight_ret`` 0.129, ``vix_change_1d`` 0.102) "
        "to 10 features causes AUC to dip for the first time to 0.7699 (−0.0037). "
        "This suggests **an optimal feature density exists**: "
        "too many features add noise, too few discard useful signal."
    )

    st.markdown("##### Brier / Log Loss")
    st.markdown(
        "Brier Score and Log Loss decrease monotonically as features are pruned, "
        "indicating that all 10 eliminated features "
        "(from ``hl_range`` 0.018 up to ``overnight_ret`` 0.129) "
        "**contribute negatively or negligibly to probability calibration**. "
        "With fewer features, the model produces sharper, better-calibrated probabilities. "
        "This reveals a counter-intuitive pattern — not all LASSO-selected non-zero "
        "coefficients improve calibration. Even the 11th–12th ranked features "
        "(``overnight_ret``, ``vix_change_1d``), when removed, "
        "improve Brier from 0.1639 → 0.1623 (−0.0016) and "
        "Log Loss from 0.4935 → 0.4912 (−0.0023)."
    )

    st.markdown("##### Recommendation")
    st.success(
        "**12 features (LASSO score ≥ 0.10) represent the optimal trade-off**: "
        "AUC peaks at 0.7736 🏆 while Brier / Log Loss are already on a "
        "declining trajectory. Versus the 10-feature set, retaining 12 features "
        "preserves a **0.0037** AUC advantage (+0.48%); versus the 20-feature set, "
        "it delivers a **0.0084** AUC gain (+1.10%) with better calibration "
        "(Brier −0.0011, Log Loss −0.0045). These 12 features span three signal "
        "families — **Parkinson RV** (1.319), **3 VIX-linked macro variables** "
        "(0.169–0.827), and **3 Kalman-filter trend features** (0.193–0.283) — "
        "achieving the best balance between discriminative power and "
        "probability calibration."
    )
