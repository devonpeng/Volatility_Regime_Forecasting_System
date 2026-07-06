from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from project.src.features.rv import compute_forward_rv
from project.src.features.steps import (
    add_bb_width,
    add_day_of_week,
    add_dist_from_ma50,
    add_downside_semivar,
    add_dxy_change_1d,
    add_ewma_rv,
    add_garman_klass_rv,
    add_gold_ret_1d,
    add_hl_range,
    add_intraday_ret,
    add_is_month_end,
    add_is_quarter_end,
    add_max_drawdown_20d,
    add_overnight_ret,
    add_overnight_vol_5d,
    add_ret_autocorr_1_20d,
    add_ret_1d,
    add_ret_kurt_20d,
    add_ret_skew_20d,
    add_rogers_satchell_rv,
    add_rv_acceleration,
    add_rv_hist_20d,
    add_rv_hist_5d,
    add_skew_index_level,
    add_vol_of_vol,
    add_volume_zscore,
    add_parkinson_rv,
    add_abs_ret_lag1,
    add_rv_ratio_5_20,
    add_vix_level,
    add_vix_rv_spread,
    add_vix_term_structure,
    add_vvix_level,
    add_vix_change_1d,
    add_yang_zhang_rv,
    add_yield_change_1d,
    add_qqq_ret_20d,
    add_qqq_rv_20d,
    add_qqq_minus_spy_ret_20d,
    add_vix_change_5d,
    add_vix_zscore_20d,
    add_max_drawdown_60d,
    add_tlt_ret_20d,
    add_irx_level,
)
FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class FeatureStep:
    """A single feature step: name + function that adds one column."""

    name: str
    fn: FeatureFn


DEFAULT_FEATURE_STEPS: list[FeatureStep] = [
    FeatureStep("ret_1d", add_ret_1d),
    FeatureStep("hl_range", add_hl_range),
    FeatureStep("rv_hist_5d", add_rv_hist_5d),
    FeatureStep("rv_hist_20d", add_rv_hist_20d),
    FeatureStep("gk_rv", add_garman_klass_rv),
    FeatureStep("rs_rv", add_rogers_satchell_rv),
    FeatureStep("yz_rv", add_yang_zhang_rv),
    FeatureStep("volume_zscore", add_volume_zscore),
    FeatureStep("parkinson_rv", add_parkinson_rv),
    FeatureStep("ewma_rv", add_ewma_rv),
    FeatureStep("downside_rv", add_downside_semivar),
    FeatureStep("vol_of_vol", add_vol_of_vol),
    FeatureStep("rv_accel", add_rv_acceleration),
    FeatureStep("ret_skew_20d", add_ret_skew_20d),
    FeatureStep("ret_kurt_20d", add_ret_kurt_20d),
    FeatureStep("max_drawdown_20d", add_max_drawdown_20d),
    FeatureStep("ret_autocorr_1_20d", add_ret_autocorr_1_20d),
    FeatureStep("abs_ret_lag1", add_abs_ret_lag1),
    FeatureStep("rv_ratio_5_20", add_rv_ratio_5_20),
    FeatureStep("vix_level", add_vix_level),
    FeatureStep("vix_change_1d", add_vix_change_1d),
    FeatureStep("vix_term_structure", add_vix_term_structure),
    FeatureStep("vix_rv_spread", add_vix_rv_spread),
    FeatureStep("vvix_level", add_vvix_level),
    FeatureStep("skew_index_level", add_skew_index_level),
    FeatureStep("yield_change_1d", add_yield_change_1d),
    FeatureStep("dxy_change_1d", add_dxy_change_1d),
    FeatureStep("gold_ret_1d", add_gold_ret_1d),
    FeatureStep("day_of_week", add_day_of_week),
    FeatureStep("is_month_end", add_is_month_end),
    FeatureStep("is_quarter_end", add_is_quarter_end),
    FeatureStep("dist_from_ma50", add_dist_from_ma50),
    FeatureStep("bb_width", add_bb_width),
    FeatureStep("overnight_ret", add_overnight_ret),
    FeatureStep("intraday_ret", add_intraday_ret),
    FeatureStep("overnight_vol_5d", add_overnight_vol_5d),
    FeatureStep("qqq_ret_20d", add_qqq_ret_20d),
    FeatureStep("qqq_rv_20d", add_qqq_rv_20d),
    FeatureStep("qqq_minus_spy_ret_20d", add_qqq_minus_spy_ret_20d),
    FeatureStep("vix_change_5d", add_vix_change_5d),
    FeatureStep("vix_zscore_20d", add_vix_zscore_20d),
    FeatureStep("max_drawdown_60d", add_max_drawdown_60d),
    FeatureStep("tlt_ret_20d", add_tlt_ret_20d),
    FeatureStep("irx_level", add_irx_level),
]

FEATURE_REGISTRY: dict[str, FeatureStep] = {
    step.name: step for step in DEFAULT_FEATURE_STEPS
}


def register_feature(step: FeatureStep) -> None:
    """Register a custom feature step for pipeline extension."""
    FEATURE_REGISTRY[step.name] = step


def get_feature_steps(names: list[str] | None = None) -> list[FeatureStep]:
    if names is None:
        return list(DEFAULT_FEATURE_STEPS)
    return [FEATURE_REGISTRY[name] for name in names]


def apply_feature(df: pd.DataFrame, step: FeatureStep) -> pd.DataFrame:
    """Apply one feature step to the DataFrame."""
    if step.name in df.columns:
        raise ValueError(f"Column already exists: {step.name}")
    return step.fn(df)


def apply_features(
    df: pd.DataFrame,
    steps: list[FeatureStep] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply feature steps one by one. Returns enriched df and column names."""
    data = df.copy()
    steps = steps or DEFAULT_FEATURE_STEPS
    feature_names: list[str] = []

    for step in steps:
        data = apply_feature(data, step)
        feature_names.append(step.name)

    return data, feature_names


def add_target_and_label(
    df: pd.DataFrame,
    horizon: int = 5,
    label_method: str = "expanding",
) -> tuple[pd.DataFrame, str]:
    """
    Add forward RV target column and binary label y.

    label_method:
        - "rolling": label vs rolling window median of forward RV
        - "expanding": label vs expanding median of forward RV
    """
    if "ret_1d" not in df.columns:
        raise ValueError("ret_1d must exist before building labels; add ret_1d feature first")

    data = df.copy()
    target_col = f"rv_fwd_{horizon}d"
    data[target_col] = compute_forward_rv(data["ret_1d"], horizon=horizon)

    if label_method == "expanding":
        threshold = data[target_col].expanding(min_periods=horizon).median()
    else:
        threshold = data[target_col].rolling(window=horizon * 10).median()

    data["y"] = (data[target_col] > threshold).astype(int)
    return data, target_col


def build_dataset(
    df: pd.DataFrame,
    horizon: int = 5,
    label_method: str ="expanding",
    feature_steps: list[FeatureStep] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build feature matrix X, binary label y, and metadata for plotting."""
    data, feature_names = apply_features(df, steps=feature_steps)
    data, target_col = add_target_and_label(data, horizon=horizon, label_method=label_method)

    meta_cols = ["Date", "Close", target_col, "y"]
    if "Date" not in data.columns:
        meta_cols = [c for c in meta_cols if c != "Date"]

    required_cols = feature_names + [target_col, "y"]
    valid = data[required_cols].notna().all(axis=1)
    data = data.loc[valid].reset_index(drop=True)

    X = data[feature_names].copy()
    y = data["y"].copy()
    meta_df = data[meta_cols].copy()
    return X, y, meta_df
