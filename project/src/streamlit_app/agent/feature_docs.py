"""Static feature documentation for agent tools (sourced from README §5)."""
from __future__ import annotations

from typing import TypedDict


class FeatureDoc(TypedDict):
    role: str
    transform: str
    formula: str


# Keep in sync with README.md §5 "Feature 含义速查表".
FEATURE_DOCS: dict[str, FeatureDoc] = {
    "ret_1d": {
        "role": "Daily return; the most basic price-change signal, serves as the core input for most volatility-related features",
        "transform": "ratio",
        "formula": "Close.pct_change()",
    },
    "hl_range": {
        "role": "Intraday range; measures the price fluctuation band within a day, a simple proxy for short-term volatility",
        "transform": "ratio",
        "formula": "(High - Low) / Close",
    },
    "rv_hist_5d": {
        "role": "5-day historical realized volatility (short-term); captures the most recent volatility regime for RV prediction",
        "transform": "—",
        "formula": "annualized 5-day rolling std of ret_1d",
    },
    "rv_hist_20d": {
        "role": "20-day historical realized volatility (medium-term); a standard baseline feature/target reference for RV models",
        "transform": "—",
        "formula": "annualized 20-day rolling std of ret_1d",
    },
    "gk_rv": {
        "role": "Garman-Klass volatility; uses OHLC to reduce noise, offering a more efficient RV estimator than close-to-close",
        "transform": "log+ratio",
        "formula": "combination of log(H/L), log(C/O), rolling mean, annualized",
    },
    "rs_rv": {
        "role": "Rogers-Satchell volatility; more robust to trends, useful as a complementary RV estimator",
        "transform": "log+ratio",
        "formula": "combination of log(H/O), log(L/O), log(C/O), rolling mean, annualized",
    },
    "yz_rv": {
        "role": "Yang-Zhang volatility; accounts for overnight gaps, considered one of the most comprehensive RV estimators",
        "transform": "log+ratio",
        "formula": "combination of overnight/open-close/RS log-return components, annualized",
    },
    "volume_zscore": {
        "role": "Degree of volume abnormality (surge/shrink); volume shocks often precede or accompany volatility changes",
        "transform": "—",
        "formula": "z-score of Volume relative to 20-day mean and std",
    },
    "parkinson_rv": {
        "role": "Parkinson volatility; estimates volatility from the high-low range, a lightweight RV estimator",
        "transform": "log+ratio",
        "formula": "annualized rolling second moment of log(High/Low)",
    },
    "ewma_rv": {
        "role": "Volatility estimate weighting recent shocks more heavily, similar to RiskMetrics-style RV features",
        "transform": "—",
        "formula": "EWMA of ret_1d^2, then square-rooted and annualized",
    },
    "downside_rv": {
        "role": "Downside semi-variance; emphasizes negative-return risk, useful for asymmetric RV modeling",
        "transform": "—",
        "formula": "rolling mean of squared negative ret_1d only, square-rooted and annualized",
    },
    "vol_of_vol": {
        "role": "Volatility of volatility; identifies regime stability, a second-order feature for RV forecasting",
        "transform": "—",
        "formula": "20-day rolling std of rv_hist_5d",
    },
    "rv_accel": {
        "role": "Volatility acceleration; detects volatility regime shifts, a momentum-style feature for RV",
        "transform": "—",
        "formula": "rv_hist_5d - rv_hist_5d.shift(5)",
    },
    "ret_skew_20d": {
        "role": "Return skewness; identifies left-tail risk tendency, a distributional feature for RV models",
        "transform": "—",
        "formula": "20-day rolling skewness of ret_1d",
    },
    "ret_kurt_20d": {
        "role": "Return kurtosis; identifies fat-tail/extreme volatility risk, a distributional feature for RV models",
        "transform": "—",
        "formula": "20-day rolling kurtosis of ret_1d",
    },
    "max_drawdown_20d": {
        "role": "Deepest recent drawdown; measures tail pressure, often correlated with volatility spikes",
        "transform": "ratio",
        "formula": "Close / rolling_max - 1, then rolling minimum",
    },
    "ret_autocorr_1_20d": {
        "role": "Lag-1 autocorrelation of returns; reflects trending/mean-reverting regime, informative for RV persistence",
        "transform": "—",
        "formula": "lag=1 autocorrelation of ret_1d within a 20-day window",
    },
    "abs_ret_lag1": {
        "role": "Yesterday's absolute return; short-term shock memory, a simple lagged volatility proxy",
        "transform": "—",
        "formula": "abs(ret_1d).shift(1)",
    },
    "rv_ratio_5_20": {
        "role": "Relative strength of short- vs medium-term volatility; captures volatility term-structure shape",
        "transform": "volatility ratio",
        "formula": "rv_hist_5d / rv_hist_20d",
    },
    "vix_level": {
        "role": "Implied volatility level; a forward-looking market-wide volatility signal for RV prediction",
        "transform": "—",
        "formula": "^VIX close",
    },
    "vix_change_1d": {
        "role": "Daily change in implied volatility; captures shifts in market fear",
        "transform": "—",
        "formula": "vix.diff().shift(1)",
    },
    "vix_term_structure": {
        "role": "VIX term structure; identifies panic/backwardation regimes relevant to future RV",
        "transform": "—",
        "formula": "vix3m - vix",
    },
    "vix_rv_spread": {
        "role": "Implied-minus-realized volatility spread; a mean-reversion signal linking IV and RV",
        "transform": "—",
        "formula": "vix / 100 - rv_hist_20d",
    },
    "vvix_level": {
        "role": "\"Vol-of-vol\" level; reflects uncertainty about future volatility itself",
        "transform": "—",
        "formula": "^VVIX close",
    },
    "skew_index_level": {
        "role": "Tail-risk premium strength; a macro sentiment feature that can precede volatility spikes",
        "transform": "—",
        "formula": "^SKEW close",
    },
    "yield_change_1d": {
        "role": "Interest-rate shock proxy; macro driver that can spill over into equity volatility",
        "transform": "—",
        "formula": "tnx.diff().shift(1)",
    },
    "dxy_change_1d": {
        "role": "Dollar risk-appetite proxy; a cross-asset macro feature for RV models",
        "transform": "ratio",
        "formula": "dxy.pct_change().shift(1)",
    },
    "gold_ret_1d": {
        "role": "Safe-haven asset move proxy; captures risk-off flows that often coincide with volatility increases",
        "transform": "ratio",
        "formula": "gold.pct_change().shift(1)",
    },
    "day_of_week": {
        "role": "Day-of-week calendar effect; a seasonal control feature in RV models",
        "transform": "—",
        "formula": "trading day dayofweek",
    },
    "is_month_end": {
        "role": "Month-end effect flag; captures calendar-driven volatility patterns",
        "transform": "—",
        "formula": "whether date is month-end (0/1)",
    },
    "is_quarter_end": {
        "role": "Quarter-end effect flag; captures calendar-driven volatility patterns",
        "transform": "—",
        "formula": "whether date is quarter-end (0/1)",
    },
    "dist_from_ma50": {
        "role": "Degree of trend deviation; distance from a moving average, often linked to volatility clustering",
        "transform": "ratio",
        "formula": "(Close - MA50) / MA50",
    },
    "bb_width": {
        "role": "Bollinger Band width; a nonlinear representation of the volatility state",
        "transform": "ratio",
        "formula": "4 * rolling_std(Close,20) / rolling_mean(Close,20)",
    },
    "overnight_ret": {
        "role": "Overnight return; captures gap information, useful for decomposing overnight vs intraday volatility",
        "transform": "log+ratio",
        "formula": "log(Open / Close.shift(1))",
    },
    "intraday_ret": {
        "role": "Intraday return; isolates the intraday volatility driver from overnight effects",
        "transform": "log+ratio",
        "formula": "log(Close / Open)",
    },
    "overnight_vol_5d": {
        "role": "Overnight volatility (short-term); a component feature for decomposed RV modeling",
        "transform": "log+ratio",
        "formula": "annualized 5-day rolling std of overnight_ret",
    },
    "garch_vol_1d": {
        "role": "GARCH(1,1)-t one-step volatility forecast; a classical parametric baseline for RV prediction",
        "transform": "log",
        "formula": "rolling fit on log(Close).diff(), annualized, then shift(1)",
    },
    "gjr_garch_vol_1d": {
        "role": "GJR-GARCH one-step volatility; captures the leverage effect (asymmetric response to negative returns)",
        "transform": "log",
        "formula": "same as above, GARCH with o=1, annualized then shift(1)",
    },
    "egarch_vol_1d": {
        "role": "EGARCH one-step volatility; models asymmetric volatility response",
        "transform": "log",
        "formula": "same as above, EGARCH(1,1)-t, annualized then shift(1)",
    },
    "garch_vol_ratio_5_20": {
        "role": "Relative strength of GARCH forecast vs historical RV; a model-disagreement feature",
        "transform": "volatility ratio",
        "formula": "garch_vol_1d / rv_hist_20d",
    },
    "garch_rv_spread": {
        "role": "Spread between GARCH forecast and historical realized volatility; another model-disagreement feature",
        "transform": "—",
        "formula": "garch_vol_1d - rv_hist_20d",
    },
    "har_rv_forecast_1d": {
        "role": "HAR-RV one-step volatility forecast; a well-established heterogeneous-horizon RV forecasting feature",
        "transform": "log+ratio",
        "formula": "daily GK (log(H/L), log(C/O)) at d/w/m horizons, Scaled-Ridge, clipped >=0, then shift(1)",
    },
    "kalman_price_medium": {
        "role": "Kalman-denoised price level (medium horizon); a smoothed price series that reduces noise for RV modeling",
        "transform": "log",
        "formula": "filtering on log(Close), mapped back via exp, shift(1)",
    },
    "kalman_ret_1d_medium": {
        "role": "Denoised-price daily return; a lower-noise return series as an alternative RV input",
        "transform": "ratio",
        "formula": "pct_change().shift(1) of the mapped-back price",
    },
    "kalman_trend_medium": {
        "role": "Denoised trend component of state (log drift); captures the underlying trend driving volatility regimes",
        "transform": "log",
        "formula": "velocity term in log space, shift(1)",
    },
    "kalman_resid_medium": {
        "role": "Residual of log price relative to denoised log level; captures short-term deviations that can precede volatility spikes",
        "transform": "log",
        "formula": "(log(Close) - filtered_log).shift(1)",
    },
    "kalman_resid_zscore_medium": {
        "role": "Standardized log residual; normalizes the deviation signal for cross-period comparability",
        "transform": "log",
        "formula": "20-day z-score of the log residual, then shift(1)",
    },
    "kalman_state_var_medium": {
        "role": "State uncertainty (log level); reflects estimation confidence, useful as an auxiliary RV feature",
        "transform": "log",
        "formula": "Kalman log-level state variance, shift(1)",
    },
    "close_to_kalman_medium": {
        "role": "Deviation rate of current price from denoised price; a mean-reversion-style feature for RV models",
        "transform": "ratio",
        "formula": "((Close - kalman_price) / kalman_price).shift(1)",
    },
}
def get_feature_doc(name: str) -> FeatureDoc | None:
    return FEATURE_DOCS.get(name)
