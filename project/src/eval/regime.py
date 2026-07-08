from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import yfinance as yf


REGIME_METHOD_NAMES: tuple[str, ...] = (
    "quantile",
    "gmm",
    "kmeans",
    "agglomerative",
    "birch",
)

METHOD_DISPLAY_NAMES: dict[str, str] = {
    "quantile": "Quantile",
    "gmm": "GMM",
    "kmeans": "KMeans",
    "agglomerative": "Agglomerative",
    "birch": "Birch",
    "custom": "Custom",
}


class ClusterEstimator(Protocol):
    def fit(self, X: np.ndarray) -> Any: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


def fetch_vix_ma_series(
    start_date: str,
    end_date: str,
    *,
    ma_window: int = 30,
) -> pd.Series:
    start_ts = pd.Timestamp(start_date) - pd.Timedelta(days=90)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    vix_df = yf.download(
        "^VIX",
        start=start_ts.strftime("%Y-%m-%d"),
        end=end_ts.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if vix_df is None or vix_df.empty:
        raise ValueError("Failed to fetch ^VIX data from yfinance (empty result)")
    if "Close" not in vix_df.columns:
        raise ValueError("^VIX data does not contain Close column")
    vix_close_raw = vix_df["Close"]
    if isinstance(vix_close_raw, pd.DataFrame):
        if vix_close_raw.shape[1] != 1:
            raise ValueError(
                f"^VIX Close has unexpected shape {vix_close_raw.shape}; expected single column"
            )
        vix_close = vix_close_raw.iloc[:, 0].astype(float)
    else:
        vix_close = vix_close_raw.astype(float)
    vix_ma = vix_close.rolling(window=ma_window, min_periods=10).mean().bfill().ffill()
    return pd.Series(vix_ma, name="vix_ma30")


def _quantile_labels(values: np.ndarray, n_regimes: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Equal-frequency binning into n groups via rank quantiles."""
    series = pd.Series(values)
    # qcut can fail on duplicate edges; rank-based bins are more stable.
    ranks = series.rank(method="first")
    try:
        bins = pd.qcut(ranks, q=n_regimes, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.cut(ranks, bins=n_regimes, labels=False, include_lowest=True)
    raw_ids = np.asarray(bins, dtype=int).reshape(-1)
    # Approximate edges on the original value scale for reporting.
    edges = np.quantile(values, np.linspace(0.0, 1.0, n_regimes + 1)).tolist()
    return raw_ids, {"quantile_edges": edges}


def _make_gmm(n: int) -> GaussianMixture:
    return GaussianMixture(n_components=n, random_state=42, n_init=10)  # type: ignore[arg-type]


def _make_kmeans(n: int) -> KMeans:
    return KMeans(n_clusters=n, random_state=42, n_init=10)


def _make_agglomerative(n: int) -> AgglomerativeClustering:
    return AgglomerativeClustering(n_clusters=n)


def _make_birch(n: int) -> Birch:
    return Birch(n_clusters=n)


REGIME_METHOD_FACTORIES: dict[str, Callable[[int], Any]] = {
    "gmm": _make_gmm,
    "kmeans": _make_kmeans,
    "agglomerative": _make_agglomerative,
    "birch": _make_birch,
}


def _fit_predict_clusterer(clusterer: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(clusterer, "fit_predict"):
        labels = clusterer.fit_predict(X)
    else:
        clusterer.fit(X)
        labels = clusterer.predict(X)
    return np.asarray(labels, dtype=int)


def _remap_by_mean_desc(
    raw_ids: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Map raw cluster ids → Cluster_1..Cluster_n by mean(macro) descending."""
    unique = sorted(int(u) for u in np.unique(raw_ids) if u >= 0)
    if not unique:
        raise ValueError("No valid cluster assignments to remap")

    means = {uid: float(np.mean(values[raw_ids == uid])) for uid in unique}
    ordered = sorted(unique, key=lambda u: means[u], reverse=True)

    rank_map = {uid: rank for rank, uid in enumerate(ordered, start=1)}
    label_names = [f"Cluster_{rank}" for rank in range(1, len(ordered) + 1)]
    remapped_names = np.array(
        [f"Cluster_{rank_map[int(uid)]}" for uid in raw_ids],
        dtype=object,
    )
    cluster_means = {
        f"Cluster_{rank_map[uid]}": means[uid] for uid in ordered
    }
    return remapped_names, label_names, cluster_means


def assign_regime_clusters(
    macro_values: np.ndarray | pd.Series,
    *,
    method: str = "kmeans",
    n_regimes: int = 2,
    clusterer: Any | None = None,
    index: pd.Index | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Assign cluster labels on a 1D macro series and rename by mean descending.

    Parameters
    ----------
    macro_values:
        1D VIX (or fallback) indicator values.
    method:
        One of REGIME_METHOD_NAMES, or ignored when ``clusterer`` is provided.
    n_regimes:
        Number of clusters / quantile bins (must be >= 2).
    clusterer:
        Optional sklearn-style estimator with fit/predict or fit_predict.
        When set, overrides ``method`` for fitting (meta method becomes ``custom``).
    """
    if n_regimes < 2:
        raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")

    values = np.asarray(macro_values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("macro_values is empty")
    if np.isnan(values).any():
        raise ValueError("macro_values contain NaN; fill before clustering")

    extra_meta: dict[str, Any] = {}
    method_key = method

    if clusterer is not None:
        method_key = "custom"
        X = values.reshape(-1, 1)
        raw_ids = _fit_predict_clusterer(clusterer, X)
        extra_meta["clusterer_type"] = type(clusterer).__name__
    elif method == "quantile":
        raw_ids, extra_meta = _quantile_labels(values, n_regimes)
    elif method in REGIME_METHOD_FACTORIES:
        est = REGIME_METHOD_FACTORIES[method](n_regimes)
        X = values.reshape(-1, 1)
        raw_ids = _fit_predict_clusterer(est, X)
        extra_meta["clusterer_type"] = type(est).__name__
        if hasattr(est, "means_"):
            # GMM
            extra_meta["raw_means"] = np.asarray(est.means_).ravel().tolist()
        elif hasattr(est, "cluster_centers_"):
            extra_meta["raw_centers"] = np.asarray(est.cluster_centers_).ravel().tolist()
    else:
        raise ValueError(
            f"Unknown regime method '{method}'. "
            f"Valid: {list(REGIME_METHOD_NAMES)} or pass clusterer="
        )

    remapped, label_order, cluster_means = _remap_by_mean_desc(raw_ids, values)
    idx = index if index is not None else pd.RangeIndex(len(remapped))
    regimes = pd.Series(remapped, index=idx, name="Regime")

    meta: dict[str, Any] = {
        "method": method_key,
        "method_display": METHOD_DISPLAY_NAMES.get(method_key, method_key),
        "n_regimes": int(n_regimes),
        "cluster_order": label_order,
        "cluster_means": cluster_means,
        "threshold_rule": "cluster_mean_descending",
        **extra_meta,
    }
    return regimes, meta


def build_macro_indicator(
    oof_valid_df: pd.DataFrame,
    pipeline_config: dict[str, Any],
    *,
    ma_window: int = 30,
) -> tuple[pd.Series, str, dict[str, Any]]:
    """Fetch / construct the 1D macro series used for regime clustering."""
    if "Date" not in oof_valid_df.columns:
        raise ValueError("oof_predictions.csv must contain Date for VIX regime split")
    start_date = str(pipeline_config.get("start_date", ""))
    end_date = str(pipeline_config.get("end_date", ""))
    if not start_date or not end_date:
        raise ValueError("pipeline.json must contain start_date and end_date for VIX split")

    dates = pd.to_datetime(oof_valid_df["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Date column contains invalid values; cannot align VIX regimes")

    try:
        vix_ma = fetch_vix_ma_series(start_date, end_date, ma_window=ma_window)
        vix_ma.index = pd.to_datetime(vix_ma.index)
        vix_for_oof = vix_ma.reindex(dates).bfill().ffill()
        if vix_for_oof.isna().all():
            raise ValueError("Unable to align fetched VIX series to OOF validation dates")
        macro_indicator = pd.Series(vix_for_oof.to_numpy(), index=oof_valid_df.index)
        indicator_name = "vix_ma30_from_yfinance"
        indicator_meta: dict[str, Any] = {
            "used_fallback": False,
            "reason": None,
            "source_symbol": "^VIX",
            "construction": "close_rolling_mean",
            "ma_window": ma_window,
        }
    except Exception as exc:
        if "Close" not in oof_valid_df.columns:
            raise ValueError(
                "VIX fetch failed and no Close column available for fallback regime split"
            )
        close = oof_valid_df["Close"].astype(float)
        returns = close.pct_change()
        hist_vol_30 = returns.rolling(window=30, min_periods=10).std() * np.sqrt(252.0)
        macro_indicator = hist_vol_30.rolling(window=ma_window, min_periods=10).mean()
        macro_indicator = macro_indicator.bfill().ffill()
        indicator_name = "fallback_rolling_volatility_ma30_from_close"
        indicator_meta = {
            "used_fallback": True,
            "reason": str(exc),
            "source_symbol": "oof_valid_df.Close",
            "construction": "annualized_close_return_volatility_then_ma",
            "annualization_factor": 252.0,
            "vol_window": 30,
            "ma_window": ma_window,
        }

    return macro_indicator, indicator_name, indicator_meta


def build_macro_regime_labels(
    oof_valid_df: pd.DataFrame,
    pipeline_config: dict[str, Any],
    *,
    ma_window: int = 30,
    method: str = "kmeans",
    n_regimes: int = 2,
    clusterer: Any | None = None,
) -> tuple[pd.Series, str, dict[str, Any], pd.Series, dict[str, Any]]:
    """Build macro series and assign Cluster_1..Cluster_n by mean(macro) desc.

    Returns
    -------
    regimes, indicator_name, indicator_meta, macro_indicator, cluster_meta
    """
    macro_indicator, indicator_name, indicator_meta = build_macro_indicator(
        oof_valid_df, pipeline_config, ma_window=ma_window
    )
    regimes, cluster_meta = assign_regime_clusters(
        macro_indicator.to_numpy(dtype=float),
        method=method,
        n_regimes=n_regimes,
        clusterer=clusterer,
        index=oof_valid_df.index,
    )
    return regimes, indicator_name, indicator_meta, macro_indicator, cluster_meta


def ordered_group_keys(group_confusion: dict[str, dict[str, Any]]) -> list[str]:
    """Sort group keys as Overall, then Cluster_1..Cluster_n, then any remaining."""

    def _rank(key: str) -> tuple[int, str]:
        if key == "Overall":
            return (-1, key)
        if key.startswith("Cluster_"):
            try:
                return (0, f"{int(key.split('_', 1)[1]):05d}")
            except ValueError:
                return (1, key)
        return (1, key)

    return sorted(group_confusion.keys(), key=_rank)


def group_confusion_matrices(
    df: pd.DataFrame,
    *,
    group_col: str,
    y_col: str = "y",
    proba_col: str = "model_proba",
    threshold: float = 0.5,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group_value, part in df.groupby(group_col, sort=False):
        y_true = part[y_col].astype(int).to_numpy()
        y_pred = (part[proba_col].astype(float).to_numpy() >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        out[str(group_value)] = {
            "n_samples": int(len(part)),
            "threshold": threshold,
            "confusion_matrix": cm.tolist(),
        }
    # Stable Cluster_1..n order for consumers
    ordered = {k: out[k] for k in ordered_group_keys(out) if k in out}
    return ordered
