"""Unit tests for project.src.features.selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project.src.features.selection import (
    SELECTOR_REGISTRY,
    BaseFeatureSelector,
    TTestSelector,
    register_selector,
    select_features,
    select_features_on_fold,
)


def _synthetic_xy(n: int = 80, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    noise_a = rng.normal(size=n)
    noise_b = rng.normal(size=n)
    y = (signal > 0).astype(int)
    # Make signal strongly separated by class for t-test
    signal = np.where(y == 1, signal + 3.0, signal - 3.0)
    X = pd.DataFrame({"signal": signal, "noise_a": noise_a, "noise_b": noise_b})
    return X, pd.Series(y, name="y")


def test_select_features_unknown_method():
    X, y = _synthetic_xy()
    with pytest.raises(ValueError, match="Unknown feature selection method"):
        select_features(X, y, method="not_a_real_method")


def test_transform_and_get_scores_before_fit_raise():
    sel = TTestSelector(top_k=2)
    X, _ = _synthetic_xy()
    with pytest.raises(RuntimeError, match="fit"):
        sel.transform(X)
    with pytest.raises(RuntimeError, match="fit"):
        sel.get_scores()


def test_ttest_selector_picks_signal_and_clamps_top_k():
    X, y = _synthetic_xy()
    result = select_features(X, y, method="ttest", top_k=1)
    assert result.selected_features == ["signal"]
    assert list(result.X_selected.columns) == ["signal"]

    oversized = TTestSelector(top_k=99).fit(X, y)
    assert len(oversized.selected_features_) == X.shape[1]


def test_select_features_on_fold_aligns_val_and_handles_none():
    X, y = _synthetic_xy()
    X_train, X_val = X.iloc[:60], X.iloc[60:]
    y_train = y.iloc[:60]

    X_tr_sel, X_va_sel, result = select_features_on_fold(
        X_train, y_train, X_val, method="ttest", top_k=2
    )
    assert list(X_tr_sel.columns) == result.selected_features
    assert X_va_sel is not None
    assert list(X_va_sel.columns) == result.selected_features

    _, X_none, _ = select_features_on_fold(
        X_train, y_train, None, method="ttest", top_k=2
    )
    assert X_none is None


def test_register_selector_round_trip():
    class DummySelector(BaseFeatureSelector):
        name = "dummy_for_test"

        def fit(self, X: pd.DataFrame, y: pd.Series) -> DummySelector:
            self.scores_ = pd.Series(1.0, index=X.columns)
            k = min(self.top_k, len(X.columns))
            self.selected_features_ = list(X.columns[:k])
            return self

    register_selector("dummy_for_test", DummySelector)
    try:
        X, y = _synthetic_xy()
        result = select_features(X, y, method="dummy_for_test", top_k=2)
        assert result.method == "dummy_for_test"
        assert result.selected_features == ["signal", "noise_a"]
    finally:
        SELECTOR_REGISTRY.pop("dummy_for_test", None)
