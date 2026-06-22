from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSelectionResult:
    method: str
    selected_features: list[str]
    scores: pd.Series
    X_selected: pd.DataFrame


class BaseFeatureSelector(ABC):
    """Base class for feature selectors. Subclass and register to extend."""

    name: str = "base"

    def __init__(self, top_k: int = 5, random_state: int = 42):
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.scores_: pd.Series | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseFeatureSelector:
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            raise RuntimeError("Call fit() before transform()")
        return X[self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_scores(self) -> pd.Series:
        if self.scores_ is None:
            raise RuntimeError("Call fit() before get_scores()")
        return self.scores_


class TreeTopKSelector(BaseFeatureSelector):
    name = "tree_topk"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TreeTopKSelector:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )
        self.scores_ = importances
        k = min(self.top_k, len(X.columns))
        self.selected_features_ = importances.head(k).index.tolist()
        return self


class LassoSelector(BaseFeatureSelector):
    name = "lasso"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LassoSelector:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=0.1,
            random_state=self.random_state,
            max_iter=2000,
            class_weight="balanced",
        )
        model.fit(X_scaled, y)

        coefs = pd.Series(np.abs(model.coef_[0]), index=X.columns).sort_values(
            ascending=False
        )
        self.scores_ = coefs

        nonzero = coefs[coefs > 0]
        if len(nonzero) >= 1:
            k = min(self.top_k, len(nonzero))
            self.selected_features_ = nonzero.head(k).index.tolist()
        else:
            k = min(self.top_k, len(X.columns))
            self.selected_features_ = coefs.head(k).index.tolist()
        return self


class TTestSelector(BaseFeatureSelector):
    name = "ttest"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TTestSelector:
        scores: dict[str, float] = {}
        for col in X.columns:
            group0 = X.loc[y == 0, col]
            group1 = X.loc[y == 1, col]
            if len(group0) < 2 or len(group1) < 2:
                scores[col] = 0.0
                continue
            stat, _ = stats.ttest_ind(group0, group1, equal_var=False, nan_policy="omit")
            scores[col] = float(abs(stat)) if not np.isnan(stat) else 0.0

        ranked = pd.Series(scores).sort_values(ascending=False)
        self.scores_ = ranked
        k = min(self.top_k, len(X.columns))
        self.selected_features_ = ranked.head(k).index.tolist()
        return self


SELECTOR_REGISTRY: dict[str, type[BaseFeatureSelector]] = {
    TreeTopKSelector.name: TreeTopKSelector,
    LassoSelector.name: LassoSelector,
    TTestSelector.name: TTestSelector,
}


def register_selector(name: str, selector_cls: type[BaseFeatureSelector]) -> None:
    """Register a custom selector for pipeline extension."""
    SELECTOR_REGISTRY[name] = selector_cls


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "tree_topk",
    top_k: int = 5,
    random_state: int = 42,
    **kwargs,
) -> FeatureSelectionResult:
    if method not in SELECTOR_REGISTRY:
        raise ValueError(
            f"Unknown feature selection method: {method}. "
            f"Choose from {list(SELECTOR_REGISTRY)}"
        )

    selector_cls = SELECTOR_REGISTRY[method]
    selector = selector_cls(top_k=top_k, random_state=random_state, **kwargs)
    X_selected = selector.fit_transform(X, y)

    return FeatureSelectionResult(
        method=method,
        selected_features=selector.selected_features_,
        scores=selector.get_scores(),
        X_selected=X_selected,
    )
