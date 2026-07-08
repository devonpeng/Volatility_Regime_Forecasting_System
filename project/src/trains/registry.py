"""Backward-compatible re-exports. Prefer project.src.models."""

from project.src.models.objectives.lgbm import objective_lgbm_classifier
from project.src.models.objectives.logistic import objective_logistic_classifier
from project.src.models.objectives.xgb import objective_xgb_classifier
from project.src.models.registry import MODEL_REGISTRY

__all__ = [
    "MODEL_REGISTRY",
    "objective_lgbm_classifier",
    "objective_logistic_classifier",
    "objective_xgb_classifier",
]
