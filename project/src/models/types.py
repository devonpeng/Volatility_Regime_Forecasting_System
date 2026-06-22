from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna


@dataclass
class ModelResult:
    name: str
    best_auc: float
    best_params: dict[str, Any]
    study: optuna.Study
