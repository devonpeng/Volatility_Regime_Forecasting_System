from project.src.utils.cache import get_data_dir
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import seaborn as sns
from typing import Generator, Literal, Optional, Tuple
import numpy as np


class TimeSeriesSplitWithPurge:
    """
    Rolling-window time series cross-validator with purge gap.
    """

    def __init__(
        self,
        n_splits: int,
        train_window_size: int,
        test_size: int,
        gap_size: int = 5,
        auto_adjust: bool = True,
        expanding: bool = False,
        alignment: Literal["left", "right"] = "right",
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if alignment not in ("left", "right"):
            raise ValueError('alignment must be "left" or "right"')
        self.n_splits = n_splits
        self.train_window_size = train_window_size
        self.test_size = test_size
        self.gap_size = gap_size
        self.auto_adjust = auto_adjust
        self.expanding = expanding
        self.alignment = alignment

    def split(
        self, X, y=None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:

        if isinstance(X, (pd.DataFrame, pd.Series)):
            n_samples = len(X)
        else:
            n_samples = X.shape[0]

        max_possible_splits = (
            n_samples - self.train_window_size - self.gap_size
        ) // self.test_size

        if max_possible_splits < 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) for even one split with "
                f"train={self.train_window_size}, test={self.test_size}, "
                f"gap={self.gap_size}"
            )

        n_splits = min(self.n_splits, max_possible_splits)
        if self.auto_adjust and n_splits < self.n_splits:
            print(
                f"[Auto-adjust] Requested {self.n_splits} splits but only "
                f"{n_splits} possible with total samples {n_samples}. "
                f"Adjusted automatically."
            )

        total_covered = (
            self.train_window_size + self.gap_size + n_splits * self.test_size
        )
        leftover = n_samples - total_covered

        if self.alignment == "right":
            # Anchor to newest data: skip leftover oldest samples
            start_offset = max(0, leftover)
        else:
            # Anchor to oldest data: start from index 0, drop trailing samples
            start_offset = 0

        for split_idx in range(n_splits):
            train_start = split_idx * self.test_size + start_offset
            train_end = train_start + self.train_window_size

            test_start = train_end + self.gap_size
            test_end = test_start + self.test_size

            if self.alignment == "left" and test_end > n_samples:
                break

            if self.expanding:
                train_start = 0

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def visualize_splits(self, X, saved_name="split_diagram.jpg") -> None:
        if isinstance(X, (pd.DataFrame, pd.Series)):
            n_samples = len(X)
        else:
            n_samples = X.shape[0]

        data_points = []
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(X)):
            gap_start = train_idx[-1] + 1
            gap_end = test_idx[0] - 1
            gap_idx = (
                np.arange(gap_start, gap_end + 1) if gap_end >= gap_start else []
            )

            data_points.extend([(i, fold_idx, "Train") for i in train_idx])
            data_points.extend([(i, fold_idx, "Gap") for i in gap_idx])
            data_points.extend([(i, fold_idx, "Test") for i in test_idx])

        df_plot = pd.DataFrame(data_points, columns=["Index", "Fold", "Set"])
        plt.figure(figsize=(12, 1.5 * len(df_plot["Fold"].unique())))
        sns.scatterplot(
            data=df_plot,
            x="Index",
            y="Fold",
            hue="Set",
            palette={"Train": "#1f77b4", "Gap": "#ff7f0e", "Test": "#2ca02c"},
            marker="|",
            s=400,
            linewidth=2,
        )

        plt.title(
            f"TimeSeries Split Visualization  "
            f"(alignment={self.alignment!r}, "
            f"auto_adjust={self.auto_adjust}, "
            f"expanding={self.expanding})"
        )
        plt.xlabel("Sample Index (Time)")
        plt.ylabel("Fold")
        plt.legend(title="Set", loc="upper right")
        plt.tight_layout()
        plt.savefig(get_data_dir() / saved_name)