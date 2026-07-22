"""Unit tests for pure pandas feature steps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from project.src.features.steps import (
    add_day_of_week,
    add_hl_range,
    add_is_month_end,
    add_ret_1d,
    add_rv_ratio_5_20,
    add_vix_level,
)


def _ohlc_frame(n: int = 10) -> pd.DataFrame:
    close = np.linspace(100.0, 110.0, n)
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        }
    )


def test_add_ret_1d_and_hl_range_do_not_mutate_input():
    df = _ohlc_frame()
    original_cols = list(df.columns)

    out_ret = add_ret_1d(df)
    assert "ret_1d" in out_ret.columns
    assert pd.isna(out_ret["ret_1d"].iloc[0])
    assert list(df.columns) == original_cols

    out_hl = add_hl_range(df)
    assert "hl_range" in out_hl.columns
    expected = (df["High"] - df["Low"]) / df["Close"]
    pd.testing.assert_series_equal(out_hl["hl_range"], expected, check_names=False)
    assert "hl_range" not in df.columns


def test_add_vix_level_defaults_when_missing():
    df = _ohlc_frame()
    out = add_vix_level(df)
    assert "vix_level" in out.columns
    assert (out["vix_level"] == 0.0).all()

    with_vix = df.copy()
    with_vix["vix"] = 20.0
    out2 = add_vix_level(with_vix)
    assert (out2["vix_level"] == 20.0).all()


def test_add_rv_ratio_5_20_handles_zero_denominator():
    df = _ohlc_frame()
    df["rv_hist_5d"] = 0.2
    df["rv_hist_20d"] = [0.1, 0.0, 0.2, 0.0, 0.05, 0.1, 0.0, 0.3, 0.1, 0.2]
    out = add_rv_ratio_5_20(df)
    assert "rv_ratio_5_20" in out.columns
    assert pd.isna(out.loc[1, "rv_ratio_5_20"])
    assert pd.isna(out.loc[3, "rv_ratio_5_20"])
    assert out.loc[0, "rv_ratio_5_20"] == pytest.approx(2.0)


def test_day_of_week_and_is_month_end_from_date_column():
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-29"]),
            "Close": [1.0, 2.0, 3.0],
        }
    )
    dow = add_day_of_week(df)
    # 2024-01-31 is Wednesday (2)
    assert dow["day_of_week"].tolist() == [2.0, 3.0, 3.0]

    me = add_is_month_end(df)
    assert me["is_month_end"].tolist() == [1.0, 0.0, 1.0]
