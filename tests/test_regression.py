import numpy as np
import pandas as pd
import pytest

from app.statistical_analysis.regression import linear_regression


def make_df(n=500):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": 2 * x + 1 + rng.normal(scale=0.1, size=n)})


def test_recovers_known_slope():
    result = linear_regression(make_df(), "y", ["x"])
    assert abs(result["coefficients"].loc["x", "Coefficient"] - 2) < 0.05
    assert result["r_squared"] > 0.99


def test_rejects_non_numeric_predictor():
    df = make_df()
    df["label"] = "a"
    with pytest.raises(ValueError, match="Non-numeric"):
        linear_regression(df, "y", ["label"])


def test_reports_dropped_rows():
    df = make_df()
    df.loc[:9, "x"] = np.nan
    assert linear_regression(df, "y", ["x"])["rows_dropped"] == 10
