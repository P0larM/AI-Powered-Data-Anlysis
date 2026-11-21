import numpy as np
import pandas as pd
import pytest

from app.statistical_analysis.logistic_regression import logistic_regression


def make_df(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    noise = rng.normal(size=n)
    probability = 1 / (1 + np.exp(-(1.5 * x - 0.5)))
    return pd.DataFrame({"x": x, "noise": noise, "y": rng.binomial(1, probability)})


def test_recovers_known_effect():
    out = logistic_regression(make_df(), "y", ["x", "noise"])
    table = out["coefficients"]
    assert abs(table.loc["x", "Coefficient"] - 1.5) < 0.3
    assert abs(table.loc["x", "Odds Ratio"] - np.exp(1.5)) < 1.5
    assert table.loc["noise", "P-value"] > 0.001
    assert out["auc"] > 0.7


def test_rejects_non_binary_target():
    df = make_df()
    df["y"] = df["y"] * 2
    with pytest.raises(ValueError, match="binary"):
        logistic_regression(df, "y", ["x"])


def test_rejects_non_numeric_predictor():
    df = make_df().assign(label="a")
    with pytest.raises(ValueError, match="Non-numeric"):
        logistic_regression(df, "y", ["label"])

@pytest.mark.filterwarnings("ignore::RuntimeWarning", "ignore::statsmodels.tools.sm_exceptions.ConvergenceWarning")

def test_perfect_separation_is_error_or_warning():
    df = make_df()
    df["y"] = (df["x"] > 0).astype(int)
    try:
        out = logistic_regression(df, "y", ["x"])
    except ValueError:
        return
    assert any("separation" in w.lower() for w in out["warnings"])