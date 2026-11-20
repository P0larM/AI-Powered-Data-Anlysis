import numpy as np
import pandas as pd

from app.statistical_analysis.categorical import analyze_categorical_features
from app.statistical_analysis.correlation import analyze_correlations
from app.statistical_analysis.multiple_testing import correct_p_values


def test_bonferroni_is_stricter_than_bh_and_ignores_nan():
    table = pd.DataFrame({"P-value": [0.001, 0.02, 0.04, np.nan]})
    out = correct_p_values(table)
    assert abs(out["Bonferroni P-value"].iloc[0] - 0.003) < 1e-12
    assert out["Bonferroni Significant"].sum() <= out["BH Significant"].sum()
    assert np.isnan(out["BH P-value"].iloc[3])


def test_correlation_finds_perfect_relationship():
    df = pd.DataFrame({"a": range(50), "b": [2 * i for i in range(50)], "c": [1, 2] * 25})
    result = analyze_correlations(df, columns=["a", "b", "c"])
    top = result["pairs"].iloc[0]
    assert {top["Variable 1"], top["Variable 2"]} == {"a", "b"}
    assert abs(top["Correlation"] - 1) < 1e-9


def test_categorical_detects_dependence_and_reports_skips():
    df = pd.DataFrame({
        "dep": ["x"] * 50 + ["y"] * 50,
        "target": ["a"] * 45 + ["b"] * 5 + ["b"] * 45 + ["a"] * 5,
        "constant": ["k"] * 100,
    })
    out = analyze_categorical_features(df, "target", ["dep", "constant"])
    assert out["results"].iloc[0]["Feature"] == "dep"
    assert out["results"].iloc[0]["Cramer's V"] > 0.7
    assert out["skipped"][0]["feature"] == "constant"
