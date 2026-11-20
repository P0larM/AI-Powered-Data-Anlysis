import numpy as np
import pandas as pd
import pytest

from app.statistical_analysis.group_comparison import compare_groups


def make_df(means, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, mean in means.items():
        rows += [{"group": name, "value": v} for v in rng.normal(mean, 1, n)]
    return pd.DataFrame(rows)


def test_two_groups_detects_large_difference():
    out = compare_groups(make_df({"a": 10, "b": 12}), "group", "value")
    assert out["tests"]["welch_t"]["p_value"] < 1e-10
    assert out["tests"]["mann_whitney_u"]["p_value"] < 1e-10
    assert out["effect_size"]["magnitude"] == "large"
    assert out["mean_difference"]["first_minus_second"] < 0


def test_two_identical_groups_are_not_significant():
    out = compare_groups(make_df({"a": 10, "b": 10}, seed=3), "group", "value")
    assert out["tests"]["welch_t"]["p_value"] > 0.01
    assert out["effect_size"]["magnitude"] in ("negligible", "small")


def test_three_groups_uses_anova_and_kruskal():
    out = compare_groups(make_df({"a": 10, "b": 10, "c": 13}), "group", "value")
    assert set(out["tests"]) == {"anova_f", "kruskal_wallis"}
    assert out["tests"]["anova_f"]["p_value"] < 1e-10
    assert out["effect_size"]["name"] == "eta_squared"


def test_errors_and_dropped_rows():
    df = make_df({"a": 1, "b": 2})
    df.loc[:4, "value"] = np.nan
    assert compare_groups(df, "group", "value")["rows_dropped"] == 5
    with pytest.raises(ValueError, match="numeric"):
        compare_groups(df.assign(value="x"), "group", "value")
    with pytest.raises(ValueError, match="two groups"):
        compare_groups(df[df.group == "a"], "group", "value")
