import numpy as np
import pandas as pd
import pytest

from app.ai import stat_tools


def patch_query(monkeypatch, df, truncated=False):
    def fake_run(conn, user_id, query, tables, max_rows=0, timeout_ms=0):
        return {"sql": query, "columns": list(df.columns),
                "rows": df.astype(object).values.tolist(),
                "row_count": len(df), "truncated": truncated}
    monkeypatch.setattr(stat_tools, "run_user_query", fake_run)


def call(name, args):
    return stat_tools.run_stat_tool(name, args, None, "demo", ["t"])


def numeric_df(n=300):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": 2 * x + rng.normal(scale=0.5, size=n), "z": rng.normal(size=n)})


def test_regression_tool(monkeypatch):
    patch_query(monkeypatch, numeric_df())
    out = call("run_regression", {"query": "q", "target": "y", "predictors": ["x", "z"]})
    assert out["r_squared"] > 0.8
    assert {row["Term"] for row in out["coefficients"]} == {"const", "x", "z"}
    assert out["rows_analyzed"] == 300 and "summary_text" not in out


def test_correlation_tool_returns_corrected_pairs(monkeypatch):
    patch_query(monkeypatch, numeric_df())
    out = call("run_correlations", {"query": "q"})
    top = out["pairs"][0]
    assert {top["Variable 1"], top["Variable 2"]} == {"x", "y"}
    assert "Bonferroni P-value" in top and out["pairs_tested"] == 3


def test_categorical_tool(monkeypatch):
    df = pd.DataFrame({"g": ["a"] * 60 + ["b"] * 60,
                       "t": ["u"] * 50 + ["v"] * 10 + ["v"] * 50 + ["u"] * 10})
    patch_query(monkeypatch, df)
    out = call("run_categorical_test", {"query": "q", "target": "t"})
    assert out["results"][0]["Feature"] == "g" and out["results"][0]["Cramer's V"] > 0.5


def test_compare_groups_tool(monkeypatch):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 100,
                       "v": np.r_[rng.normal(0, 1, 100), rng.normal(2, 1, 100)]})
    patch_query(monkeypatch, df)
    out = call("compare_groups", {"query": "q", "group_column": "g", "value_column": "v"})
    assert out["effect_size"]["magnitude"] == "large"


def test_refuses_truncated_results(monkeypatch):
    patch_query(monkeypatch, numeric_df(), truncated=True)
    with pytest.raises(ValueError, match="more than"):
        call("run_correlations", {"query": "q"})


def test_bad_calls_give_readable_errors(monkeypatch):
    patch_query(monkeypatch, numeric_df())
    with pytest.raises(ValueError, match="Unknown tool"):
        call("hack", {"query": "q"})
    with pytest.raises(ValueError, match="Missing argument"):
        call("run_regression", {"query": "q"})


def test_clean_makes_results_json_safe():
    out = stat_tools._clean({"a": np.float64("nan"), "b": np.int64(3), "p": 1.23456789e-300,
                             "df": pd.DataFrame({"k": [1.5]})})
    assert out == {"a": None, "b": 3, "p": 1.23457e-300, "df": [{"k": 1.5}]}
