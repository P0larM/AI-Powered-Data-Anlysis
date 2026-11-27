import math

import numpy as np
import pandas as pd
import pytest

from app.api import dashboard
from app.data.profiler import profile_dataset
from app.statistical_analysis.simple_regression import (
    binary_outcome,
    simple_linear_regressions,
    simple_logistic_regressions,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(50, 10, n)
    noise = rng.normal(0, 30, n)
    y = 2.0 * x + rng.normal(0, 5, n)
    p = 1 / (1 + np.exp(-(x - 50) / 5))
    return pd.DataFrame({
        "row_id": np.arange(n),
        "x": x,
        "noise": noise,
        "y": y,
        "group": np.where(rng.random(n) < p, "high", "low"),
        "fnlwgt": rng.normal(1e5, 1e4, n),
    })


def test_linear_ranks_predictors_and_recovers_slope(df):
    out = simple_linear_regressions(df, "y", ["noise", "x"])
    models = out["models"]
    assert list(models["Predictor"]) == ["x", "noise"]  # strongest R-squared first
    assert models.loc[0, "Slope"] == pytest.approx(2.0, abs=0.1)
    assert models.loc[0, "Slope Lower CI"] < 2.0 < models.loc[0, "Slope Upper CI"]
    assert "BH P-value" in models.columns and "Bonferroni Significant" in models.columns
    assert bool(models.loc[0, "Bonferroni Significant"])
    best = out["best"]
    assert best["predictor"] == "x" and len(best["points"]) == 300


def test_linear_skips_constant_predictors(df):
    df["flat"] = 1.0
    out = simple_linear_regressions(df, "y", ["flat", "x"])
    assert [s["predictor"] for s in out["skipped"]] == ["flat"]
    assert list(out["models"]["Predictor"]) == ["x"]


def test_binary_outcome_uses_rarer_level_as_event():
    coded, event = binary_outcome(pd.Series(["a", "a", "a", "b", None]))
    assert event == "b"
    assert coded.tolist()[:4] == [0.0, 0.0, 0.0, 1.0] and math.isnan(coded.iloc[4])
    with pytest.raises(ValueError):
        binary_outcome(pd.Series(["a", "b", "c"]))


def test_logistic_odds_ratios_are_per_sd(df):
    out = simple_logistic_regressions(df, "group", ["x", "noise"])
    models = out["models"].set_index("Predictor")
    assert out["event"] in ("high", "low")
    assert models.index[0] == "x"  # best AUC first
    assert models.loc["x", "AUC"] > 0.75
    assert models.loc["x", "OR Lower CI"] < models.loc["x", "Odds Ratio per SD"] < models.loc["x", "OR Upper CI"]
    # x drives the outcome: its odds ratio is far from 1, noise's is not.
    assert abs(math.log(models.loc["x", "Odds Ratio per SD"])) > abs(math.log(models.loc["noise", "Odds Ratio per SD"]))


def test_model_kind_skips_ids_weights_and_wide_categories(df):
    df["city"] = [f"c{i % 7}" for i in range(len(df))]
    profile = profile_dataset(df)
    assert profile["columns"]["row_id"]["type"] == "id"
    assert dashboard.model_kind(profile, "row_id") is None
    assert dashboard.model_kind(profile, "fnlwgt") is None  # survey weight
    assert dashboard.model_kind(profile, "city") is None
    assert dashboard.model_kind(profile, "y") == "linear"
    assert dashboard.model_kind(profile, "group") == "logistic"
    assert "row_id" not in dashboard.model_predictors(profile)
    assert "fnlwgt" not in dashboard.model_predictors(profile)


def test_get_variable_models_returns_json_safe_results(df, monkeypatch):
    profile = profile_dataset(df)
    seen = {}

    def fake_query(conn, user_id, query, tables, max_rows=1000):
        seen["query"], seen["max_rows"] = query, max_rows
        cols = [c.strip().strip('"') for c in query.split("SELECT ")[1].split(" FROM")[0].split(",")]
        return {"columns": cols, "rows": df[cols].values.tolist()}

    monkeypatch.setattr(dashboard, "run_user_query", fake_query)

    linear = dashboard.get_variable_models(None, "u", "t", profile, "y")
    assert seen["max_rows"] == dashboard.SAMPLE_ROWS
    assert '"row_id"' not in seen["query"] and '"fnlwgt"' not in seen["query"]
    assert linear["kind"] == "linear" and linear["logistic"] is None
    assert linear["linear"]["models"][0]["Predictor"] == "x"

    logistic = dashboard.get_variable_models(None, "u", "t", profile, "group")
    assert logistic["logistic"]["models"][0]["Predictor"] == "x"

    none = dashboard.get_variable_models(None, "u", "t", profile, "row_id")
    assert none["kind"] is None and none["reason"]


def test_json_safe_replaces_nan():
    assert dashboard._json_safe({"a": float("nan"), "b": [np.float64(1.5), np.inf]}) == {"a": None, "b": [1.5, None]}
