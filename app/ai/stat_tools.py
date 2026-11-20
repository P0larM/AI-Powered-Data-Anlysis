"""Statistics tools the model can call.

The model never runs code. It picks one of these fixed tools and supplies a
SQL query (validated and run by run_user_query) plus column names. The
result of the query becomes a DataFrame that goes into the reviewed
functions in app/statistical_analysis/.
"""
import math

import numpy as np
import pandas as pd

from app.database.queries import run_user_query
from app.statistical_analysis.categorical import analyze_categorical_features
from app.statistical_analysis.correlation import analyze_correlations
from app.statistical_analysis.group_comparison import compare_groups
from app.statistical_analysis.multiple_testing import correct_p_values
from app.statistical_analysis.regression import linear_regression
from app.statistical_analysis.logistic_regression import logistic_regression

STATS_MAX_ROWS = 200_000
STATS_TIMEOUT_MS = 30_000
MAX_CORRELATION_PAIRS_SHOWN = 20

_QUERY_PROPERTY = {
    "type": "string",
    "description": (
        "A single SELECT returning one row per observation and only the columns "
        "needed. Use numeric columns where the tool needs numbers, e.g. "
        "(sex = 'Male')::int AS is_male."
    ),
}


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"query": _QUERY_PROPERTY, **properties},
                "required": ["query", *required],
            },
        },
    }


STAT_TOOLS = [
    _tool(
        "run_regression",
        "Multiple linear regression (OLS, robust standard errors) with R-squared, "
        "F-test, VIF and assumption warnings. All columns must be numeric.",
        {
            "target": {"type": "string", "description": "Dependent variable column."},
            "predictors": {"type": "array", "items": {"type": "string"},
                           "description": "Independent variable columns."},
        },
        ["target", "predictors"],
    ),
    _tool(
        "run_logistic_regression",
        "Logistic regression for a yes/no outcome. Returns odds ratios (per "
        "one-unit increase of each predictor), McFadden R-squared, AUC and "
        "accuracy versus a baseline. The target must be coded 0/1 and all "
        "columns numeric.",
        {
            "target": {"type": "string",
                       "description": "0/1 outcome column, e.g. (income = '>50K')::int AS high_income."},
            "predictors": {"type": "array", "items": {"type": "string"},
                           "description": "Independent variable columns."},
        },
        ["target", "predictors"],
    ),
    _tool(
        "run_correlations",
        "Pairwise correlations between numeric columns with p-values corrected "
        "for multiple testing.",
        {
            "method": {"type": "string", "enum": ["pearson", "spearman"],
                       "description": "Pearson = linear, Spearman = rank-based. Default pearson."},
            "columns": {"type": "array", "items": {"type": "string"},
                        "description": "Columns to include (default: all numeric columns)."},
        },
        [],
    ),
    _tool(
        "run_categorical_test",
        "Chi-square tests of independence plus Cramer's V between a categorical "
        "target and categorical features, with corrected p-values.",
        {
            "target": {"type": "string", "description": "Categorical target column."},
            "features": {"type": "array", "items": {"type": "string"},
                         "description": "Feature columns (default: all non-numeric columns)."},
        },
        ["target"],
    ),
    _tool(
        "compare_groups",
        "Compare a numeric column across the groups of a categorical column "
        "(t-test + Mann-Whitney for 2 groups, ANOVA + Kruskal-Wallis for 3+), "
        "with effect size.",
        {
            "group_column": {"type": "string", "description": "Categorical column defining the groups."},
            "value_column": {"type": "string", "description": "Numeric column to compare."},
        },
        ["group_column", "value_column"],
    ),
]


# ---------- helpers ----------------------------------------------------------

def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _clean(value):
    """Make any result JSON-safe: plain types, NaN -> None, 6 significant digits."""
    if isinstance(value, pd.DataFrame):
        return _clean(value.to_dict("records"))
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(f"{value:.6g}") if math.isfinite(value) else None
    return str(value)


def _coerce_numeric(df):
    """Query results arrive as Python objects; turn numeric-looking columns into numbers."""
    df = df.copy()
    for column in df.columns:
        series = df[column]
        if series.dtype == object and series.notna().any():
            converted = pd.to_numeric(series, errors="coerce")
            if converted.notna().sum() == series.notna().sum():
                df[column] = converted
    return df


def _load_dataframe(query_conn, user_id, query, allowed_tables):
    result = run_user_query(
        query_conn, user_id, query, allowed_tables,
        max_rows=STATS_MAX_ROWS, timeout_ms=STATS_TIMEOUT_MS,
    )
    if result["truncated"]:
        raise ValueError(
            f"The query returned more than {STATS_MAX_ROWS} rows. "
            "Filter or aggregate in SQL first."
        )
    if not result["rows"]:
        raise ValueError("The query returned no rows.")
    if len(set(result["columns"])) != len(result["columns"]):
        raise ValueError("Give every selected column a unique alias.")

    df = pd.DataFrame(result["rows"], columns=result["columns"])
    return _coerce_numeric(df), result["sql"]


# ---------- handlers ---------------------------------------------------------

def _regression(df, args):
    output = linear_regression(df, args["target"], _as_list(args.get("predictors")))
    output.pop("summary_text", None)
    output["coefficients"] = output["coefficients"].reset_index().rename(columns={"index": "Term"})
    return output

def _logistic(df, args):
    output = logistic_regression(df, args["target"], _as_list(args.get("predictors")))
    output["coefficients"] = output["coefficients"].reset_index().rename(columns={"index": "Term"})
    return output

def _correlations(df, args):
    method = args.get("method") or "pearson"
    outcome = analyze_correlations(df, method=method, columns=_as_list(args.get("columns")) or None)
    pairs = outcome["pairs"]
    if pairs.empty:
        return {"method": method, "pairs": [], "note": "No column pair had enough valid data."}

    corrected = correct_p_values(pairs).drop(columns=["Absolute Correlation"])
    return {
        "method": method,
        "pairs_tested": int(len(corrected)),
        "pairs": corrected.head(MAX_CORRELATION_PAIRS_SHOWN),
        "note": "Sorted by strength. P-values are corrected for multiple testing "
                "(Bonferroni and Benjamini-Hochberg).",
    }


def _categorical(df, args):
    outcome = analyze_categorical_features(
        df, args["target"], _as_list(args.get("features")) or None
    )
    table = outcome["results"]
    if not table.empty:
        table = correct_p_values(table)
    return {
        "results": table,
        "skipped": outcome["skipped"],
        "note": "Cramer's V is the effect size (0 = no association, 1 = perfect). "
                "Assumptions Met = False means the chi-square p-value is unreliable.",
    }


def _compare(df, args):
    return compare_groups(df, args["group_column"], args["value_column"])


_HANDLERS = {
    "run_regression": _regression,
    "run_correlations": _correlations,
    "run_categorical_test": _categorical,
    "compare_groups": _compare,
    "run_logistic_regression": _logistic,
}


def run_stat_tool(name, arguments, query_conn, user_id, allowed_tables):
    """
    Execute one statistics tool call. Returns a JSON-safe dict.
    Raises ValueError (including QueryRejected) with a message the model can act on.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")

    df, sql_used = _load_dataframe(query_conn, user_id, arguments.get("query", ""), allowed_tables)

    try:
        output = handler(df, arguments)
    except KeyError as error:
        raise ValueError(f"Missing argument or unknown column: {error.args[0]}") from error

    output = _clean(output)
    output["sql"] = sql_used
    output["rows_analyzed"] = int(len(df))
    return output
