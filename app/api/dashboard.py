import pandas as pd
import numpy as np
from app.data.profiler import profile_dataset
from app.statistical_analysis.correlation import analyze_correlations
from app.database.queries import run_user_query
from app.statistical_analysis.simple_regression import (
    simple_linear_regressions,
    simple_logistic_regressions,
)

# Rows sampled for dashboard statistics. run_user_query caps rows at 1,000 by
# default, so the cap is passed explicitly to match each query's LIMIT.
SAMPLE_ROWS = 10_000
HISTOGRAM_ROWS = 5_000
NUMERIC_TYPES = ("continuous", "discrete")


def _quote(name):
    """Quote an identifier for the SQL we build ourselves."""
    return '"' + str(name).replace('"', '""') + '"'


def _json_safe(value):
    """NaN/inf become None so the response is valid JSON."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def model_predictors(profile, exclude=None):
    """Numeric, non-ID, non-weight columns usable as regression predictors."""
    return [name for name, info in profile["columns"].items()
            if info["type"] in NUMERIC_TYPES and not info["is_weight"] and name != exclude]


def model_kind(profile, column):
    """
    Which per-variable model applies to a column as the outcome:
    "linear" for numeric columns, "logistic" for two-level columns, else None.
    IDs, weights, constants and free text never get a model.
    """
    info = profile["columns"][column]
    if info["is_weight"] or not model_predictors(profile, exclude=column):
        return None
    if info["type"] in NUMERIC_TYPES:
        return "linear"
    if info["type"] == "categorical" and info["n_unique"] == 2:
        return "logistic"
    return None

def get_dashboard_data(conn, user_id, table_name, profile):
    """
    Compute detailed dashboard statistics for a dataset.
    
    Parameters:
        conn:       Query connection.
        user_id:    User ID.
        table_name: The table to analyze.
        profile:    The stored profile from app_meta.datasets.
    """
    # 1. Data Overview is already mostly in profile (rows, n_columns, duplicate_rows)
    overview = {
        "total_observations": profile["rows"],
        "total_variables": profile["n_columns"],
        "duplicate_rows": profile["duplicate_rows"],
        "missing_cells_pct": _calculate_total_missing_pct(profile),
    }

    # 2. Data Quality Assessment
    # Completeness: % of non-null values across all cells
    # Uniqueness: Based on duplicate rows (1 - duplicate_rows/rows)
    # Validity: For now, we'll use a simple heuristic (e.g., all non-empty is valid)
    quality = {
        "completeness": 100 - overview["missing_cells_pct"],
        "uniqueness": round(100 * (1 - profile["duplicate_rows"] / profile["rows"]), 2) if profile["rows"] > 0 else 100,
        "validity": 100.0, # Placeholder for more complex rules
    }

    # 3. Variable Distributions & Stats
    # We'll enhance the profile info for the dashboard
    variables = {}
    for col_name, col_info in profile["columns"].items():
        var_data = {
            "name": col_name,
            "type": col_info["type"],
            "missing_pct": col_info["null_pct"],
            "n_unique": col_info["n_unique"],
            "model": model_kind(profile, col_name),
        }
        
        # Add basic stats for numeric types
        if "mean" in col_info:
            var_data.update({
                "mean": col_info["mean"],
                "median": col_info["median"],
                "std": col_info["std"],
                "min": col_info["min"],
                "max": col_info["max"],
            })
            
        # For distributions, we need histograms or top values
        # We'll fetch them from the database
        var_data["distribution"] = _fetch_distribution(conn, user_id, table_name, col_name, col_info)
        
        variables[col_name] = var_data

    # 4. Correlations (Numeric only)
    numeric_cols = [name for name, info in profile["columns"].items() 
                    if info["type"] in ("continuous", "discrete") and not info["is_weight"]]
    
    correlations = {}
    if len(numeric_cols) >= 2:
        # We need the actual data to calculate correlation
        # Fetch numeric data from DB (limited to avoid OOM for very large datasets)
        data_query = f"SELECT {', '.join([f'\"{c}\"' for c in numeric_cols])} FROM \"{table_name}\" LIMIT 10000"
        try:
            res = run_user_query(conn, user_id, data_query, [table_name], max_rows=SAMPLE_ROWS)
            df_numeric = pd.DataFrame(res["rows"], columns=res["columns"])
            corr_res = analyze_correlations(df_numeric, method="pearson", columns=numeric_cols)
            correlations = {
                "columns": list(corr_res["matrix"].columns),
                "matrix": _json_safe(corr_res["matrix"].to_dict()),
                "top_pairs": _json_safe(corr_res["pairs"].head(10).to_dict(orient="records"))
            }
        except Exception:
            correlations = {"error": "Could not calculate correlations"}

    return {
        "overview": overview,
        "quality": quality,
        "variables": variables,
        "correlations": correlations
    }

def _calculate_total_missing_pct(profile):
    total_cells = profile["rows"] * profile["n_columns"]
    if total_cells == 0:
        return 0
    total_missing = sum(col["null_count"] for col in profile["columns"].values())
    return round(float(total_missing / total_cells * 100), 2)

def _fetch_distribution(conn, user_id, table_name, col_name, col_info):
    """Fetch histogram bins for numeric or top values for categorical."""
    if col_info["type"] in ("continuous", "discrete"):
        # Simple histogram using SQL if possible, or just fetch sample and bin in Python
        # For simplicity and robustness, let's fetch a sample
        query = f"SELECT \"{col_name}\" FROM \"{table_name}\" WHERE \"{col_name}\" IS NOT NULL LIMIT 5000"
        try:
            res = run_user_query(conn, user_id, query, [table_name], max_rows=HISTOGRAM_ROWS)
            vals = [r[0] for r in res["rows"]]
            if not vals:
                return []
            counts, bin_edges = np.histogram(vals, bins=10)
            return [
                {"bin_start": float(bin_edges[i]), "bin_end": float(bin_edges[i+1]), "count": int(counts[i])}
                for i in range(len(counts))
            ]
        except Exception:
            return []
    else:
        # Use top_values from profile if available, otherwise fetch
        if "top_values" in col_info:
            return [{"label": k, "count": v} for k, v in col_info["top_values"].items()]
        
        query = f"SELECT \"{col_name}\", COUNT(*) as count FROM \"{table_name}\" GROUP BY \"{col_name}\" ORDER BY count DESC LIMIT 10"
        try:
            res = run_user_query(conn, user_id, query, [table_name])
            return [{"label": str(r[0]), "count": int(r[1])} for r in res["rows"]]
        except Exception:
            return []


def get_variable_models(conn, user_id, table_name, profile, column):
    """
    Simple regressions with `column` as the outcome, one model per predictor.

    Numeric columns get one-predictor linear regressions; two-level columns get
    one-predictor logistic regressions. Fitted on up to SAMPLE_ROWS rows.
    """
    if column not in profile["columns"]:
        raise KeyError(column)
    kind = model_kind(profile, column)
    result = {"column": column, "kind": kind, "total_rows": profile["rows"],
              "linear": None, "logistic": None}
    if kind is None:
        result["reason"] = "Regression needs a numeric or two-level column and at least one numeric predictor."
        return result

    predictors = model_predictors(profile, exclude=column)
    query = f"SELECT {', '.join(_quote(c) for c in [column, *predictors])} FROM {_quote(table_name)} LIMIT {SAMPLE_ROWS}"
    res = run_user_query(conn, user_id, query, [table_name], max_rows=SAMPLE_ROWS)
    df = pd.DataFrame(res["rows"], columns=res["columns"])
    for name in predictors:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    result["sample_rows"] = int(len(df))

    if kind == "linear":
        df[column] = pd.to_numeric(df[column], errors="coerce")
        result["linear"] = simple_linear_regressions(df, column, predictors)
    else:
        result["logistic"] = simple_logistic_regressions(df, column, predictors)
        result["logistic"]["event"] = str(result["logistic"]["event"])
    return _json_safe(result)
