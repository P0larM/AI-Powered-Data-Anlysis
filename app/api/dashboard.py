import pandas as pd
import numpy as np
from app.data.profiler import profile_dataset
from app.statistical_analysis.correlation import analyze_correlations
from app.database.queries import run_user_query

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
            res = run_user_query(conn, user_id, data_query, [table_name])
            df_numeric = pd.DataFrame(res["rows"], columns=res["columns"])
            corr_res = analyze_correlations(df_numeric, method="pearson", columns=numeric_cols)
            correlations = {
                "matrix": corr_res["matrix"].to_dict(),
                "top_pairs": corr_res["pairs"].head(10).to_dict(orient="records")
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
            res = run_user_query(conn, user_id, query, [table_name])
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
