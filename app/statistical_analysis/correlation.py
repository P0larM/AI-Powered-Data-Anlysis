from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def analyze_correlations(df, method="pearson", columns=None):
    """
    Calculate pairwise correlations with p-values.

    Parameters:
        df:      Dataset.
        method:  "pearson" (linear) or "spearman" (monotonic, rank-based).
        columns: Columns to include. Defaults to every numeric column, but
                 you should normally pass the ones your profiler marked as
                 continuous/discrete (not IDs or weights).

    Returns:
        dict with "method", "matrix" (DataFrame) and "pairs" (DataFrame
        sorted from strongest to weakest).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("Method must be 'pearson' or 'spearman'.")

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    if len(columns) < 2:
        raise ValueError("At least two numerical variables are required.")

    data = df[columns].astype(float).replace([np.inf, -np.inf], np.nan)
    test = pearsonr if method == "pearson" else spearmanr

    rows = []
    for col1, col2 in combinations(columns, 2):
        pair = data[[col1, col2]].dropna()

        # Need enough observations and variation in both variables.
        if len(pair) < 3:
            continue
        if pair[col1].nunique() < 2 or pair[col2].nunique() < 2:
            continue

        coefficient, p_value = test(pair[col1], pair[col2])
        rows.append({
            "Variable 1": col1,
            "Variable 2": col2,
            "Correlation": float(coefficient),
            "P-value": float(p_value),
            "Observations": int(len(pair)),
        })

    pairs = pd.DataFrame(
        rows,
        columns=["Variable 1", "Variable 2", "Correlation", "P-value", "Observations"],
    )

    # Build the matrix from the pairs so nothing is computed twice.
    matrix = pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)
    for column in columns:
        matrix.loc[column, column] = 1.0
    for row in rows:
        matrix.loc[row["Variable 1"], row["Variable 2"]] = row["Correlation"]
        matrix.loc[row["Variable 2"], row["Variable 1"]] = row["Correlation"]

    if not pairs.empty:
        pairs["Absolute Correlation"] = pairs["Correlation"].abs()
        pairs = pairs.sort_values(
            by="Absolute Correlation", ascending=False
        ).reset_index(drop=True)

    return {"method": method, "matrix": matrix, "pairs": pairs}
