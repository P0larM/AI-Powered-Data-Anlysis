
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def analyze_correlations(df, method="pearson"):
    """
    Calculate correlations between numerical variables.

    Supported methods:
        pearson: Measures linear relationships.
        spearman: Measures monotonic relationships using ranks.
    """

    if method not in ("pearson", "spearman"):
        raise ValueError("Method must be 'pearson' or 'spearman'.")

    # Select numerical columns and replace infinite values.
    numerical_data = (
        df.select_dtypes(include="number")
        .replace([float("inf"), float("-inf")], float("nan"))
    )

    if numerical_data.shape[1] < 2:
        raise ValueError("At least two numerical variables are required.")

    # Calculate the correlation matrix.
    correlation_matrix = numerical_data.corr(
        method=method,
        min_periods=3
    )

    # Calculate individual correlations and p-values.
    results = []

    columns = numerical_data.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            col1 = columns[i]
            col2 = columns[j]

            pair = numerical_data[[col1, col2]].dropna()

            # Correlation requires at least three observations
            # and variation in both variables.
            if len(pair) < 3:
                continue

            if pair[col1].nunique() < 2 or pair[col2].nunique() < 2:
                continue

            if method == "pearson":
                coefficient, p_value = pearsonr(
                    pair[col1], pair[col2]
                )
            else:
                coefficient, p_value = spearmanr(
                    pair[col1], pair[col2]
                )

            results.append({
                "Variable 1": col1,
                "Variable 2": col2,
                "Correlation": coefficient,
                "P-value": p_value,
                "Observations": len(pair)
            })

    # Organize results from strongest to weakest relationship.
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df["Absolute Correlation"] = (
            results_df["Correlation"].abs()
        )

        results_df = results_df.sort_values(
            by="Absolute Correlation",
            ascending=False
        ).reset_index(drop=True)

    return {
        "method": method,
        "matrix": correlation_matrix,
        "pairs": results_df
    }
