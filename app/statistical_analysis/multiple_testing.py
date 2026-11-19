
import pandas as pd
from statsmodels.stats.multitest import multipletests


def correct_p_values(results, p_value_column="P-value"):
    """
    Apply Bonferroni and Benjamini-Hochberg corrections
    to a collection of hypothesis tests.

    Returns a copy of the original results with adjusted
    p-values and significance decisions.
    """

    if results.empty:
        return results.copy()

    if p_value_column not in results.columns:
        raise ValueError(
            f"Column '{p_value_column}' was not found."
        )

    corrected_results = results.copy()

    # Extract the original p-values
    p_values = corrected_results[p_value_column].to_numpy()

    # Bonferroni correction
    bonferroni = multipletests(
        p_values,
        alpha=0.05,
        method="bonferroni"
    )

    # Benjamini-Hochberg correction
    benjamini_hochberg = multipletests(
        p_values,
        alpha=0.05,
        method="fdr_bh"
    )

    # Store adjusted p-values
    corrected_results["Bonferroni P-value"] = bonferroni[1]
    corrected_results["BH P-value"] = benjamini_hochberg[1]

    # Store significance decisions
    corrected_results["Bonferroni Significant"] = bonferroni[0]
    corrected_results["BH Significant"] = benjamini_hochberg[0]

    return corrected_results
