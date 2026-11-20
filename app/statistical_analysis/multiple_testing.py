import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def correct_p_values(results, p_value_column="P-value", alpha=0.05):
    """
    Add Bonferroni and Benjamini-Hochberg adjusted p-values and
    significance decisions to a table of hypothesis tests.

    Rows with a missing p-value are left uncorrected (NaN, not significant)
    and are not counted toward the number of tests.
    """
    if p_value_column not in results.columns:
        raise ValueError(f"Column '{p_value_column}' was not found.")

    corrected = results.copy()

    methods = {"Bonferroni": "bonferroni", "BH": "fdr_bh"}
    for label in methods:
        corrected[f"{label} P-value"] = np.nan
        corrected[f"{label} Significant"] = False

    valid = corrected[p_value_column].notna()
    if not valid.any():
        return corrected

    p_values = corrected.loc[valid, p_value_column].to_numpy(dtype=float)

    for label, method in methods.items():
        significant, adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)
        corrected.loc[valid, f"{label} P-value"] = adjusted
        corrected.loc[valid, f"{label} Significant"] = significant

    return corrected
