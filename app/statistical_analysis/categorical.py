import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def analyze_categorical_relationship(df, feature, target):
    """
    Chi-square test of independence plus Cramer's V for two categorical
    variables. Missing values must already be real NaN (see cleaner.py).
    """
    for column in (feature, target):
        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")

    data = df[[feature, target]].dropna()
    contingency_table = pd.crosstab(data[feature], data[target])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError("Both variables must have at least two categories.")

    chi2, p_value, degrees_of_freedom, expected = chi2_contingency(contingency_table)

    sample_size = int(contingency_table.to_numpy().sum())
    minimum_dimension = min(contingency_table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (sample_size * minimum_dimension)))

    expected_values = expected.flatten()
    share_below_5 = float((expected_values < 5).mean())

    assumption_check = {
        "Minimum Expected Frequency": float(expected_values.min()),
        "Percentage Below 5": share_below_5 * 100,
        "Assumptions Met": bool(
            (expected_values >= 1).all() and share_below_5 <= 0.20
        ),
    }

    return {
        "feature": feature,
        "target": target,
        "contingency_table": contingency_table,
        "chi_square": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(degrees_of_freedom),
        "cramers_v": cramers_v,
        "observations": sample_size,
        "expected_frequencies": expected,
        "assumption_check": assumption_check,
    }


def analyze_categorical_features(df, target, features=None):
    """
    Test each feature against the target.

    Parameters:
        features: Columns to test. Defaults to every non-numeric column.
                  Better: pass the columns your profiler marked categorical.

    Returns:
        dict with "results" (DataFrame sorted by Cramer's V) and
        "skipped" (list of {feature, reason}) so nothing fails silently.
    """
    if features is None:
        features = df.select_dtypes(exclude="number").columns.tolist()

    features = [f for f in features if f != target]

    rows, skipped = [], []

    for feature in features:
        try:
            analysis = analyze_categorical_relationship(df, feature, target)
        except ValueError as error:
            skipped.append({"feature": feature, "reason": str(error)})
            continue

        rows.append({
            "Feature": feature,
            "Chi-Square": analysis["chi_square"],
            "P-value": analysis["p_value"],
            "Cramer's V": analysis["cramers_v"],
            "Observations": analysis["observations"],
            "Assumptions Met": analysis["assumption_check"]["Assumptions Met"],
        })

    results = pd.DataFrame(
        rows,
        columns=["Feature", "Chi-Square", "P-value", "Cramer's V",
                 "Observations", "Assumptions Met"],
    )
    if not results.empty:
        results = results.sort_values(
            by="Cramer's V", ascending=False
        ).reset_index(drop=True)

    return {"results": results, "skipped": skipped}
