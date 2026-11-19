
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency


def analyze_categorical_relationship(df, feature, target):
    """
    Analyze the relationship between two categorical variables
    using a chi-square test of independence and Cramer's V.
    """

    # Make a copy containing only the two variables
    data = df[[feature, target]].copy()

    # The Adult dataset uses '?' to represent unknown values.
    # Convert these to missing values before conducting the test.
    data = data.replace("?", np.nan)

    # Remove observations with missing values in either variable
    data = data.dropna()

    # Create a contingency table
    contingency_table = pd.crosstab(
        data[feature],
        data[target]
    )

    # A chi-square test requires at least two categories per variable
    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        raise ValueError("Both variables must have at least two categories.")

    # Perform the chi-square test
    chi2, p_value, degrees_of_freedom, expected = chi2_contingency(
        contingency_table
    )

    # Calculate Cramer's V
    sample_size = contingency_table.to_numpy().sum()
    minimum_dimension = min(
        contingency_table.shape[0] - 1,
        contingency_table.shape[1] - 1
    )

    cramers_v = np.sqrt(
        chi2 / (sample_size * minimum_dimension)
    )

    # Check expected frequency assumptions
    expected_values = expected.flatten()

    assumption_check = {
        "Minimum Expected Frequency": expected_values.min(),
        "Percentage Below 5": (
                                      (expected_values < 5).sum() / len(expected_values)
                              ) * 100,
        "Assumptions Met": (
                (expected_values >= 1).all()
                and (expected_values < 5).sum() / len(expected_values) <= 0.20
        )
    }

    return {
        "feature": feature,
        "target": target,
        "contingency_table": contingency_table,
        "chi_square": chi2,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
        "cramers_v": cramers_v,
        "observations": sample_size,
        "expected_frequencies": expected,
        "assumption_check": assumption_check
    }


def analyze_categorical_features(df, target):
    """
    Test every categorical feature against a specified target.
    """

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns.tolist()

    # Do not compare the target against itself
    features = [
        column for column in categorical_columns
        if column != target
    ]

    results = []

    for feature in features:
        try:
            analysis = analyze_categorical_relationship(
                df,
                feature,
                target
            )

            results.append({
                "Feature": feature,
                "Chi-Square": analysis["chi_square"],
                "P-value": analysis["p_value"],
                "Cramer's V": analysis["cramers_v"],
                "Observations": analysis["observations"],
                "Assumptions Met": analysis["assumption_check"]["Assumptions Met"]
            })

        except ValueError:
            continue

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            by="Cramer's V",
            ascending=False
        ).reset_index(drop=True)

    return results_df
