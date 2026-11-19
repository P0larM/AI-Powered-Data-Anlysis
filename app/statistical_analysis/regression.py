
import pandas as pd
import statsmodels.api as sm


def linear_regression(df, target, predictors):
    """
    Perform simple or multiple linear regression.

    Parameters:
        df: Dataset
        target: Dependent variable
        predictors: List of independent variables

    Returns:
        Regression model and summary statistics
    """

    # Validate requested columns
    required_columns = [target] + predictors

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Columns not found: {missing_columns}"
        )

    if target in predictors:
        raise ValueError("The target cannot also be a predictor.")

    # Select relevant columns and convert to numeric
    data = df[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove missing and infinite observations
    data = data.replace([float("inf"), float("-inf")], float("nan"))
    data = data.dropna()

    if len(data) <= len(predictors) + 1:
        raise ValueError("Not enough observations for regression.")

    if data[target].nunique() < 2:
        raise ValueError("The target must contain variation.")

    # Separate dependent and independent variables
    X = data[predictors]
    y = data[target]

    # Add intercept
    X = sm.add_constant(X)

    # Fit ordinary least squares regression
    model = sm.OLS(y, X).fit()

    # Confidence intervals
    confidence_intervals = model.conf_int()
    confidence_intervals.columns = [
        "Lower CI",
        "Upper CI"
    ]

    # Regression coefficient table
    coefficient_table = pd.DataFrame({
        "Coefficient": model.params,
        "P-value": model.pvalues,
        "Standard Error": model.bse
    })

    coefficient_table = coefficient_table.join(
        confidence_intervals
    )

    return {
        "model": model,
        "coefficients": coefficient_table,
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_p_value": model.f_pvalue,
        "observations": int(model.nobs),
        "residuals": model.resid,
        "predictions": model.fittedvalues,
        "summary": model.summary()
    }
