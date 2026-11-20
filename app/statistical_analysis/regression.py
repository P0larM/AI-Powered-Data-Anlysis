import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


def linear_regression(df, target, predictors, robust=True):
    """
    Simple or multiple linear regression (OLS) with diagnostics.

    Parameters:
        df:         Dataset (columns must already be numeric).
        target:     Dependent variable.
        predictors: List of independent variables.
        robust:     Use HC3 heteroscedasticity-robust standard errors for the
                    coefficient table (recommended). R-squared and the F-test
                    always come from the classic fit.

    Returns a dict. "coefficients" is a DataFrame; everything else is plain
    numbers/strings/lists so it can be serialized or shown to an LLM.
    """
    required_columns = [target] + list(predictors)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    if target in predictors:
        raise ValueError("The target cannot also be a predictor.")

    non_numeric = [c for c in required_columns if not is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric columns (encode them first): {non_numeric}")

    data = df[required_columns].astype(float)
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    rows_dropped = int(len(df) - len(data))

    if len(data) <= len(predictors) + 1:
        raise ValueError("Not enough observations for regression.")

    if data[target].nunique() < 2:
        raise ValueError("The target must contain variation.")

    X = sm.add_constant(data[list(predictors)], has_constant="add")
    y = data[target]

    classic = sm.OLS(y, X).fit()
    fitted = sm.OLS(y, X).fit(cov_type="HC3") if robust else classic

    confidence_intervals = fitted.conf_int()
    confidence_intervals.columns = ["Lower CI", "Upper CI"]

    coefficient_table = pd.DataFrame({
        "Coefficient": fitted.params,
        "P-value": fitted.pvalues,
        "Standard Error": fitted.bse,
    }).join(confidence_intervals)

    # Diagnostics
    vif = {
        column: float(variance_inflation_factor(X.values, i))
        for i, column in enumerate(X.columns)
        if column != "const"
    }
    bp_p_value = float(het_breuschpagan(classic.resid, X)[1])

    warnings = []
    if rows_dropped:
        warnings.append(f"{rows_dropped} rows dropped because of missing values.")
    for column, value in vif.items():
        if value > 10:
            warnings.append(f"High multicollinearity: {column} (VIF={value:.1f}).")
    if bp_p_value < 0.05:
        warnings.append(
            "Heteroscedasticity detected (Breusch-Pagan p < 0.05); "
            "classic standard errors are unreliable."
        )
    for column in predictors:
        zero_share = float((data[column] == 0).mean())
        if zero_share > 0.5:
            warnings.append(
                f"{column} is {zero_share:.0%} zeros; a linear fit may be a poor model."
            )

    return {
        "coefficients": coefficient_table,
        "robust_standard_errors": bool(robust),
        "r_squared": float(classic.rsquared),
        "adjusted_r_squared": float(classic.rsquared_adj),
        "f_statistic": float(classic.fvalue),
        "f_p_value": float(classic.f_pvalue),
        "observations": int(classic.nobs),
        "rows_dropped": rows_dropped,
        "vif": vif,
        "breusch_pagan_p_value": bp_p_value,
        "warnings": warnings,
        "summary_text": str(fitted.summary()),
    }
