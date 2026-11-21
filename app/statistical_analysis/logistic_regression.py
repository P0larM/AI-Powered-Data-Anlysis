import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
from scipy.stats import rankdata
from statsmodels.stats.outliers_influence import variance_inflation_factor


def logistic_regression(df, target, predictors):
    """
    Logistic regression for a binary (0/1) outcome.

    Parameters:
        df:         Dataset (all columns numeric).
        target:     Outcome column coded 0/1 (bools are fine).
        predictors: List of predictor columns.

    Returns a dict. "coefficients" is a DataFrame with log-odds coefficients
    and odds ratios (per one-unit increase of the predictor); everything else
    is plain numbers/strings/lists.
    """
    predictors = list(predictors)
    required_columns = [target] + predictors

    if not predictors:
        raise ValueError("Provide at least one predictor.")

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

    if data[target].nunique() < 2 or not set(data[target].unique()) <= {0.0, 1.0}:
        raise ValueError(
            "The target must be binary and coded 0/1, e.g. (income = '>50K')::int."
        )

    no_variation = [c for c in predictors if data[c].nunique() < 2]
    if no_variation:
        raise ValueError(f"Predictors with no variation: {no_variation}")

    y = data[target]
    events = int(y.sum())
    non_events = int(len(y) - events)

    if min(events, non_events) <= len(predictors) + 1:
        raise ValueError("Not enough observations in the smaller outcome class for this many predictors.")

    X = sm.add_constant(data[predictors], has_constant="add")

    try:
        model = sm.Logit(y, X).fit(disp=False, maxiter=100)
    except np.linalg.LinAlgError as error:
        raise ValueError("The model could not be fitted (collinear predictors?).") from error

    if not model.mle_retvals.get("converged", True):
        raise ValueError(
            "The model did not converge. Try fewer predictors or rescale large "
            "ones in SQL (e.g. capital_gain / 1000.0)."
        )
    if not np.isfinite(model.bse).all():
        raise ValueError("The model is unstable (possible perfect separation or collinearity).")

    # Coefficient table with odds ratios
    confidence_intervals = model.conf_int()
    confidence_intervals.columns = ["lo", "hi"]
    coefficient_table = pd.DataFrame({
        "Coefficient": model.params,
        "Std Error": model.bse,
        "Odds Ratio": np.exp(model.params),
        "OR Lower CI": np.exp(confidence_intervals["lo"]),
        "OR Upper CI": np.exp(confidence_intervals["hi"]),
        "P-value": model.pvalues,
        "Odds Ratio": np.exp(model.params),
        "Odds Change %": (np.exp(model.params) - 1) * 100,
    })

    # In-sample fit measures (AUC via the rank-sum identity)
    probabilities = model.predict(X).to_numpy()
    outcomes = y.to_numpy()
    ranks = rankdata(probabilities)
    auc = (ranks[outcomes == 1].sum() - events * (events + 1) / 2) / (events * non_events)
    accuracy = float(((probabilities >= 0.5) == outcomes).mean())
    baseline_accuracy = max(events, non_events) / len(y)

    vif = {
        column: float(variance_inflation_factor(X.values, i))
        for i, column in enumerate(X.columns)
        if column != "const"
    }

    warnings = [
        "Accuracy and AUC are measured on the same data used to fit the model, "
        "so they are optimistic."
    ]
    if rows_dropped:
        warnings.append(f"{rows_dropped} rows dropped because of missing values.")
    if min(events, non_events) / len(predictors) < 10:
        warnings.append("Fewer than 10 observations of the rarer outcome per predictor; estimates may be unstable.")
    for column, value in vif.items():
        if value > 10:
            warnings.append(f"High multicollinearity: {column} (VIF={value:.1f}).")
    if (model.params.abs() > 15).any():
        warnings.append("Very large coefficients: possible (near-)perfect separation; results unreliable.")

    return {
        "coefficients": coefficient_table,
        "observations": int(len(y)),
        "rows_dropped": rows_dropped,
        "events": events,
        "event_rate": events / len(y),
        "mcfadden_r_squared": float(model.prsquared),
        "likelihood_ratio_p_value": float(model.llr_pvalue),
        "aic": float(model.aic),
        "auc": float(auc),
        "accuracy": accuracy,
        "baseline_accuracy": float(baseline_accuracy),
        "vif": vif,
        "warnings": warnings,
    }