"""
One-predictor-at-a-time regressions, used to profile a single variable.

Each predictor gets its own model (target ~ predictor), so the results answer
"how much does each column on its own tell us about this one?". They are not
a multiple regression: effects are unadjusted for the other predictors.

Both functions reuse the full-featured fits in regression.py and
logistic_regression.py, so the estimates, robust standard errors and checks
are exactly the ones the rest of the app reports.
"""
import numpy as np
import pandas as pd

from app.statistical_analysis.logistic_regression import logistic_regression
from app.statistical_analysis.multiple_testing import correct_p_values
from app.statistical_analysis.regression import linear_regression

MAX_SCATTER_POINTS = 300


def _rank_and_correct(rows, sort_column):
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values(sort_column, ascending=False).reset_index(drop=True)
    return correct_p_values(table)


def simple_linear_regressions(df, target, predictors, max_points=MAX_SCATTER_POINTS):
    """
    Fit target ~ predictor separately for every predictor (OLS, HC3 robust errors).

    Returns a dict:
        models:  DataFrame, one row per predictor, strongest R-squared first,
                 with Bonferroni/BH-corrected p-values for the slopes.
        skipped: list of {"predictor", "reason"} for predictors that could not be fitted.
        best:    the top model's line plus a sample of points for a scatter plot
                 (None when nothing could be fitted).
    """
    rows, skipped, fits = [], [], {}
    for predictor in predictors:
        pair = df[[target, predictor]].replace([np.inf, -np.inf], np.nan).dropna()
        if pair[predictor].nunique() < 2:
            skipped.append({"predictor": predictor, "reason": "No variation in this column."})
            continue
        try:
            fit = linear_regression(df, target, [predictor], robust=True)
        except ValueError as error:
            skipped.append({"predictor": predictor, "reason": str(error)})
            continue
        coefficients = fit["coefficients"]
        slope = coefficients.loc[predictor]
        rows.append({
            "Predictor": predictor,
            "Slope": float(slope["Coefficient"]),
            "Intercept": float(coefficients.loc["const", "Coefficient"]),
            "R-squared": fit["r_squared"],
            "P-value": float(slope["P-value"]),
            "Slope Lower CI": float(slope["Lower CI"]),
            "Slope Upper CI": float(slope["Upper CI"]),
            "Observations": fit["observations"],
        })
        fits[predictor] = fit

    models = _rank_and_correct(rows, "R-squared")

    best = None
    if not models.empty:
        top = models.iloc[0]
        predictor = top["Predictor"]
        points = df[[predictor, target]].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(points) > max_points:
            points = points.sample(max_points, random_state=0)
        best = {
            "predictor": predictor,
            "slope": float(top["Slope"]),
            "intercept": float(top["Intercept"]),
            "r_squared": float(top["R-squared"]),
            "x_min": float(df[predictor].min()),
            "x_max": float(df[predictor].max()),
            "points": points.to_numpy().tolist(),
            "warnings": [w for w in fits[predictor]["warnings"] if "rows dropped" not in w],
        }

    return {"target": target, "models": models, "skipped": skipped, "best": best}


def binary_outcome(series):
    """
    Code a two-level column as 0/1 with the rarer level as the event (1).
    Missing values stay missing. Returns (coded series, event label).
    """
    counts = series.dropna().value_counts()
    if len(counts) != 2:
        raise ValueError("The outcome must have exactly two distinct values.")
    # Rarer level is the event; ties fall back to sorted order for a stable choice.
    ordered = sorted(counts.items(), key=lambda kv: (kv[1], str(kv[0])))
    event = ordered[0][0]
    coded = series.map(lambda v: np.nan if pd.isna(v) else float(v == event))
    return coded.astype(float), event


def simple_logistic_regressions(df, target, predictors):
    """
    Fit a one-predictor logistic regression of a two-level target on each predictor.

    Predictors are standardized (mean 0, SD 1) before fitting, so every odds
    ratio means "per one standard deviation increase" and they can be compared
    with each other. P-values and AUC are unaffected by the scaling.

    Returns a dict:
        event:      the level coded as 1 (the rarer one).
        event_rate: share of rows where the event happens.
        models:     DataFrame, one row per predictor, best AUC first, with
                    Bonferroni/BH-corrected p-values.
        skipped:    list of {"predictor", "reason"}.
    """
    outcome, event = binary_outcome(df[target])
    rows, skipped = [], []
    for predictor in predictors:
        x = pd.to_numeric(df[predictor], errors="coerce").replace([np.inf, -np.inf], np.nan)
        sd = x.std()
        if not np.isfinite(sd) or sd == 0:
            skipped.append({"predictor": predictor, "reason": "No variation in this column."})
            continue
        data = pd.DataFrame({"__outcome": outcome, predictor: (x - x.mean()) / sd})
        try:
            fit = logistic_regression(data, "__outcome", [predictor])
        except ValueError as error:
            skipped.append({"predictor": predictor, "reason": str(error)})
            continue
        row = fit["coefficients"].loc[predictor]
        rows.append({
            "Predictor": predictor,
            "Odds Ratio per SD": float(row["Odds Ratio"]),
            "OR Lower CI": float(row["OR Lower CI"]),
            "OR Upper CI": float(row["OR Upper CI"]),
            "Coefficient per unit": float(row["Coefficient"] / sd),
            "P-value": float(row["P-value"]),
            "AUC": fit["auc"],
            "McFadden R-squared": fit["mcfadden_r_squared"],
            "Observations": fit["observations"],
        })

    return {
        "target": target,
        "event": event,
        "event_rate": float(outcome.mean()),
        "models": _rank_and_correct(rows, "AUC"),
        "skipped": skipped,
    }
