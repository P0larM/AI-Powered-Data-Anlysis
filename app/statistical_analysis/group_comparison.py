import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats

MAX_GROUPS = 20


def _magnitude(value, thresholds):
    value = abs(value)
    for limit, label in zip(thresholds, ("negligible", "small", "medium")):
        if value < limit:
            return label
    return "large"


def compare_groups(df, group, value):
    """
    Compare a numeric variable across the groups of a categorical one.

    Two groups:  Welch t-test, Mann-Whitney U, mean difference, Cohen's d.
    3+ groups:   one-way ANOVA, Kruskal-Wallis, eta squared.

    Both a parametric and a rank-based test are returned so skewed data
    doesn't mislead. Groups with fewer than 2 observations are ignored.
    """
    for column in (group, value):
        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")
    if group == value:
        raise ValueError("The group column and the value column must differ.")
    if not is_numeric_dtype(df[value]):
        raise ValueError(f"The value column '{value}' must be numeric.")

    data = df[[group, value]].copy()
    data[value] = data[value].astype(float).replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    rows_dropped = int(len(df) - len(data))

    sizes = data.groupby(group)[value].count()
    data = data[data[group].isin(sizes[sizes >= 2].index)]

    arrays = {str(name): part[value].to_numpy(dtype=float) for name, part in data.groupby(group)}
    if len(arrays) < 2:
        raise ValueError("Need at least two groups with two or more observations each.")
    if len(arrays) > MAX_GROUPS:
        raise ValueError(f"Too many groups ({len(arrays)}); the maximum is {MAX_GROUPS}.")

    summaries = [
        {
            "group": name,
            "n": int(len(values)),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std(ddof=1)),
        }
        for name, values in arrays.items()
    ]

    result = {
        "group_column": group,
        "value_column": value,
        "groups": summaries,
        "observations": int(len(data)),
        "rows_dropped": rows_dropped,
        "warnings": [],
    }

    small = [s["group"] for s in summaries if s["n"] < 30]
    if small:
        result["warnings"].append(f"Small groups (n < 30): {small}. Results are less reliable.")
    if rows_dropped:
        result["warnings"].append(f"{rows_dropped} rows dropped because of missing values.")

    if len(arrays) == 2:
        (name_a, a), (name_b, b) = arrays.items()
        welch = stats.ttest_ind(a, b, equal_var=False)
        mann_whitney = stats.mannwhitneyu(a, b, alternative="two-sided")

        pooled_sd = np.sqrt(
            ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
        )
        d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else float("nan")

        result["tests"] = {
            "welch_t": {"statistic": float(welch.statistic), "p_value": float(welch.pvalue)},
            "mann_whitney_u": {"statistic": float(mann_whitney.statistic),
                               "p_value": float(mann_whitney.pvalue)},
        }
        result["mean_difference"] = {
            "first_group": name_a, "second_group": name_b,
            "first_minus_second": float(a.mean() - b.mean()),
        }
        result["effect_size"] = {
            "name": "cohens_d", "value": float(d),
            "magnitude": _magnitude(d, (0.2, 0.5, 0.8)) if np.isfinite(d) else "undefined",
        }
    else:
        values = list(arrays.values())
        anova = stats.f_oneway(*values)
        kruskal = stats.kruskal(*values)

        everything = np.concatenate(values)
        grand_mean = everything.mean()
        ss_between = sum(len(v) * (v.mean() - grand_mean) ** 2 for v in values)
        ss_total = ((everything - grand_mean) ** 2).sum()
        eta_squared = ss_between / ss_total if ss_total > 0 else float("nan")

        result["tests"] = {
            "anova_f": {"statistic": float(anova.statistic), "p_value": float(anova.pvalue)},
            "kruskal_wallis": {"statistic": float(kruskal.statistic), "p_value": float(kruskal.pvalue)},
        }
        result["effect_size"] = {
            "name": "eta_squared", "value": float(eta_squared),
            "magnitude": _magnitude(eta_squared, (0.01, 0.06, 0.14)) if np.isfinite(eta_squared) else "undefined",
        }
        result["note"] = "ANOVA/Kruskal-Wallis say whether any group differs, not which ones."

    return result
