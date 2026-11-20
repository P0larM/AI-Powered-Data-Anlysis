import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

WEIGHT_NAME_HINTS = ("weight", "wgt")
MAX_DISCRETE_VALUES = 20
MAX_CATEGORICAL_VALUES = 50
LONG_TEXT_LENGTH = 50


def classify_column(series):
    """
    Guess a column's analytical type from its data, not just its dtype.

    Returns one of: continuous, discrete, categorical, id, datetime,
    text, high_cardinality, constant, empty.
    """
    values = series.dropna()
    n = len(values)
    n_unique = values.nunique()

    if n == 0:
        return "empty"
    if n_unique == 1:
        return "constant"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_bool_dtype(series):
        return "categorical"

    if is_numeric_dtype(series):
        is_integer = bool((values % 1 == 0).all())
        if is_integer and n > 20 and n_unique / n > 0.95:
            return "id"
        if n_unique <= 2:
            return "categorical"
        if n_unique <= MAX_DISCRETE_VALUES:
            return "discrete"
        return "continuous"

    average_length = values.astype(str).str.len().mean()
    if average_length > LONG_TEXT_LENGTH:
        return "text"
    if n > 20 and n_unique / n > 0.9:
        return "id"
    if n_unique > MAX_CATEGORICAL_VALUES:
        return "high_cardinality"
    return "categorical"


def is_weight_column(name):
    """Survey-style weight columns (like fnlwgt) are not normal features."""
    return any(hint in name.lower() for hint in WEIGHT_NAME_HINTS)


def profile_dataset(df):
    """
    Return a plain-dict profile of the dataset (JSON-friendly).
    The same dict can be shown to a user or given to an LLM.
    """
    columns = {}

    for name in df.columns:
        series = df[name]

        info = {
            "dtype": str(series.dtype),
            "type": classify_column(series),
            "is_weight": is_weight_column(name),
            "null_count": int(series.isna().sum()),
            "null_pct": round(float(series.isna().mean() * 100), 2),
            "n_unique": int(series.nunique()),
        }

        if is_numeric_dtype(series) and not is_bool_dtype(series) and series.notna().any():
            info.update(
                min=float(series.min()),
                median=float(series.median()),
                mean=float(series.mean()),
                max=float(series.max()),
                std=float(series.std()) if series.notna().sum() > 1 else None,
                zero_pct=round(float((series == 0).mean() * 100), 2),
            )
        elif not is_numeric_dtype(series):
            counts = series.value_counts().head(10)
            info["top_values"] = {str(k): int(v) for k, v in counts.items()}

        columns[name] = info

    return {
        "rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": columns,
    }


def columns_of_type(profile, *types, include_weights=False):
    """Column names whose profiled type is one of `types`."""
    return [
        name
        for name, info in profile["columns"].items()
        if info["type"] in types and (include_weights or not info["is_weight"])
    ]


def format_profile(profile):
    """Human-readable text version of a profile (for dev/CLI use only)."""
    lines = [
        "========== DATASET PROFILE ==========",
        f"Rows: {profile['rows']}   Columns: {profile['n_columns']}   "
        f"Duplicate rows: {profile['duplicate_rows']}",
        "",
        f"{'column':<20}{'type':<18}{'nulls %':>8}{'unique':>8}",
        "-" * 54,
    ]
    for name, info in profile["columns"].items():
        flag = " (weight)" if info["is_weight"] else ""
        lines.append(
            f"{name:<20}{info['type'] + flag:<18}"
            f"{info['null_pct']:>8}{info['n_unique']:>8}"
        )
    return "\n".join(lines)
