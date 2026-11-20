import re

import pandas as pd

# Strings that mean "missing" in many datasets. Adult uses "?".
MISSING_TOKENS = ["?", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", ""]


def standardize_column_name(name):
    """Turn any header into a safe snake_case identifier."""
    name = re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")
    if not name:
        name = "column"
    if name[0].isdigit():
        name = f"col_{name}"
    return name


def standardize_column_names(columns):
    """Standardize names and make duplicates unique (x, x_2, x_3...)."""
    seen = {}
    result = []
    for column in columns:
        base = standardize_column_name(column)
        seen[base] = seen.get(base, 0) + 1
        result.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return result


def _is_text(series):
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def clean_dataframe(df, missing_tokens=None, convert_numeric=True):
    """
    The single place where raw data becomes analysis-ready data.

    - standardizes column names
    - strips whitespace from text values
    - converts missing-value tokens (like "?") to real NaN
    - converts text columns that are entirely numeric into numbers
    """
    if missing_tokens is None:
        missing_tokens = MISSING_TOKENS

    df = df.copy()
    df.columns = standardize_column_names(df.columns)

    for column in df.columns:
        series = df[column]

        if not _is_text(series):
            continue

        series = series.str.strip()
        series = series.mask(series.isin(missing_tokens))

        if convert_numeric and series.notna().any():
            converted = pd.to_numeric(series, errors="coerce")
            if converted.notna().sum() == series.notna().sum():
                series = converted

        df[column] = series

    return df
