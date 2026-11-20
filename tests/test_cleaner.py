import numpy as np
import pandas as pd

from app.data.cleaner import clean_dataframe, standardize_column_names


def test_question_marks_become_nan_and_whitespace_is_stripped():
    raw = pd.DataFrame({"Work-Class": [" Private", " ?", "Private "]})
    df = clean_dataframe(raw)
    assert list(df.columns) == ["work_class"]
    assert df["work_class"].isna().sum() == 1
    assert set(df["work_class"].dropna()) == {"Private"}


def test_numeric_text_columns_are_converted():
    df = clean_dataframe(pd.DataFrame({"age": ["39", " ?", "50"]}))
    assert pd.api.types.is_numeric_dtype(df["age"])
    assert df["age"].isna().sum() == 1


def test_duplicate_and_odd_column_names():
    assert standardize_column_names(["a b", "a-b", "1x"]) == ["a_b", "a_b_2", "col_1x"]
