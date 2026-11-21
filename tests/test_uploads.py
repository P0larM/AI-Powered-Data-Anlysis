import numpy as np
import pytest

from app.api.uploads import UploadError, detect_delimiter, parse_upload


@pytest.mark.parametrize("sep", [",", "\t", ";", "|"])
def test_detects_common_delimiters(sep):
    text = "\n".join(sep.join(row) for row in [["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]])
    assert detect_delimiter(text) == sep


def test_comma_followed_by_spaces_is_still_a_comma():
    text = "39, State-gov, 77516, Bachelors\n50, Private, 83311, HS-grad\n38, Private, 215646, HS-grad\n"
    assert detect_delimiter(text) == ","


def test_parse_cleans_names_and_missing_values():
    df = parse_upload(b"First Name, Age\nAna, 39\nBo, ?\n")
    assert list(df.columns) == ["first_name", "age"]
    assert df["age"].isna().sum() == 1
    assert df["age"].dropna().iloc[0] == 39


def test_headerless_files():
    df = parse_upload(b"1,2\n3,4\n", has_header=False)
    assert list(df.columns) == ["column_0", "column_1"] or df.shape == (2, 2)


def test_latin1_and_bom_files_are_readable():
    assert parse_upload("name,city\nJos\xe9,Montr\xe9al\n".encode("latin-1")).shape == (1, 2)
    assert list(parse_upload(b"\xef\xbb\xbfa,b\n1,2\n").columns) == ["a", "b"]


@pytest.mark.parametrize("content", [b"", b"   \n", b"a,b\n", b"just one column\nx\ny\n"])
def test_unusable_files_are_rejected(content):
    with pytest.raises(UploadError):
        parse_upload(content)