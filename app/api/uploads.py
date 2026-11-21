import io

import pandas as pd

from app.data.cleaner import clean_dataframe
from app.data.loader import load_dataset

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_COLUMNS = 500
CANDIDATE_DELIMITERS = [",", "\t", ";", "|"]


class UploadError(ValueError):
    """The uploaded file could not be turned into a usable table."""


def decode_bytes(content):
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1")


def detect_delimiter(text):
    """
    Pick the delimiter that appears a consistent number of times per line.
    (csv.Sniffer can mistake the space in "a, b, c" for the delimiter.)
    """
    lines = [line for line in text.splitlines()[:20] if line.strip()]
    best, best_score = ",", (0, 0)
    for delimiter in CANDIDATE_DELIMITERS:
        counts = [line.count(delimiter) for line in lines]
        if not counts or max(counts) == 0:
            continue
        modal = max(set(counts), key=counts.count)
        score = (counts.count(modal) / len(counts), modal)
        if modal > 0 and score > best_score:
            best, best_score = delimiter, score
    return best


def parse_upload(content, has_header=True):
    """Bytes of a delimited text file -> cleaned DataFrame."""
    if not content.strip():
        raise UploadError("The file is empty.")

    text = decode_bytes(content)
    delimiter = detect_delimiter(text)

    try:
        df = load_dataset(io.StringIO(text), has_header=has_header, sep=delimiter)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as error:
        raise UploadError(f"Could not read the file as a delimited table: {error}") from error

    df = clean_dataframe(df)

    if df.empty:
        raise UploadError("The file has no data rows.")
    if df.shape[1] < 2:
        raise UploadError("Only one column was found. Check the delimiter (comma, tab, semicolon or pipe).")
    if df.shape[1] > MAX_COLUMNS:
        raise UploadError(f"Too many columns ({df.shape[1]}); the maximum is {MAX_COLUMNS}.")

    return df