import pandas as pd


def load_dataset(file_path, column_names=None, has_header=True, sep=",", skiprows=None):
    """
    Load a delimited text file into a DataFrame.

    Nothing here is specific to any dataset.

    Parameters:
        file_path:    Path to the file.
        column_names: Optional list of names. Use this when the file has no
                      header row (like the raw Adult files).
        has_header:   Whether the first row is a header. Ignored if
                      column_names is given.
        sep:          Delimiter. Pass None to let pandas sniff it (slower).
        skiprows:     Junk lines to skip at the top (the Adult test file
                      has one).
    """
    if column_names is not None:
        header, names = None, column_names
    else:
        header, names = (0 if has_header else None), None

    read_kwargs = {}
    if sep is None:
        read_kwargs["engine"] = "python"

    return pd.read_csv(
        file_path,
        sep=sep,
        header=header,
        names=names,
        skiprows=skiprows,
        skipinitialspace=True,
        **read_kwargs,
    )
