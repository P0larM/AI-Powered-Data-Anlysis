import pandas as pd

def load_dataset(file_path):

    column_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income"
    ]

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        skipinitialspace=True
    )

    return df