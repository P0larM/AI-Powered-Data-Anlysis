
import pandas as pd


def profile_dataset(df):

    print("\n========== DATASET PROFILE ==========")

    # 1. General dataset information
    print("\n--- Dataset Overview ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # 2. Column information
    print("\n--- Column Information ---")
    print(df.dtypes)

    numerical_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(exclude="number").columns

    print("\nNumerical Variables:")
    print(list(numerical_columns))

    print("\nCategorical Variables:")
    print(list(categorical_columns))

    # 3. Missing values
    print("\n--- Missing Values ---")

    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100

    missing_summary = pd.DataFrame({
        "Missing Count": missing_values,
        "Missing Percentage": missing_percentages
    })

    print(missing_summary)

    # 4. Duplicate rows
    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    # 5. Unique values
    print("\n--- Unique Values ---")
    print(df.nunique())

    # 6. Numerical descriptive statistics
    print("\n--- Numerical Statistics ---")
    print(df[numerical_columns].describe().T)

    # 7. Categorical distributions
    print("\n--- Categorical Distributions ---")

    for column in categorical_columns:
        print(f"\n{column}:")
        print(
            df[column]
            .value_counts(dropna=False)
            .to_frame("Count")
            .assign(
                Percentage=lambda x: x["Count"] / len(df) * 100
            )
        )
