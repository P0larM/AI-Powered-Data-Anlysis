from pathlib import Path

from app.data.cleaner import clean_dataframe
from app.data.loader import load_dataset
from app.data.profiler import columns_of_type, format_profile, profile_dataset
from app.statistical_analysis.categorical import analyze_categorical_features
from app.statistical_analysis.correlation import analyze_correlations
from app.statistical_analysis.multiple_testing import correct_p_values
from app.statistical_analysis.regression import linear_regression

# ---- Dataset-specific settings live ONLY here ------------------------------
DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "adult.data"

ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

TARGET = "income"
REGRESSION_TARGET = "hours_per_week"
REGRESSION_PREDICTORS = ["age", "education_num", "capital_gain", "capital_loss"]
# -----------------------------------------------------------------------------


def section(title):
    print(f"\n========== {title} ==========")


def run_correlations(df, columns):
    for method in ("pearson", "spearman"):
        results = analyze_correlations(df, method=method, columns=columns)

        section(f"{method.upper()} CORRELATION")
        print(results["matrix"].round(3))

        corrected = correct_p_values(results["pairs"])
        print(f"\nStrongest relationships (with corrected p-values):")
        print(corrected.head(10).to_string(index=False))


def run_categorical(df, features):
    analysis = analyze_categorical_features(df, target=TARGET, features=features)

    section("CATEGORICAL ANALYSIS (corrected p-values)")
    corrected = correct_p_values(analysis["results"])
    print(corrected.to_string(index=False))

    if analysis["skipped"]:
        print("\nSkipped:")
        for item in analysis["skipped"]:
            print(f"  {item['feature']}: {item['reason']}")


def run_regression(df):
    results = linear_regression(df, REGRESSION_TARGET, REGRESSION_PREDICTORS)

    section("MULTIPLE LINEAR REGRESSION")
    print(results["coefficients"].round(4))
    print(f"\nR-squared: {results['r_squared']:.4f}")
    print(f"Adjusted R-squared: {results['adjusted_r_squared']:.4f}")
    print(f"F-statistic: {results['f_statistic']:.4f}")
    print(f"F-test p-value: {results['f_p_value']:.6e}")
    print(f"Observations: {results['observations']} "
          f"({results['rows_dropped']} dropped)")
    print(f"VIF: { {k: round(v, 2) for k, v in results['vif'].items()} }")

    if results["warnings"]:
        print("\nWarnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")


def main():
    raw = load_dataset(DATASET_PATH, column_names=ADULT_COLUMNS)
    df = clean_dataframe(raw)

    profile = profile_dataset(df)
    print(format_profile(profile))

    numeric = columns_of_type(profile, "continuous", "discrete")  # weights/IDs excluded
    categorical = columns_of_type(profile, "categorical")

    run_correlations(df, numeric)
    run_categorical(df, categorical)
    run_regression(df)


if __name__ == "__main__":
    main()
