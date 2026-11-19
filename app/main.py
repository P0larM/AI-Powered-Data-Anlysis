
from app.data.loader import load_dataset
from app.data.profiler import profile_dataset
from statistical_analysis.correlation import analyze_correlations
from statistical_analysis.categorical import analyze_categorical_features
from statistical_analysis.multiple_testing import correct_p_values
from statistical_analysis.regression import linear_regression


def main():

    file_path = "datasets/adult.data"

    # 1. Load dataset
    df = load_dataset(file_path)

    if df is not None:

        # 2. Profile dataset
        profile_dataset(df)


        # 3. Pearson correlation analysis
        pearson_results = analyze_correlations(df, method="pearson")

        print("\n========== PEARSON CORRELATION ==========")
        print(pearson_results["matrix"].round(3))

        print("\nStrongest Relationships:")
        print(pearson_results["pairs"].head(10).to_string(index=False))

        # 4. Spearman correlation analysis
        spearman_results = analyze_correlations(df, method="spearman")

        print("\n========== SPEARMAN CORRELATION ==========")
        print(spearman_results["matrix"].round(3))

        print("\nStrongest Relationships:")
        print(spearman_results["pairs"].head(10).to_string(index=False))

        # 5. Categorical analysis
        categorical_results = analyze_categorical_features(
            df,
            target="income"
        )

        print("\n========== CATEGORICAL ANALYSIS ==========")
        print(categorical_results.to_string(index=False))

        # Multiple Testing Corrections

        # Correct Pearson correlation p-values
        pearson_corrected = correct_p_values(
            pearson_results["pairs"]
        )

        # Correct Spearman correlation p-values
        spearman_corrected = correct_p_values(
            spearman_results["pairs"]
        )

        # Correct categorical analysis p-values
        categorical_corrected = correct_p_values(
            categorical_results
        )

        print("\n========== PEARSON: CORRECTED P-VALUES ==========")
        print(pearson_corrected.head(10).to_string(index=False))

        print("\n========== SPEARMAN: CORRECTED P-VALUES ==========")
        print(spearman_corrected.head(10).to_string(index=False))

        print("\n========== CATEGORICAL: CORRECTED P-VALUES ==========")
        print(categorical_corrected.to_string(index=False))

        # Multiple Linear Regression

        regression_results = linear_regression(
            df,
            target="hours_per_week",
            predictors=[
                "age",
                "education_num",
                "capital_gain",
                "capital_loss"
            ]
        )

        print("\n========== MULTIPLE LINEAR REGRESSION ==========")

        print("\nCoefficient Results:")
        print(regression_results["coefficients"].round(4))

        print("\nModel Performance:")
        print(f"R-squared: {regression_results['r_squared']:.4f}")
        print(f"Adjusted R-squared: {regression_results['adjusted_r_squared']:.4f}")
        print(f"F-statistic: {regression_results['f_statistic']:.4f}")
        print(f"F-test p-value: {regression_results['f_p_value']:.6e}")
        print(f"Observations: {regression_results['observations']}")

        print("\nFull Regression Summary:")
        print(regression_results["summary"])









if __name__ == "__main__":
    main()
