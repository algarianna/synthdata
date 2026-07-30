"""
Small script that ties everything together and demonstrates running the pipeline on the Wine dataset.
Run: python example_run.py
"""
from ucimlrepo import fetch_ucirepo
from synthdata.data_io import load_dataset, select_columns
from synthdata.gaussian_copula import synthesize_and_evaluate
from synthdata.metrics import plot_corr_matrices


def main():
    # Fetch the same wine dataset you used
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features
    columns_to_keep = [
        "fixed_acidity",
        "volatile_acidity",
        "citric_acid",
        "residual_sugar",
        "density",
        "pH",
        "alcohol",
    ]
    df = X[columns_to_keep]

    synthetic_gc, report_gc, percol_scores_gc, univ_table = synthesize_and_evaluate(df, n_samples=len(df), random_state=123)

    print("Report:")
    for k, v in report_gc.items():
        print(k, ":", v)

    print("\nTop columns by KS (worst matches):")
    print(univ_table.sort_values("ks", ascending=False).head(5))

    plot_corr_matrices(df, synthetic_gc, method="pearson", title_suffix=" (Wine)")
    plot_corr_matrices(df, synthetic_gc, method="spearman", title_suffix=" (Wine)")


if __name__ == "__main__":
    main()
