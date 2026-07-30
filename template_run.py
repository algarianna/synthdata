from synthdata.data_io import load_dataset
from synthdata.gaussian_copula import synthesize_and_evaluate
from synthdata.metrics import plot_corr_matrices
import matplotlib

def main():
    # 1. Load your dataset
    # Replace with your own file path
    df = load_dataset("your_dataset.csv")

    # 2. Select the columns you want to synthesize
    columns_to_keep = [
        "column1",
        "column2",
        "column3",
    ]
    df = df[columns_to_keep]

    # 3. Generate synthetic data
    synthetic_df, report, percol_scores, univ_table = synthesize_and_evaluate(
        df,
        n_samples=len(df),      # or choose any number of samples
        random_state=123
    )

    # 4. Save the synthetic dataset
    synthetic_df.to_csv("synthetic_dataset.csv", index=False)

    # 5. Print evaluation results
    print(report)
    print(univ_table)

    # 6. Plot correlation matrices
    plot_corr_matrices(df, synthetic_df, method="pearson")
    plot_corr_matrices(df, synthetic_df, method="spearman")


if __name__ == "__main__":
    matplotlib.use("TkAgg")
    main()