"""
Lightweight CLI to run synthesis from the terminal. Example:
  python -m synthdata.cli --input data.csv --cols a b c --out synthetic.csv --n 1000
"""
import argparse
import pandas as pd
from .data_io import load_dataset, select_columns, save_dataset
from .gaussian_copula import synthesize_and_evaluate

def main(argv=None):
    parser = argparse.ArgumentParser(prog="synthdata-cli")
    parser.add_argument("--input", "-i", required=True, help="Input dataset path or URL")
    parser.add_argument("--cols", "-c", nargs="+", required=False, help="Columns to use (default: numeric detected)")
    parser.add_argument("--out", "-o", default="synthetic.csv", help="Output path for synthetic CSV")
    parser.add_argument("--n", "-n", type=int, default=None, help="Number of synthetic samples (default = input rows)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    args = parser.parse_args(argv)

    df = load_dataset(args.input)
    if args.cols:
        df_sel, _ = select_columns(df, include=args.cols, numeric_only=True)
    else:
        df_sel, _ = select_columns(df, include=None, numeric_only=True)

    synthetic, report, _, _ = synthesize_and_evaluate(df_sel, n_samples=args.n, random_state=args.random_state)
    save_dataset(synthetic, args.out)
    print("Saved synthetic data to", args.out)
    print("Report:", report)


if __name__ == "__main__":
    main()
