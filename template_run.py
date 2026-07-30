#!/usr/bin/env python3
"""
template_run.py — example/template to run synthesis on your own CSV.

Usage:
  python template_run.py --input data/mydata.csv --out synthetic.csv --cols col1 col2 col3 --n 1000
"""
import argparse
import os
import json

from synthdata.data_io import load_dataset, select_columns
from synthdata.gaussian_copula import synthesize_and_evaluate
from synthdata.postprocessing import convert_synthetic_types


def main(args):
    if args.input:
        df = load_dataset(args.input)
    else:
        raise SystemExit("Please provide --input path to your CSV")

    if args.cols:
        df_sel, cols = select_columns(df, include=list(args.cols))
    else:
        df_sel = df.select_dtypes(include=["number"])
        cols = df_sel.columns.tolist()

    n_samples = args.n or len(df_sel)
    synthetic, report, percol, univ = synthesize_and_evaluate(
        df_sel,
        n_samples=n_samples,
        random_state=args.seed,
        postprocess=False
    )

    print("Synthesis report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    if args.postprocess:
        decimals_map = json.loads(args.decimals_map) if args.decimals_map else None
        synthetic = convert_synthetic_types(
            real=df_sel,
            syn=synthetic,
            precision_method=args.precision_method,
            decimals_map=decimals_map,
            match_nan_fraction=args.match_nan_fraction
        )

    out_path = args.out or "synthetic_dataset.csv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    synthetic.to_csv(out_path, index=False)
    print(f"Synthetic dataset saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data with synthdata (template).")
    parser.add_argument("--input", "-i", help="Path to input CSV file", required=True)
    parser.add_argument("--out", "-o", help="Path to save synthetic CSV", default="synthetic_dataset.csv")
    parser.add_argument("--cols", "-c", nargs="+", help="List of columns to synthesize (space separated). If omitted, all numeric columns are used.")
    parser.add_argument("--n", type=int, help="Number of synthetic samples to generate. Default: same as input rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--postprocess", action="store_true", help="Match types and precision from original data")
    parser.add_argument("--precision-method", choices=["auto","resolution","decimals","fixed"], default="auto")
    parser.add_argument("--decimals-map", type=str, help="JSON map of column->decimals, e.g. '{\"alcohol\":2}'")
    parser.add_argument("--match-nan-fraction", action="store_true", help="Match NaN fraction from original data")
    args = parser.parse_args()
    main(args)
