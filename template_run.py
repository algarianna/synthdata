#!/usr/bin/env python3
"""
template_run.py
================

Example script to generate a synthetic dataset from your own CSV file.

HOW TO USE
----------

1. Put your CSV file in the project folder (or specify its full path).

2. Edit the INPUT_FILE variable below with the path to your dataset.

3. Choose the columns you want to synthesize.
   - If you leave COLUMNS = None, all numeric columns will be used.
   - Otherwise, list the column names you want.

4. (Optional) Change the number of synthetic samples.
   By default, the synthetic dataset will have the same number of rows
   as the original dataset.

5. Run:

       python template_run.py

6. The synthetic dataset will be saved as:
       synthetic_dataset.csv
"""

import json

from synthdata.data_io import load_dataset, select_columns
from synthdata.gaussian_copula import synthesize_and_evaluate
from synthdata.postprocessing import convert_synthetic_types


# ============================================================
# USER SETTINGS
# ============================================================

# Path to your dataset
INPUT_FILE = "my_dataset.csv"

# Output file
OUTPUT_FILE = "synthetic_dataset.csv"

# Columns to synthesize
#
# Option 1:
# Use ALL numeric columns
COLUMNS = None

# Option 2:
# Uncomment and edit this instead
#
# COLUMNS = [
#     "age",
#     "income",
#     "education",
# ]

# Number of synthetic rows
#
# None = same number of rows as original dataset
N_SAMPLES = None

# Random seed (for reproducibility)
RANDOM_SEED = 42

# ============================================================
# OPTIONAL POST-PROCESSING
# ============================================================

POSTPROCESS = True

PRECISION_METHOD = "auto"

# Example:
# DECIMALS_MAP = {"alcohol": 2}
DECIMALS_MAP = None

MATCH_NAN_FRACTION = False


# ============================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================

# Load dataset
df = load_dataset(INPUT_FILE)

# Select columns
if COLUMNS is None:
    df = df.select_dtypes(include=["number"])
else:
    df, _ = select_columns(df, include=COLUMNS)

# Number of synthetic samples
if N_SAMPLES is None:
    N_SAMPLES = len(df)

# Generate synthetic data
synthetic, report, percol_scores, univ_table = synthesize_and_evaluate(
    df,
    n_samples=N_SAMPLES,
    random_state=RANDOM_SEED,
    postprocess=False,
)

print("\nSYNTHESIS REPORT")
print("----------------")
for key, value in report.items():
    print(f"{key}: {value}")

# Optional post-processing
if POSTPROCESS:
    synthetic = convert_synthetic_types(
        real=df,
        syn=synthetic,
        precision_method=PRECISION_METHOD,
        decimals_map=DECIMALS_MAP,
        match_nan_fraction=MATCH_NAN_FRACTION,
    )

# Save synthetic dataset
synthetic.to_csv(OUTPUT_FILE, index=False)

print(f"\nSynthetic dataset saved to: {OUTPUT_FILE}")
