"""
Utility: convert synthetic dataframe columns to match types of an original dataframe.

Usage:
    syn_fixed = convert_synthetic_types(df_original, df_synthetic, strategy="round", clamp=True)

Strategies for integer conversion:
 - 'round' (default): round to nearest integer
 - 'floor': np.floor
 - 'ceil': np.ceil'

Options:
 - clamp: clamp values to original min/max where available
 - preserve_cats: if True, sample categorical values according to original frequencies when dtype is object/categorical
"""
from typing import Optional
import numpy as np
import pandas as pd
from pandas.api import types as pdt


def convert_synthetic_types(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    strategy: str = "round",
    clamp: bool = True,
    preserve_cats: bool = True,
) -> pd.DataFrame:
    syn = syn.copy()
    for col in real.columns:
        if col not in syn.columns:
            continue

        real_col = real[col]
        syn_col = syn[col]

        # --- INTEGER TYPES (including pandas nullable Int64) ---
        if pdt.is_integer_dtype(real_col.dtype) or str(real_col.dtype).startswith("Int"):
            # rounding strategy
            if strategy == "round":
                converted = np.rint(syn_col).astype(float)  # keep float to allow NaN then cast
            elif strategy == "floor":
                converted = np.floor(syn_col).astype(float)
            elif strategy == "ceil":
                converted = np.ceil(syn_col).astype(float)
            else:
                converted = np.rint(syn_col).astype(float)

            # clamp to observed original range if requested
            if clamp:
                rmin = real_col.min(skipna=True)
                rmax = real_col.max(skipna=True)
                if pd.notna(rmin) and pd.notna(rmax):
                    converted = np.clip(converted, rmin, rmax)

            # preserve NaNs: where original has NaN fraction, keep proportion? Here we keep NaN where syn had NaN.
            # Cast back to pandas nullable Int64 if original is nullable, else to standard int (may fail on NaN)
            if str(real_col.dtype).startswith("Int"):
                # nullable integer dtype
                syn[col] = pd.Series(converted).astype("Int64")
            else:
                # conventional int: fill nan with nearest integer (or with 0?) — better to preserve NaN or fill explicitly
                syn[col] = pd.Series(converted).astype("int64")

        # --- FLOAT TYPES ---
        elif pdt.is_float_dtype(real_col.dtype):
            # Optionally clamp to original range
            if clamp:
                rmin = real_col.min(skipna=True)
                rmax = real_col.max(skipna=True)
                if pd.notna(rmin) and pd.notna(rmax):
                    syn[col] = syn_col.clip(lower=rmin, upper=rmax)
                else:
                    syn[col] = syn_col
            else:
                syn[col] = syn_col

        # --- BOOLEAN ---
        elif pdt.is_bool_dtype(real_col.dtype):
            # get threshold 0.5 on normalized probabilities or round
            # If syn_col is continuous, threshold at 0.5 after scaling between 0 and 1
            vals = syn_col
            # Try to coerce numeric-like to probability
            if pdt.is_numeric_dtype(vals):
                probs = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-12)
                syn[col] = probs >= 0.5
            else:
                # fallback: map truthy strings
                syn[col] = syn_col.astype(bool)

        # --- DATETIME ---
        elif pdt.is_datetime64_any_dtype(real_col.dtype):
            # If syn produced numeric values, interpret them as timestamps (try to infer)
            if pdt.is_numeric_dtype(syn_col):
                # assume syn are epochs in seconds; if too large, try milliseconds
                med = np.nanmedian(syn_col.dropna())
                if med > 1e12:
                    unit = "ms"
                else:
                    unit = "s"
                syn[col] = pd.to_datetime(syn_col, unit=unit, errors="coerce")
            else:
                syn[col] = pd.to_datetime(syn_col, errors="coerce")

        # --- CATEGORICAL / OBJECT (strings) ---
        elif pdt.is_categorical_dtype(real_col.dtype) or pdt.is_object_dtype(real_col.dtype):
            # If original was categorical, prefer to sample from original categories by frequency
            if preserve_cats:
                freqs = real_col.value_counts(normalize=True, dropna=False)
                categories = freqs.index.to_list()
                probs = freqs.values
                # Draw new samples with same length as syn_col
                # But keep NaN where syn has NaN
                n = len(syn_col)
                draws = np.random.choice(categories, size=n, p=probs)
                # convert NaN tokens back to actual np.nan where category is NaN
                draws = [np.nan if (isinstance(x, float) and np.isnan(x)) else x for x in draws]
                syn[col] = pd.Series(draws, index=syn.index)
            else:
                # simple cast
                syn[col] = syn_col.astype(str)

        # --- OTHER (leave as-is) ---
        else:
            # fallback: try to cast to the same dtype
            try:
                syn[col] = syn_col.astype(real_col.dtype)
            except Exception:
                syn[col] = syn_col

    return syn