# postprocessing utilities for synthdata
from typing import Optional, Dict
import numpy as np
import pandas as pd
from pandas.api import types as pdt
from synthdata.utils import match_numeric_precision


def convert_synthetic_types(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    *,
    strategy: str = "round",
    clamp: bool = True,
    preserve_cats: bool = True,
    precision_method: str = "auto",
    decimals_map: Optional[Dict[str, int]] = None,
    sig_figs_map: Optional[Dict[str, int]] = None,
    min_resolution: float = 1e-12,
    match_nan_fraction: bool = False,
) -> pd.DataFrame:
    """
    Convert synthetic dataframe `syn` to match dtypes of `real`, with numeric precision matching.

    Parameters described in the README / docstrings above.
    """
    syn_out = syn.copy()

    # 0) precision rounding first
    syn_out = match_numeric_precision(
        real,
        syn_out,
        method=precision_method,
        decimals_map=decimals_map,
        sig_figs_map=sig_figs_map,
        min_resolution=min_resolution,
    )

    # 0b) match NaN fraction if requested
    if match_nan_fraction:
        for col in real.columns:
            if col not in syn_out.columns:
                continue
            real_col = real[col]
            syn_col = syn_out[col]
            p_nan = real_col.isna().mean()
            n = len(syn_col)
            target_nan = int(round(p_nan * n))
            current_nan = syn_col.isna().sum()
            if target_nan > current_nan:
                candidate_idx = syn_col[~syn_col.isna()].index.to_numpy()
                if candidate_idx.size > 0:
                    choose = np.random.choice(candidate_idx, size=min(len(candidate_idx), target_nan - current_nan), replace=False)
                    syn_out.loc[choose, col] = np.nan
            elif target_nan < current_nan:
                # do not attempt to resurrect original values
                pass

    # 1) type conversions
    for col in real.columns:
        if col not in syn_out.columns:
            continue

        real_col = real[col]
        syn_col = syn_out[col]

        # INTEGER (including pandas nullable Int64)
        if pdt.is_integer_dtype(real_col.dtype) or str(real_col.dtype).startswith("Int"):
            if strategy == "round":
                converted = np.rint(syn_col.to_numpy(dtype=float)).astype(float)
            elif strategy == "floor":
                converted = np.floor(syn_col.to_numpy(dtype=float)).astype(float)
            elif strategy == "ceil":
                converted = np.ceil(syn_col.to_numpy(dtype=float)).astype(float)
            else:
                converted = np.rint(syn_col.to_numpy(dtype=float)).astype(float)

            if clamp:
                rmin = real_col.min(skipna=True)
                rmax = real_col.max(skipna=True)
                if pd.notna(rmin) and pd.notna(rmax):
                    converted = np.clip(converted, rmin, rmax)

            if str(real_col.dtype).startswith("Int"):
                syn_out[col] = pd.Series(converted, index=syn_out.index).astype("Int64")
            else:
                conv_series = pd.Series(converted, index=syn_out.index)
                if conv_series.isna().any():
                    fill_val = int(real_col.median(skipna=True)) if pd.notna(real_col.median(skipna=True)) else 0
                    conv_series = conv_series.fillna(fill_val)
                syn_out[col] = conv_series.astype("int64")

        # FLOAT
        elif pdt.is_float_dtype(real_col.dtype):
            if clamp:
                rmin = real_col.min(skipna=True)
                rmax = real_col.max(skipna=True)
                if pd.notna(rmin) and pd.notna(rmax):
                    syn_out[col] = syn_col.clip(lower=rmin, upper=rmax)
                else:
                    syn_out[col] = syn_col
            else:
                syn_out[col] = syn_col

        # BOOLEAN
        elif pdt.is_bool_dtype(real_col.dtype):
            vals = syn_col
            if pdt.is_numeric_dtype(vals):
                arr = vals.to_numpy(dtype=float)
                mn = np.nanmin(arr) if np.isfinite(np.nanmin(arr)) else 0.0
                mx = np.nanmax(arr) if np.isfinite(np.nanmax(arr)) else 1.0
                probs = (arr - mn) / (mx - mn + 1e-12)
                syn_out[col] = pd.Series(probs >= 0.5, index=syn_out.index)
            else:
                syn_out[col] = syn_col.astype(bool)

        # DATETIME
        elif pdt.is_datetime64_any_dtype(real_col.dtype):
            if pdt.is_numeric_dtype(syn_col):
                med = np.nanmedian(syn_col.dropna().astype(float)) if syn_col.dropna().size > 0 else 0
                unit = "ms" if med > 1e12 else "s"
                syn_out[col] = pd.to_datetime(syn_col, unit=unit, errors="coerce")
            else:
                syn_out[col] = pd.to_datetime(syn_col, errors="coerce")

        # CATEGORICAL / OBJECT
        elif pdt.is_categorical_dtype(real_col.dtype) or pdt.is_object_dtype(real_col.dtype):
            if preserve_cats:
                freqs = real_col.value_counts(normalize=True, dropna=False)
                categories = freqs.index.to_list()
                probs = freqs.values
                n = len(syn_col)
                draws = np.random.choice(categories, size=n, p=probs)
                draws = [np.nan if (isinstance(x, float) and np.isnan(x)) else x for x in draws]
                syn_out[col] = pd.Series(draws, index=syn_out.index)
            else:
                syn_out[col] = syn_col.astype(str)

        # fallback
        else:
            try:
                syn_out[col] = syn_col.astype(real_col.dtype)
            except Exception:
                syn_out[col] = syn_col

    return syn_out
