"""
Utility helpers and small configuration for synthdata package.
Merged precision utilities (match_numeric_precision and helpers) here to avoid utils module/package collision.
"""
from typing import Dict, Any, Optional
import logging

# Basic logger for package users to configure as they like
logger = logging.getLogger("synthdata")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s synthdata: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)

# Default configuration
# Note: these are sensible defaults. Users can override via function arguments.
DEFAULT_CONFIG = {
    "n_quantiles": 1000,
    "qt_subsample": int(1e9),
    "clip_quantiles": (0.001, 0.999),
    "prdc_k": 5,
    "prdc_n_components": 10,
    "random_state": 42,
    # Improvements added:
    "corr_regularization": 1e-6,  # small ridge added to diagonal of estimated correlation before Cholesky
    "impute_strategy": "mean",     # used if impute option selected (other option could be 'median')
}

# --- BEGIN precision utilities (moved from synthdata/utils/precision.py) ---
from decimal import Decimal, InvalidOperation
import numpy as np
import pandas as pd
import math
import statistics


def _infer_resolution(series: pd.Series, min_positive=1e-12) -> float:
    vals = series.dropna().to_numpy()
    if vals.size < 2:
        return 0.0
    try:
        uniq = np.unique(vals)
    except Exception:
        uniq = np.unique(series.dropna().astype(float).to_numpy())
    if uniq.size < 2:
        return 0.0
    uniq_sorted = np.sort(uniq)
    diffs = np.diff(uniq_sorted)
    diffs = diffs[diffs > min_positive]
    if diffs.size == 0:
        return 0.0
    return float(np.percentile(diffs, 5.0))


def _infer_decimals_from_strings(series: pd.Series, default=6) -> int:
    counts = []
    for v in series.dropna().astype(str):
        try:
            d = Decimal(v)
        except InvalidOperation:
            try:
                d = Decimal(v.replace(",", ""))
            except Exception:
                continue
        exp = d.as_tuple().exponent
        if exp < 0:
            counts.append(-exp)
    if not counts:
        return default
    return int(statistics.median(counts))


def _round_to_resolution(arr: np.ndarray, res: float) -> np.ndarray:
    if res <= 0:
        return arr
    mask = np.isfinite(arr)
    out = arr.copy()
    if np.any(mask):
        out_masked = np.round(out[mask] / res) * res
        out[mask] = out_masked
    return out


def _round_to_decimals(arr: np.ndarray, decimals: int) -> np.ndarray:
    return np.round(arr, decimals=decimals)


def match_numeric_precision(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    method: str = "auto",
    decimals_map: Optional[Dict[str, int]] = None,
    sig_figs_map: Optional[Dict[str, int]] = None,
    min_resolution: float = 1e-12,
) -> pd.DataFrame:
    syn_out = syn.copy()
    decimals_map = decimals_map or {}
    sig_figs_map = sig_figs_map or {}

    for col in real.columns:
        if col not in syn_out.columns:
            continue
        if not pd.api.types.is_numeric_dtype(real[col]):
            continue

        real_col = real[col]
        syn_col = syn_out[col].to_numpy(dtype=float)

        if col in sig_figs_map:
            sig = int(sig_figs_map[col])

            def round_sig_array(a, sigfigs):
                out = a.copy()
                mask = np.isfinite(a)
                if not np.any(mask):
                    return out
                vals = a[mask]
                # avoid log10(0)
                mags = np.floor(np.log10(np.abs(vals) + 1e-20)).astype(int)
                factors = np.power(10.0, mags - (sigfigs - 1))
                out[mask] = np.round(vals / factors) * factors
                return out

            syn_col = round_sig_array(syn_col, sig)
            syn_out[col] = syn_col
            continue

        if col in decimals_map:
            dec = int(decimals_map[col])
            syn_col = _round_to_decimals(syn_col, dec)
            syn_out[col] = syn_col
            continue

        if method in ("auto", "resolution"):
            res = _infer_resolution(real_col, min_positive=min_resolution)
            if method == "resolution" or (method == "auto" and res >= min_resolution):
                if res >= 1:
                    decimals = 0
                else:
                    decimals = max(0, int(math.ceil(-math.log10(res))))
                syn_col = _round_to_resolution(syn_col, res) if res > 0 else syn_col
                syn_col = _round_to_decimals(syn_col, decimals)
                syn_out[col] = syn_col
                continue

        if method in ("auto", "decimals"):
            inferred = _infer_decimals_from_strings(real_col)
            syn_col = _round_to_decimals(syn_col, inferred)
            syn_out[col] = syn_col
            continue

        syn_out[col] = syn_col

    return syn_out

# --- END precision utilities ---
