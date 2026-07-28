"""
Utility helpers and small configuration for synthdata package.
"""
from typing import Dict, Any
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