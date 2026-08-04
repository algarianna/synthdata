"""Compatibility shim for precision utilities.

The original implementation was consolidated into synthdata.utils to avoid the
module vs package collision (synthdata/utils.py vs synthdata/utils/precision.py).
This shim keeps the old import path working while delegating to the canonical
implementation in synthdata.utils.
"""
from __future__ import annotations
import warnings
from typing import Optional, Dict
import pandas as pd

# Delegate to consolidated implementation
from synthdata.utils import match_numeric_precision  # noqa: E402

warnings.warn(
    "synthdata.utils.precision is deprecated and its implementation has been moved to synthdata.utils; "
    "please import match_numeric_precision from synthdata.utils instead.",
    DeprecationWarning,
)

__all__ = ["match_numeric_precision"]
