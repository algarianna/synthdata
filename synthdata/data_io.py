"""
Data loading and simple cleaning helpers.

Supported sources:
 - local path
 - URL
 - file-like object with .read()
 - pandas.DataFrame (returned as a copy)
"""
from typing import Optional, Sequence, Tuple, Union, IO
import io
import os
import pandas as pd
import requests
import csv

from .utils import logger

def _read_with_sniffer(path_or_buf, sep_hint=",", **kwargs):
    try:
        if hasattr(path_or_buf, "read"):
            sample = path_or_buf.read(4096)
            path_or_buf.seek(0)
            if isinstance(sample, bytes):
                sample = sample.decode("utf-8", errors="ignore")
        else:
            with open(path_or_buf, "r", encoding=kwargs.get("encoding", "utf-8",)) as f:
                sample = f.read(4096)
        sniff = csv.Sniffer()
        dialect = sniff.sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = sep_hint
    return pd.read_csv(path_or_buf, sep=sep, low_memory=False, **kwargs)


def load_dataset(
    source: Union[str, IO, pd.DataFrame],
    file_type: Optional[str] = None,
    sep: str = ",",
    sheet_name: Union[int, str] = 0,
    nrows: Optional[int] = None,
    encoding: str = "utf-8",
    infer_numeric: bool = True,
    drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
    """
    Load dataset from path/URL/file-like/DataFrame and perform light cleaning.

    Returns a pandas.DataFrame copy.
    """
    if isinstance(source, pd.DataFrame):
        logger.debug("load_dataset: source is already a DataFrame, returning a copy")
        df = source.copy()
        if drop_all_nan_rows:
            df = df.dropna(axis=0, how="all")
        return df

    is_url = isinstance(source, str) and source.lower().startswith(("http://", "https://", "ftp://"))
    source_buf = source
    if is_url:
        logger.debug("load_dataset: fetching URL %s", source)
        resp = requests.get(source, stream=True, timeout=30)
        resp.raise_for_status()
        content = resp.content
        source_buf = io.BytesIO(content)

        if not file_type:
            _, ext = os.path.splitext(source.split("?")[0])
            ext = ext.lower()
            if ext in (".csv", ".tsv", ".txt"):
                file_type = "csv"
            elif ext in (".xls", ".xlsx"):
                file_type = "excel"
            elif ext in (".parquet",):
                file_type = "parquet"
            elif ext in (".json",):
                file_type = "json"

    if not file_type:
        if isinstance(source_buf, str):
            _, ext = os.path.splitext(source_buf)
            ext = ext.lower()
            if ext in (".csv", ".tsv", ".txt"):
                file_type = "csv"
            elif ext in (".xls", ".xlsx"):
                file_type = "excel"
            elif ext in (".parquet",):
                file_type = "parquet"
            elif ext in (".json",):
                file_type = "json"
            else:
                file_type = "csv"
        else:
            file_type = "csv"

    logger.debug("load_dataset: reading as file_type=%s", file_type)
    if file_type in ("csv", "tsv", "txt"):
        sep_hint = "\t" if file_type == "tsv" else sep
        try:
            df = _read_with_sniffer(source_buf, sep_hint, encoding=encoding, nrows=nrows)
        except Exception:
            df = pd.read_csv(source_buf, sep=sep, encoding=encoding, nrows=nrows, low_memory=False)
    elif file_type == "excel":
        df = pd.read_excel(source_buf, sheet_name=sheet_name, nrows=nrows)
    elif file_type == "parquet":
        if hasattr(source_buf, "seek"):
            source_buf.seek(0)
            df = pd.read_parquet(source_buf)
        else:
            df = pd.read_parquet(source_buf)
    elif file_type == "json":
        if hasattr(source_buf, "seek"):
            source_buf.seek(0)
            df = pd.read_json(source_buf)
        else:
            df = pd.read_json(source_buf)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

    # Ensure columns are strings
    df.columns = [str(c) for c in df.columns]

    if drop_all_nan_rows:
        df = df.dropna(axis=0, how="all")

    if infer_numeric:
        for col in df.select_dtypes(include=["object", "string"]).columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            non_na_frac = coerced.notna().mean()
            if non_na_frac > 0.5:
                df[col] = coerced

    return df.copy()


def select_columns(
    df: pd.DataFrame,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    numeric_only: bool = True,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    Return a copy of df with selected columns.
    """
    cols = list(df.columns)
    if include is not None:
        missing = [c for c in include if c not in cols]
        if missing:
            raise KeyError(f"Requested include columns not found: {missing}")
        cols = list(include)
    if exclude is not None:
        cols = [c for c in cols if c not in set(exclude)]
    df_sel = df.loc[:, cols].copy()
    if numeric_only:
        numeric_cols = [c for c in df_sel.columns if pd.api.types.is_numeric_dtype(df_sel[c])]
        df_sel = df_sel[numeric_cols]
        return df_sel, numeric_cols
    else:
        return df_sel, list(df_sel.columns)


def summarize_dataframe(df: pd.DataFrame, nrows: int = 5) -> dict:
    """
    Return a dictionary summary for inspection.
    """
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(nrows).to_dict(orient="list"),
        "missing_per_column": df.isna().mean().to_dict(),
        "numeric_columns": [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
    }


def save_dataset(df: pd.DataFrame, path: str, file_type: Optional[str] = None, index: bool = False):
    """
    Save dataframe to disk. file_type inferred by extension if not provided.
    """
    if not file_type:
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext in (".csv", ".tsv", ".txt"):
            file_type = "csv"
        elif ext in (".xls", ".xlsx"):
            file_type = "excel"
        elif ext in (".parquet",):
            file_type = "parquet"
        elif ext in (".json",):
            file_type = "json"
        else:
            raise ValueError("Cannot infer file_type from path; provide file_type explicitly.")

    if file_type == "csv":
        df.to_csv(path, index=index)
    elif file_type == "parquet":
        df.to_parquet(path, index=index)
    elif file_type == "excel":
        df.to_excel(path, index=index)
    elif file_type == "json":
        df.to_json(path, orient="records", lines=False)
    else:
        raise ValueError(f"Unsupported file_type: {file_type}")
