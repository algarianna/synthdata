"""
Gaussian copula model and an orchestration function.

Improvements added:
 - GaussianCopula class encapsulates state (QuantileTransformer + correlation matrix + column info)
 - Input validation: numeric-only, drop constant columns (and restore them when sampling)
 - NaN handling options: 'drop' (default) or 'impute' (mean/median)
 - Correlation regularization option before Cholesky to improve numerical stability
 - Save/load model using joblib
 - Documentation and comments describing limitations and next steps (Higham, categorical handling, privacy)
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.preprocessing import QuantileTransformer

from .utils import logger, DEFAULT_CONFIG
from .metrics import (
    correlation_matrix_metrics,
    univariate_stats,
    prdc_metrics,
    statistical_sim_score,
    two_sample_classifier_auc,
)


def nearest_pd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Symmetrize and clip eigenvalues to guarantee a positive-definite matrix.

    NOTE: This is a simple eigenvalue clipping approach. It ensures PD-ness but may not
    return the mathematically 'nearest' PD matrix. For minimal distortion, consider
    implementing Higham's algorithm (TODO).
    """
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals_clipped = np.clip(vals, eps, None)
    return (vecs * vals_clipped) @ vecs.T


class GaussianCopula:
    """
    Encapsulates a fitted Gaussian copula model.

    Attributes stored:
      - qt: fitted QuantileTransformer
      - R: correlation matrix in the Gaussian (normal) space
      - columns: ordered list of columns used for modeling (numeric)
      - dropped_constant_columns: dict of columns dropped due to zero variance with their constant values
      - impute_stats: dict of per-column imputation values if impute was used
    """

    def __init__(
        self,
        n_quantiles: Optional[int] = None,
        qt_subsample: Optional[int] = None,
        corr_regularization: float = DEFAULT_CONFIG["corr_regularization"],
        random_state: int = DEFAULT_CONFIG["random_state"],
    ):
        self.n_quantiles = n_quantiles if n_quantiles is not None else DEFAULT_CONFIG["n_quantiles"]
        self.qt_subsample = qt_subsample if qt_subsample is not None else DEFAULT_CONFIG["qt_subsample"]
        self.corr_regularization = float(corr_regularization)
        self.random_state = int(random_state)

        # placeholders set during fit
        self.qt = None
        self.R = None
        self.columns = None
        self.dropped_constant_columns = {}
        self.impute_stats = {}

    def _validate_and_prepare(self, df: pd.DataFrame, handle_na: str = "drop", impute_strategy: str = "mean") -> pd.DataFrame:
        """
        Validate input df, select numeric columns, handle NaNs and drop constant columns.

        handle_na:
          - 'drop' -> drop rows with any NaN (simple and conservative)
          - 'impute' -> fill NaNs columnwise with mean/median per impute_strategy
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Keep only numeric columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found. Gaussian copula requires numeric features.")
        df_num = df.loc[:, numeric_cols].copy()

        # Drop columns with zero variance (constant); store them to restore later
        const_cols = [c for c in df_num.columns if df_num[c].dropna().nunique() <= 1]
        for c in const_cols:
            # preserve the constant value (could be NaN if all missing)
            val = df_num[c].dropna().iloc[0] if df_num[c].dropna().size > 0 else np.nan
            self.dropped_constant_columns[c] = val
            df_num.drop(columns=c, inplace=True)

        # Handle NaNs
        if handle_na == "drop":
            df_num = df_num.dropna(axis=0, how="any")
            self.impute_stats = {}
        elif handle_na == "impute":
            if impute_strategy not in ("mean", "median"):
                raise ValueError("impute_strategy must be 'mean' or 'median'")
            impute_stats = {}
            for c in df_num.columns:
                if impute_strategy == "mean":
                    v = float(np.nanmean(df_num[c].astype(float)))
                else:
                    v = float(np.nanmedian(df_num[c].astype(float)))
                impute_stats[c] = v
                df_num[c] = df_num[c].fillna(v)
            self.impute_stats = impute_stats
        else:
            raise ValueError("handle_na must be 'drop' or 'impute'")

        if df_num.shape[0] < 2:
            raise ValueError("Not enough rows remain after NaN handling to fit the model.")

        return df_num

    def fit(self, df: pd.DataFrame, handle_na: str = "drop", impute_strategy: str = "mean") -> "GaussianCopula":
        """
        Fit QuantileTransformer and estimate correlation matrix in Gaussian space.

        After fitting:
          - self.qt is a trained QuantileTransformer
          - self.R is a positive-definite correlation matrix (regularized)
          - self.columns is the list of columns modeled
        """
        df_ready = self._validate_and_prepare(df, handle_na=handle_na, impute_strategy=impute_strategy)
        self.columns = df_ready.columns.to_list()

        # Fit QuantileTransformer on numeric data
        qt = QuantileTransformer(
            n_quantiles=min(self.n_quantiles, len(df_ready)),
            output_distribution="normal",
            subsample=int(self.qt_subsample),
            random_state=self.random_state,
        )
        Z = qt.fit_transform(df_ready.values)
        R = np.corrcoef(Z, rowvar=False)

        # Regularize correlation for numerical stability before Cholesky (small ridge)
        if self.corr_regularization and self.corr_regularization > 0.0:
            R = R + np.eye(R.shape[0]) * float(self.corr_regularization)

        # Ensure PD (fallback)
        try:
            # Quick PD check via Cholesky
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not PD; applying nearest_pd eigen-clipping fallback.")
            R = nearest_pd(R)

        self.qt = qt
        self.R = R
        return self

    def sample(self, n_samples: int, random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Sample synthetic data and restore any dropped constant columns.

        Returns a DataFrame with the same column order as the original numeric columns used for fit
        plus the constant columns that were dropped (filled with their original constant values).
        """
        if self.qt is None or self.R is None or self.columns is None:
            raise ValueError("Model is not fitted. Call .fit(...) before sampling.")

        rng = np.random.default_rng(self.random_state if random_state is None else random_state)

        # Ensure R is PD for Cholesky; fallback to nearest_pd if necessary
        try:
            L = np.linalg.cholesky(self.R)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky failed; applying nearest_pd to correlation matrix and retrying.")
            R_pd = nearest_pd(self.R)
            L = np.linalg.cholesky(R_pd)

        Z = rng.standard_normal(size=(n_samples, self.R.shape[0])) @ L.T
        X_syn = self.qt.inverse_transform(Z)
        synthetic = pd.DataFrame(X_syn, columns=self.columns)

        # Re-add constant columns (use the stored values)
        for c, val in self.dropped_constant_columns.items():
            synthetic[c] = val
        # Ensure original column order: constants may need to be inserted in original positions
        # If user wants exact original order (including non-numerics), they should store original df columns externally.

        return synthetic

    def save(self, path: str):
        """
        Save this fitted model to disk using joblib.

        Warning: saved model contains the fitted QuantileTransformer and correlation matrix.
        """
        dirname = os.path.dirname(path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "GaussianCopula":
        """
        Load a saved GaussianCopula instance from disk (joblib).
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("Loaded object is not a GaussianCopula instance.")
        return obj


# Backwards-compatible functions (thin wrappers)
def gaussian_copula_fit(df: pd.DataFrame, n_quantiles: int = None, random_state: int = DEFAULT_CONFIG["random_state"]) -> Dict[str, Any]:
    gc = GaussianCopula(n_quantiles=n_quantiles, random_state=random_state)
    gc.fit(df)
    return {"qt": gc.qt, "R": gc.R, "columns": gc.columns}


def gaussian_copula_sample(model: Dict[str, Any], n_samples: int, random_state: int = DEFAULT_CONFIG["random_state"]) -> pd.DataFrame:
    # Accept either a GaussianCopula instance or the older dict-style model
    if isinstance(model, GaussianCopula):
        return model.sample(n_samples=n_samples, random_state=random_state)
    else:
        # legacy dict: build a temporary GaussianCopula-like object
        temp = GaussianCopula()
        temp.qt = model["qt"]
        temp.R = model["R"]
        temp.columns = model["columns"]
        return temp.sample(n_samples=n_samples, random_state=random_state)


def synthesize_and_evaluate(
    df: pd.DataFrame,
    n_samples: Optional[int] = None,
    random_state: int = DEFAULT_CONFIG["random_state"],
    handle_na: str = "drop",
    impute_strategy: str = DEFAULT_CONFIG["impute_strategy"],
    corr_regularization: float = DEFAULT_CONFIG["corr_regularization"],
) -> Tuple[pd.DataFrame, Dict[str, Any], Any, pd.DataFrame]:
    """
    Full pipeline: fit copula, sample, clip extremes, compute evaluation metrics.

    Improvements included in the report:
      - two_sample_auc: AUC from a RandomForest classifier separating real vs synthetic (lower closer to 0.5 is better)
    """
    if n_samples is None:
        n_samples = len(df)

    # Fit model (handles numeric selection / NaNs / constants)
    gc = GaussianCopula(n_quantiles=None, corr_regularization=corr_regularization, random_state=random_state)
    gc.fit(df, handle_na=handle_na, impute_strategy=impute_strategy)
    synthetic = gc.sample(n_samples=n_samples, random_state=random_state + 1)

    # Clip extremes per column (same as before)
    lo_q, hi_q = DEFAULT_CONFIG["clip_quantiles"]
    for c in gc.columns:
        lo, hi = np.quantile(df[c].dropna(), [lo_q, hi_q])
        synthetic[c] = synthetic[c].clip(lo, hi)

    # Evaluation metrics
    sss = statistical_sim_score(df[gc.columns], synthetic[gc.columns])
    corr_pear = correlation_matrix_metrics(df[gc.columns], synthetic[gc.columns], method="pearson")
    corr_spea = correlation_matrix_metrics(df[gc.columns], synthetic[gc.columns], method="spearman")
    prdc = prdc_metrics(df[gc.columns], synthetic[gc.columns], use_pca=True, n_components=DEFAULT_CONFIG["prdc_n_components"], k=DEFAULT_CONFIG["prdc_k"])
    univ = univariate_stats(df[gc.columns], synthetic[gc.columns])

    # Two-sample classifier AUC (useful practical metric: how distinguishable the synthetic data are)
    auc = two_sample_classifier_auc(df[gc.columns], synthetic[gc.columns], random_state=random_state)

    report = {
        "StatisticalSimScore": sss["global_stats"],
        "PRDC": {k: prdc[k] for k in ["IP", "IR", "DEN", "COV"]},
        "Corr(Pearson)": corr_pear,
        "Corr(Spearman)": corr_spea,
        "two_sample_auc": auc,
    }

    return synthetic, report, sss["per_column"], univ