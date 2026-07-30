"""
Evaluation metrics for synthetic tabular data.

Provides:
 - correlation_matrix_metrics
 - univariate_stats (KS, Wasserstein, mean/std differences)
 - prdc_metrics (IP/IR/Density/Coverage via pymdma)
 - statistical_sim_score (via pymdma StatisticalSimScore)
 - plot_corr_matrices (convenience plotting)
 - two_sample_classifier_auc: practical two-sample test using a classifier (RandomForest)
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# For two-sample classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Try to import pymdma components; present a clear error if missing.
try:
    from pymdma.tabular.measures.synthesis_val import (ImprovedPrecision, ImprovedRecall, Density, Coverage)
    from pymdma.tabular.measures.synthesis_val import StatisticalSimScore
except Exception as e:
    raise ImportError("pymdma is required for prdc_metrics and StatisticalSimScore. Install it with pip.") from e

def correlation_matrix_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
    corr_r = real_df.corr(method=method).values
    corr_s = syn_df.corr(method=method).values
    mask_off = ~np.eye(corr_r.shape[0], dtype=bool)

    diff = corr_r - corr_s
    mae = float(np.mean(np.abs(diff[mask_off])))
    fro = float(np.linalg.norm(diff[mask_off]))
    sign_agree = float(np.mean(np.sign(corr_r[mask_off]) == np.sign(corr_s[mask_off])))
    return {"mae": mae, "fro_off": fro, "sign_agree": sign_agree}


def univariate_stats(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in real_df.columns:
        r = real_df[c].dropna().values
        s = syn_df[c].dropna().values
        mu_r, mu_s = np.mean(r), np.mean(s)
        sd_r, sd_s = np.std(r, ddof=0), np.std(s, ddof=0)
        ks = ks_2samp(r, s, alternative="two-sided").statistic
        wd = wasserstein_distance(r, s)

        rows.append(
            {
                "col": c,
                "mean_diff": float(mu_s - mu_r),
                "std_ratio": float(sd_s / sd_r) if sd_r > 0 else np.nan,
                "ks": float(ks),
                "wass": float(wd),
            }
        )
    return pd.DataFrame(rows)


def prdc_metrics(real_df: pd.DataFrame, syn_df: pd.DataFrame, use_pca: bool = True, n_components: int = 10, k: int = 5) -> Dict[str, Any]:
    scaler = StandardScaler()
    real_std = scaler.fit_transform(real_df.values)
    fake_std = scaler.transform(syn_df.values)

    if use_pca:
        pca = PCA(n_components=min(n_components, real_std.shape[1]))
        real_feat = pca.fit_transform(real_std)
        fake_feat = pca.transform(fake_std)
    else:
        real_feat, fake_feat = real_std, fake_std

    ctx = {}
    ip = ImprovedPrecision(k=k, metric="euclidean", n_workers=4)
    ir = ImprovedRecall(k=k, metric="euclidean", n_workers=4)
    den = Density(k=k, metric="euclidean", n_workers=4)
    cov = Coverage(k=k, metric="euclidean", n_workers=4)

    IP = ip.compute(real_feat, fake_feat, context=ctx)
    IR = ir.compute(real_feat, fake_feat, context=ctx)
    DEN = den.compute(real_feat, fake_feat, context=ctx)
    COV = cov.compute(real_feat, fake_feat, context=ctx)

    return {
        "IP": float(IP.dataset_level.value),
        "IR": float(IR.dataset_level.value),
        "DEN": float(DEN.dataset_level.value),
        "COV": float(COV.dataset_level.value),
        "IP_instance": np.array(IP.instance_level.value),
        "IR_instance": np.array(IR.instance_level.value),
    }


def statistical_sim_score(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Dict[str, Any]:
    real_data = real_df.to_numpy()
    syn_data = syn_df.to_numpy()
    col_map = {c: {"type": {"tag": "continuous"}} for c in real_df.columns}

    sim = StatisticalSimScore(col_map=col_map).compute(real_data, syn_data)
    return {"per_column": sim.dataset_level.value, "global_stats": sim.dataset_level.stats}


def plot_corr_matrices(real_df: pd.DataFrame, syn_df: pd.DataFrame, method: str = "pearson", title_suffix: str = ""):
    corr_r = real_df.corr(method=method)
    corr_s = syn_df.corr(method=method)
    diff = corr_s - corr_r

    vmax = 1.0
    vmin = -1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.heatmap(corr_r, vmin=vmin, vmax=vmax, center=0, cmap="coolwarm", annot=False, square=True, ax=axes[0])
    axes[0].set_title(f"Correlazioni {method.capitalize()} — Reale{title_suffix}")

    sns.heatmap(corr_s, vmin=vmin, vmax=vmax, center=0, cmap="coolwarm", annot=False, square=True, ax=axes[1])
    axes[1].set_title(f"Correlazioni {method.capitalize()} — Sintetico{title_suffix}")

    dmax = np.nanmax(np.abs(diff.values[np.triu_indices_from(diff, k=1)]))
    dlim = max(0.05, float(dmax))
    sns.heatmap(diff, vmin=-dlim, vmax=dlim, center=0, cmap="bwr", annot=False, square=True, ax=axes[2])
    axes[2].set_title(f"Δ correlazioni (Synth − Real) — {method.capitalize()}{title_suffix}")
    plt.tight_layout()
    plt.show()

def two_sample_classifier_auc(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> float:
    """Two-sample test via classifier AUC. Lower AUC (~0.5) => more realistic synthetic data."""
    if len(real_df) == 0 or len(syn_df) == 0:
        raise ValueError("real_df and syn_df must be non-empty")

    # Merge datasets
    X = pd.concat([real_df.reset_index(drop=True), syn_df.reset_index(drop=True)], ignore_index=True)
    y = np.concatenate([np.zeros(len(real_df)), np.ones(len(syn_df))])

    # Keep numeric columns only (try convert objects to numeric first)
    for col in X.select_dtypes(include=[object]).columns:
        try:
            X[col] = pd.to_numeric(X[col])
        except Exception:
            X = X.drop(columns=[col])

    X = X.select_dtypes(include=[np.number])  # ensure numeric-only

    if X.shape[1] == 0:
        raise ValueError("No numeric columns available for classification after dropping/conv.")

    # If stratify would be invalid (very small groups), disable it
    strat = y if (min(y.sum(), len(y) - y.sum()) >= 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, probs))
    return auc
