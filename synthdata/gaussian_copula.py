*** Begin Patch
*** Update File: synthdata/gaussian_copula.py
@@
 def synthesize_and_evaluate(
     df: pd.DataFrame,
     n_samples: Optional[int] = None,
     random_state: int = DEFAULT_CONFIG["random_state"],
     handle_na: str = "drop",
     impute_strategy: str = DEFAULT_CONFIG["impute_strategy"],
     corr_regularization: float = DEFAULT_CONFIG["corr_regularization"],
+    # postprocessing options
+    postprocess: bool = False,
+    precision_method: str = "auto",
+    decimals_map: Optional[Dict[str, int]] = None,
+    sig_figs_map: Optional[Dict[str, int]] = None,
+    match_nan_fraction: bool = False,
 ) -> Tuple[pd.DataFrame, Dict[str, Any], Any, pd.DataFrame]:
@@
     # Two-sample classifier AUC (useful practical metric: how distinguishable the synthetic data are)
     auc = two_sample_classifier_auc(df[gc.columns], synthetic[gc.columns], random_state=random_state)
@@
     report = {
         "StatisticalSimScore": sss["global_stats"],
         "PRDC": {k: prdc[k] for k in ["IP", "IR", "DEN", "COV"]},
         "Corr(Pearson)": corr_pear,
         "Corr(Spearman)": corr_spea,
         "two_sample_auc": auc,
     }
-
-    return synthetic, report, sss["per_column"], univ
+
+    # Optional postprocessing: match types and numeric precision to the real data
+    if postprocess:
+        from .postprocessing import convert_synthetic_types
+
+        synthetic = convert_synthetic_types(
+            real=df,
+            syn=synthetic,
+            precision_method=precision_method,
+            decimals_map=decimals_map,
+            sig_figs_map=sig_figs_map,
+            match_nan_fraction=match_nan_fraction,
+            clamp=True,
+        )
+
+    return synthetic, report, sss["per_column"], univ
*** End Patch
