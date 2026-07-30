*** Begin Patch
*** Update File: synthdata/__init__.py
@@
 from .metrics import (
     correlation_matrix_metrics,
     univariate_stats,
     prdc_metrics,
     statistical_sim_score,
     plot_corr_matrices,
     two_sample_classifier_auc,
 )
+from .postprocessing import convert_synthetic_types
@@
     "plot_corr_matrices",
     "two_sample_classifier_auc",
+    "convert_synthetic_types",
 ]
*** End Patch
