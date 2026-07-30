*** Begin Patch
*** Update File: README.md
@@
 ## Usage
@@
 +
 +Post-processing: preserve types & precision
 +------------------------------------------
 +After sampling, you may want to convert the synthetic DataFrame to match the original column types and numeric precision.
 +
 +You can either:
 +
 +- call `synthesize_and_evaluate(..., postprocess=True, precision_method="auto")`
 +- or call `convert_synthetic_types(original_df, synthetic_df, precision_method="auto", match_nan_fraction=True)`
 +
 +This ensures integers remain integers, datetimes are parsed, categorical values are sampled from original categories, and numeric columns keep their typical decimal resolution.
 +
*** End Patch
