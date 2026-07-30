*** Begin Patch
*** Add File: tests/test_postprocessing.py
+import pandas as pd
+import numpy as np
+from synthdata.postprocessing import convert_synthetic_types
+
+
+def test_convert_types_and_precision():
+    real = pd.DataFrame({
+        "a_int": pd.Series([1,2,3], dtype="int64"),
+        "b_float": pd.Series([0.10, 0.20, 0.30], dtype="float64"),
+        "c_cat": pd.Series(["x","y","x"], dtype="object"),
+        "d_bool": pd.Series([True, False, True], dtype="bool"),
+    })
+    # create synthetic floats that need rounding/casting
+    syn = pd.DataFrame({
+        "a_int": np.array([1.2, 2.8, 3.6]),
+        "b_float": np.array([0.12345, 0.20456, 0.29999]),
+        "c_cat": np.array(["z","z","z"]),
+        "d_bool": np.array([0.6, 0.2, 0.9]),
+    })
+
+    out = convert_synthetic_types(real, syn, precision_method="auto", match_nan_fraction=False)
+    assert out["a_int"].dtype.name in ("int64", "Int64")
+    # b_float should be rounded to similar decimals as real (2 decimals)
+    assert (out["b_float"].round(2) == out["b_float"]).all()
+    assert out["c_cat"].isin(["x","y"]).all() or out["c_cat"].isna().any()
+    assert out["d_bool"].dtype == bool or out["d_bool"].dtype.name == "bool"
+
*** End Patch
