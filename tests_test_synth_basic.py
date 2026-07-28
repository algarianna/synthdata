"""
Minimal pytest tests: check that synthesize_and_evaluate runs on small synthetic data.
"""
import numpy as np
import pandas as pd
from synthdata.gaussian_copula import synthesize_and_evaluate

def test_synthesize_on_simple_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=200),
        "b": 2 * rng.normal(size=200) + 0.5 * rng.normal(size=200),
        "c": rng.exponential(scale=1.0, size=200)
    })
    synthetic, report, percol, univ = synthesize_and_evaluate(df, n_samples=200, random_state=0)
    assert synthetic.shape == df.shape
    assert "PRDC" in report
    assert univ.shape[0] == df.shape[1]