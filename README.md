# synthdata

synthdata is a small package for generating synthetic tabular data using a Gaussian copula, with a set of evaluation metrics (KS/Wasserstein, correlation differences, PRDC, StatisticalSimScore via `pymdma`, and a practical two-sample classifier test).

Usage
-----
1. Place your CSV file in the project folder.  
2. Change the filename in:
   `load_dataset("your_dataset.csv")`  
3. Replace the column names in `columns_to_keep` with the columns you want to synthesize.  
4. Run:
   `python template_run.py`  
5. The synthetic dataset will be saved as:
   `synthetic_dataset.csv`

What this repository provides
-----------------------------
- a Python package `synthdata` with separate modules (I/O, model, metrics, utilities)  
- a `GaussianCopula` class with `fit`, `sample`, `save`, `load`  
- a `synthesize_and_evaluate` function that runs fit → sample → clipping → evaluations  
- a minimal CLI `synthdata-cli`  
- an example `example_run.py` and a simple Streamlit app `app_streamlit.py`
- a template `template_run.py` to use on your own dataset.

Prerequisites
-------------
- Python 3.8+  
- It is recommended to create a virtualenv (venv/conda) before installing dependencies.

Installation
-------------
1. Clone the repository (or copy the files) into your working folder.

2. Install in editable mode (development):
```bash
python -m pip install -e .
```

3. Install with extras for the Streamlit app:
```bash
python -m pip install -e ".[streamlit]"
```

4. Install for development (tests/format):
```bash
python -m pip install -e ".[dev]"
```

Quick usage (Python example)
------------------------------
Quick example to generate synthetic data from a DataFrame `df` (numeric columns only):

```python
from synthdata import synthesize_and_evaluate, load_dataset, select_columns

df = load_dataset("data/winequality-red.csv")   # or use a loaded DataFrame
df_sel, cols = select_columns(df, include=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','density','pH','alcohol'])
synthetic, report, percol, univ = synthesize_and_evaluate(df_sel, n_samples=len(df_sel), random_state=42)

print("Report:", report)
```

CLI
---
After installation, you can use:
```bash
synthdata-cli --input data/mydata.csv --out synthetic.csv --n 1000
```
See `synthdata/cli.py` for all options.

Streamlit app
-------------
For the example web interface:
```bash
streamlit run app_streamlit.py
```

Saving and loading the model
----------------------------
The `GaussianCopula` class implements:
- `model.save(path)` → saves the instance (joblib)  
- `GaussianCopula.load(path)` → loads the model

Included evaluations
-------------------
- `correlation_matrix_metrics` — errors between correlation matrices (mae, off-diagonal Frobenius norm, sign agreement)  
- `univariate_stats` — per-column mean diff, std ratio, KS, Wasserstein  
- `prdc_metrics` — IP/IR/Coverage/Density via `pymdma`  
- `statistical_sim_score` — per-column and global via `pymdma`  
- `two_sample_classifier_auc` — AUC of a RandomForest distinguishing real vs synthetic (value close to 0.5 indicates the classifier cannot distinguish)

Important notes
---------------
- The model assumes continuous numeric columns. Categorical columns are not handled automatically.  
- `pymdma` is a key dependency for PRDC/StatisticalSimScore. If you don’t want to use it, alternatives can be provided.  
- Privacy: synthetic data is not automatically private. If your data is sensitive, consider adding mechanisms such as Differential Privacy and membership-inference testing.  
- For numerical stability a small regularization term is added to the diagonal of the estimated correlation matrix; for a minimal-distortion nearest-PD solution consider implementing Higham’s algorithm (TODO).

Testing
-------
Run tests with:
```bash
pytest tests/
```