# synthdata

synthdata è un piccolo package per generare dati sintetici tabellari usando una Gaussian copula, con un set di metriche di valutazione (KS/Wasserstein, differenze di correlazione, PRDC, StatisticalSimScore via `pymdma`, e un test pratico two-sample basato su classificatore).

1. Place your CSV file in the project folder.
2. Change the filename in:
       load_dataset("your_dataset.csv")
3. Replace the column names in `columns_to_keep` with the columns you want to synthesize.
4. Run:
       python template_run.py
5. The synthetic dataset will be saved as:
       synthetic_dataset.csv

Questo repository fornisce:
- un package Python `synthdata` con moduli separati (I/O, modello, metriche, utilità),
- una classe `GaussianCopula` con `fit`, `sample`, `save`, `load`,
- una funzione `synthesize_and_evaluate` che esegue fit → sample → clipping → valutazioni,
- un CLI minimale `synthdata-cli`,
- un esempio `example_run.py` e una semplice app Streamlit `app_streamlit.py`.

Prerequisiti
-------------
- Python 3.8+
- Si raccomanda di creare un virtualenv (venv/conda) prima di installare le dipendenze.

Installazione
-------------
1. Clone del repository (o copia dei file) nella cartella di lavoro.

2. Installazione in editable mode (sviluppo):
```bash
python -m pip install -e .
```

3. Installazione con gli extras per l'app Streamlit:
```bash
python -m pip install -e ".[streamlit]"
```

4. Installazione per sviluppo (test/format):
```bash
python -m pip install -e ".[dev]"
```

Uso veloce (esempio in Python)
------------------------------
Esempio rapido per generare sintetici a partire da un DataFrame `df` (solo colonne numeriche):

```python
from synthdata import synthesize_and_evaluate, load_dataset, select_columns

df = load_dataset("data/winequality-red.csv")   # o un DataFrame già caricato
df_sel, cols = select_columns(df, include=['fixed_acidity','volatile_acidity','citric_acid','residual_sugar','density','pH','alcohol'])
synthetic, report, percol, univ = synthesize_and_evaluate(df_sel, n_samples=len(df_sel), random_state=42)

print("Report:", report)
```

CLI
---
Dopo installazione, puoi usare il comando:
```bash
synthdata-cli --input data/mydata.csv --out synthetic.csv --n 1000
```
Vedi `synthdata/cli.py` per tutte le opzioni.

Streamlit app
-------------
Per l'interfaccia web di esempio:
```bash
streamlit run app_streamlit.py
```

Salvataggio e caricamento del modello
------------------------------------
La classe `GaussianCopula` implementa:
- `model.save(path)` → salva l'istanza (joblib)
- `GaussianCopula.load(path)` → carica il modello

Valutazioni incluse
-------------------
- `correlation_matrix_metrics` — errori tra matrici di correlazione (mae, Frobenius off-diagonal, sign agreement)
- `univariate_stats` — per-colonna mean diff, std ratio, KS, Wasserstein
- `prdc_metrics` — IP/IR/Coverage/Density via `pymdma`
- `statistical_sim_score` — per-colonna e global via `pymdma`
- `two_sample_classifier_auc` — AUC di un RandomForest che distingue reale vs sintetico (valore vicino a 0.5 indica che il classificatore non distingue)

Note importanti
---------------
- Il modello assume colonne continue e numeriche. Le colonne categoriche non vengono gestite automaticamente.
- `pymdma` è una dipendenza chiave per PRDC/StatisticalSimScore. Se non vuoi usarla, è possibile fornire alternative.
- Privacy: i dati sintetici non sono automaticamente privati. Se i tuoi dati sono sensibili, valuta aggiunte come meccanismi di Differential Privacy e test di membership-inference.
- Per maggior stabilità numerica è incluso un piccolo termine di regolarizzazione sulla diagonale della matrice di correlazione; se vuoi un'approssimazione minimale (nearest correlation matrix) considera l'implementazione di Higham (TODO).

Testing
-------
Esegui i test con:
```bash
pytest tests/
```

Contributi
----------
Pull request benvenute. Segui lo stile del repository, aggiungi test per nuove funzionalità e documenta i cambi di API nel README.

Licenza
-------
Aggiungi un file `LICENSE` (es. MIT) nella root del progetto se vuoi dichiarare esplicitamente la licenza del progetto.

