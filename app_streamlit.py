"""
Minimal Streamlit app to upload a dataset, choose numeric columns and run the Gaussian copula synthesis.
Run with: streamlit run app_streamlit.py
"""
import streamlit as st
import io
from synthdata.data_io import load_dataset, select_columns, summarize_dataframe
from synthdata.gaussian_copula import synthesize_and_evaluate
from synthdata.metrics import plot_corr_matrices

st.title("synthdata — upload and generate synthetic data (Gaussian copula)")

uploaded = st.file_uploader("Upload CSV / Excel / Parquet / JSON", type=["csv","tsv","txt","xls","xlsx","parquet","json"])
if uploaded is None:
    st.info("Upload a dataset to begin.")
else:
    try:
        df = load_dataset(uploaded, file_type=None)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.write("Preview:")
    st.dataframe(df.head(20))

    summary = summarize_dataframe(df)
    st.write("Shape:", summary["shape"])
    st.write("Missing fraction per column (first 10):")
    st.table(pd.Series(summary["missing_per_column"]).head(10).rename("missing_fraction"))

    numeric_cols = summary["numeric_columns"]
    st.write(f"Detected {len(numeric_cols)} numeric columns.")

    selected = st.multiselect("Select columns to keep for synthesis (numeric only recommended):", numeric_cols, default=numeric_cols[:6])

    if st.button("Generate synthetic data"):
        df_sel, chosen_cols = select_columns(df, include=selected, numeric_only=True)
        with st.spinner("Fitting Gaussian copula and sampling..."):
            synthetic, report, percol, univ = synthesize_and_evaluate(df_sel, n_samples=len(df_sel), random_state=42)
        st.success("Synthetic data generated")
        st.write("Report:")
        st.json(report)
        st.write("Top columns by KS (worst matches):")
        st.dataframe(univ.sort_values("ks", ascending=False).head(10))
        st.download_button("Download synthetic CSV", synthetic.to_csv(index=False).encode("utf-8"), file_name="synthetic.csv", mime="text/csv")

        # show correlation matrices (small inline plotting)
        st.pyplot(plot_corr_matrices(df_sel, synthetic, method="pearson", title_suffix=" (Streamlit)"))