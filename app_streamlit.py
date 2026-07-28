"""
Minimal Streamlit app to let users upload their dataset, preview, pick numeric columns,
and download the processed CSV/parquet. Run with: streamlit run streamlit_upload_app.py
"""
import streamlit as st
import pandas as pd
from upload_dataset import load_dataset, select_columns, summarize_dataframe, save_dataset
import io

st.title("Upload and prepare dataset for synthetic generation")
uploaded = st.file_uploader("Upload CSV / Excel / Parquet / JSON", type=["csv","tsv","txt","xls","xlsx","parquet","json"])

if uploaded is not None:
    # streamlit gives a file-like object
    st.info("Reading file...")
    try:
        # Let helper detect type
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

    if st.button("Prepare selection"):
        df_sel, chosen_cols = select_columns(df, include=selected, numeric_only=True)
        st.write("Selection shape:", df_sel.shape)
        st.dataframe(df_sel.head(20))

        # Offer download
        fmt = st.selectbox("Download format", ["csv","parquet"])
        if st.button("Download prepared dataset"):
            buf = io.BytesIO()
            if fmt == "csv":
                buf2 = io.StringIO()
                df_sel.to_csv(buf2, index=False)
                b = buf2.getvalue().encode("utf-8")
                st.download_button("Download CSV", b, file_name="prepared.csv", mime="text/csv")
            else:
                df_sel.to_parquet(buf, index=False)
                st.download_button("Download Parquet", buf.getvalue(), file_name="prepared.parquet", mime="application/octet-stream")
