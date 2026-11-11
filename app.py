"""Streamlit basket-completion demo trained exclusively on historical invoices.

Run: streamlit run app.py
Requires the UCI workbook at data/raw/Online Retail.xlsx.
"""
from pathlib import Path
import numpy as np
import streamlit as st
from src.data import load_transactions, clean_transactions, download_dataset, extract_dataset
from src.chronological_holdout import invoice_matrix
from src.recommendation_benchmark import fit_models, rank

st.set_page_config(page_title="UK Retail Basket Intelligence", layout="wide")
st.title("UK Retail Basket Intelligence")
st.caption("Historical UCI Online Retail data (2010–2011). Research demonstration, not a live shopping service.")
workbook = Path("data/raw/Online Retail.xlsx")
if not workbook.exists():
    st.info("Preparing the openly licensed UCI Online Retail dataset on first launch. This can take a few minutes.")
    try:
        with st.spinner("Downloading and extracting official UCI archive…"):
            workbook = extract_dataset(download_dataset())
    except Exception as exc:
        st.error(f"Dataset preparation failed: {exc}")
        st.stop()

@st.cache_resource(show_spinner="Preparing training-only recommendation models…")
def load_models(path, modified):
    df, summary = clean_transactions(load_transactions(Path(path)))
    invoices, codes, matrix = invoice_matrix(df)
    cutoff = int(len(invoices) * .7)
    freq, co, rules = fit_models(matrix[:cut].tocsr())
    descriptions = (df.dropna(subset=["Description"])
                    .groupby(["StockCode", "Description"]).size()
                    .reset_index(name="count")
                    .sort_values(["StockCode", "count"], ascending=[True, False])
                    .drop_duplicates("StockCode")
                    .set_index("StockCode")["Description"].to_dict())
    return codes, freq, co, rules, descriptions, cutoff, len(invoices)

codes, freq, co, rules, descriptions, train_n, all_n = load_models(str(workbook), workbook.stat().st_mtime)
code_to_idx = {code: i for i, code in enumerate(codes)}
method = st.selectbox("Recommendation method", ["apriori", "cooccurrence", "popularity"])
options = [code for code, count in zip(codes, freq) if count > 0]
chosen = st.multiselect("Products currently in the basket", options,
    format_func=lambda c: f"{c} — {descriptions.get(c, 'No description')}", max_selections=30)
k = st.slider("Number of suggestions", 1, 20, 10)
if not chosen:
    st.info("Choose one or more products to generate suggestions.")
else:
    indices = np.array([code_to_idx[c] for c in chosen], dtype=int)
    recommendations = rank(indices, freq, co, rules, method, k)
    st.dataframe([{"StockCode": codes[i], "Description": descriptions.get(codes[i], ""),
                   "Training invoice frequency": int(freq[i])} for i in recommendations],
                 use_container_width=True, hide_index=True)
st.caption(f"Models trained on earliest {train_n:,} of {all_n:,} UK invoices. Later invoices are excluded from training.")
st.info("Association is not causation. Recommendations are illustrative; this app does not predict individual purchases.")
