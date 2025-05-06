# app.py

# requirements.txt should include:
# streamlit
# pandas
# numpy
# polyfuzz
# chardet
# openai
# scikit-learn

import base64
import chardet
import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF

# ------------------------------------------
# Custom CSS Injection (Blue Array branding)
# ------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
      /* Hide default Streamlit elements */
      #MainMenu, footer, header { visibility: hidden; }

      /* App background and text */
      .stApp {
        background-color: #ffffff;
        color: #002f6c;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      }

      /* Buttons */
      .stButton > button {
        background-color: #002f6c;
        color: #ffffff;
        border-radius: 4px;
        font-weight: bold;
        padding: 10px 20px;
      }
      .stButton > button:hover {
        background-color: #01447E;
      }

      /* File uploader dropzone */
      div[data-testid="stFileUploadDropzone"] {
        position: relative;
        border: 2px dashed #f48024;
        border-radius: 4px;
        padding: 20px;
        background-color: #f9f9f9;
      }
      div[data-testid="stFileUploadDropzone"] > p {
        visibility: hidden !important;
        margin: 0;
        padding: 0;
      }
      div[data-testid="stFileUploadDropzone"]::before {
        content: "Add your CSV file here";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: #002f6c;
        font-size: 16px;
        pointer-events: none;
      }

      /* Styled expander for Select Columns */
      .streamlit-expanderHeader {
        font-weight: bold;
        color: #002f6c;
      }
      .streamlit-expanderContent {
        background-color: #f0f4f8;
        border-left: 4px solid #002f6c;
        padding: 12px;
      }

      /* Sidebar styling */
      .css-1v0mbdj.e1fqkh3o3 {
        font-size: 22px;
        color: #002f6c;
        font-weight: bold;
      }
      .sidebar .sidebar-content {
        background-color: #f0f4f8;
      }

      /* DataFrame header */
      .stDataFrame thead tr th {
        background-color: #002f6c;
        color: #ffffff;
      }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------
# Interface Setup
# ------------------------------------------
def setup_streamlit_interface():
    st.set_page_config(
        page_title="Blue Array Redirect Mapping Tool",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()

    # Main title
    st.markdown(
        "<h1 style='color:#002f6c; text-align:center; margin-bottom:10px;'>"
        "Blue Array Redirect Mapping Tool</h1>",
        unsafe_allow_html=True
    )

    # Sidebar instructions & model selection
    with st.sidebar:
        st.markdown("## How to Use")
        st.markdown("""
        1. **Crawl** both sets of URLs using Screaming Frog (live & target).  
        2. **Export** the *Internal HTML* reports as CSV (`redirect_urls.csv`, `redirect_to_urls.csv`).  
        3. **If you cannot crawl** (e.g., URLs not yet live or staging unavailable), prepare a CSV with your URLs in a column named **Address** and upload that instead.  
        4. **Upload** under **Redirect data** & **Redirect to data**.  
        5. **Choose** matching method below.  
        6. **Select** your columns, set confidence threshold, then **Map redirects**.  
        """)
        st.markdown("---")
        st.markdown("## Matching Method")

        method = st.selectbox(
            "Choose method",
            ["TF-IDF", "Embeddings"],
            help="TF-IDF: keyword overlap; Embeddings: semantic similarity via OpenAI"
        )

        # Show description and API key input conditionally
        if method == "TF-IDF":
            st.info(
                "**TF-IDF** is fast, keyword-based matching. "
                "Works best when URLs/titles share common terms."
            )
            api_key = None
        else:
            st.info(
                "**Embeddings** use OpenAI’s `text-embedding-ada-002` model "
                "to capture semantic similarity and synonyms."
            )
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Required for OpenAI embeddings"
            )

    return method, api_key

# ------------------------------------------
# File Upload & Validation
# ------------------------------------------
def create_file_uploader_widget(label):
    return st.file_uploader(label, type=['csv'])

def validate_uploaded(f1, f2):
    if not f1 or not f2 or f1.getvalue() == f2.getvalue():
        st.warning("🚨 Please upload two distinct, non-empty CSV files.")
        return False
    return True

# ------------------------------------------
# File Reading & Preprocessing
# ------------------------------------------
def read_file(f):
    enc = chardet.detect(f.getvalue())['encoding']
    return pd.read_csv(f, dtype="str", encoding=enc, on_bad_lines='skip')

def preprocess_df(df):
    return df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)

# ------------------------------------------
# Column Selection (alphabetical expander)
# ------------------------------------------
def select_columns_for_matching(df_live, df_staging):
    common = sorted(set(df_live.columns) & set(df_staging.columns))
    defaults = ['address', 'url', 'link']
    default_addr = next((c for c in common if c.lower() in defaults), common[0])

    with st.expander("Select Columns for Matching", expanded=True):
        st.markdown("Use the search box to filter columns quickly.")
        addr = st.selectbox(
            "Primary URL Column",
            common,
            index=common.index(default_addr),
            help="Start typing to search"
        )
        additional = [c for c in common if c != addr]
        selected = st.multiselect(
            "Additional Columns (max 2)",
            additional,
            max_selections=2,
            help="Start typing to search"
        )
    return addr, selected

# ------------------------------------------
# TF-IDF Matching
# ------------------------------------------
def match_tfidf(df_live, df_staging, cols):
    model = PolyFuzz(TFIDF(min_similarity=0))
    matches = {}
    for col in cols:
        fl = df_live[col].fillna('').astype(str).tolist()
        tl = df_staging[col].fillna('').astype(str).tolist()
        model.match(fl, tl)
        matches[col] = model.get_matches()
    rows = []
    for _, r in df_live.iterrows():
        best = {'Source': r[cols[0]], 'Match': None, 'Score': 0}
        for col in cols:
            mdf = matches[col]
            m = mdf[mdf['From'] == r[col]]
            if not m.empty and m.iloc[0]['Similarity'] > best['Score']:
                best.update({
                    'Match': m.iloc[0]['To'],
                    'Score': m.iloc[0]['Similarity']
                })
        rows.append(best)
    return pd.DataFrame(rows)

# ------------------------------------------
# OpenAI Embeddings Matching
# ------------------------------------------
def match_openai(df_live, df_staging, cols, api_key):
    openai.api_key = api_key

    # combine columns into single text per row
    live_texts = df_live[cols].fillna('').agg(' '.join, axis=1).tolist()
    stag_texts = df_staging[cols].fillna('').agg(' '.join, axis=1).tolist()

    # request embeddings
    resp_live = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=live_texts
    )
    live_emb = [d.embedding for d in resp_live.data]

    resp_stag = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=stag_texts
    )
    stag_emb = [d.embedding for d in resp_stag.data]

    # compute best matches
    rows = []
    for idx, vec in enumerate(live_emb):
        sims = cosine_similarity([vec], stag_emb)[0]
        best_i = int(np.argmax(sims))
        score  = float(sims[best_i])
        rows.append({
            'Source': df_live.iloc[idx][cols[0]],
            'Match':  df_staging.iloc[best_i][cols[0]] if score > 0 else None,
            'Score':  score
        })
    return pd.DataFrame(rows)

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    method, api_key = setup_streamlit_interface()

    threshold_pct = st.slider("Confidence threshold (%)", 0, 100, 90)
    threshold     = threshold_pct / 100.0
    include_low   = st.checkbox("Include URLs below threshold", True)

    col1, col2 = st.columns(2)
    with col1:
        live_file = create_file_uploader_widget("Redirect data")
    with col2:
        stag_file = create_file_uploader_widget("Redirect to data")

    if live_file and stag_file and validate_uploaded(live_file, stag_file):
        df_live    = preprocess_df(read_file(live_file))
        df_staging = preprocess_df(read_file(stag_file))
        addr_col, add_cols = select_columns_for_matching(df_live, df_staging)
        cols = [addr_col] + add_cols

        if st.button("Map redirects"):
            if method == "Embeddings":
                if not api_key:
                    st.error("Enter your OpenAI API key to use Embeddings.")
                    return
                df_best = match_openai(df_live, df_staging, cols, api_key)
            else:
                df_best = match_tfidf(df_live, df_staging, cols)

            # Mapping Quality Summary
            avg_score = df_best['Score'].mean()
            med_score = df_best['Score'].median()
            pct_above = (df_best['Score'] >= threshold).mean() * 100

            st.markdown("## Mapping Quality Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Average Confidence", f"{avg_score:.2%}")
            m2.metric("Median Confidence",  f"{med_score:.2%}")
            m3.metric(f"% ≥ {threshold_pct}%", f"{pct_above:.1f}%")

            # Filter based on threshold
            df_show = df_best if include_low else df_best[df_best['Score'] >= threshold]

            # Top Matches table
            st.markdown(f"### Top Matches ( < {threshold_pct}% highlighted )")
            styled = df_show.style.apply(
                lambda r: ['background-color:#fde2e2' if r['Score'] < threshold else '' for _ in r],
                axis=1
            )
            st.dataframe(styled)

            # Download CSV
            csv = df_show.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(
                f"<a href='data:text/csv;base64,{b64}' download='mapping.csv'>"
                "💾 Download CSV</a>",
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()















