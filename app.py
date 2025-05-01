# app.py

# requirements.txt should include:
# streamlit
# pandas
# numpy
# matplotlib
# polyfuzz
# chardet
# sentence-transformers
# scikit-learn

import base64
import chardet
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Blue Array Redirect Mapping Tool | Styled for Blue Array Branding | 1st May 2025

# ------------------------------------------
# Custom CSS Injection for Blue Array Branding
# ------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
      /* Hide default Streamlit elements */
      #MainMenu, footer, header { visibility: hidden; }

      /* Page background and text */
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

      /* File uploader dropzone before file is added */
      div[data-testid="stFileUploadDropzone"] {
        position: relative;
        border: 2px dashed #f48024;
        border-radius: 4px;
        padding: 20px;
        background-color: #f9f9f9;
      }
      /* Hide default placeholder text */
      div[data-testid="stFileUploadDropzone"] > p {
        visibility: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
      }
      /* Inject custom placeholder */
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

    st.markdown(
        "<h1 style='color:#002f6c; text-align:center; margin-bottom:5px;'>"
        "Blue Array Redirect Mapping Tool</h1>",
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("## How to Use")
        st.markdown("""
        - **Crawl both sets of URLs using Screaming Frog**  
          Crawl the URLs you want to redirect, then repeat for the candidate URLs to redirect to.

        - **Export Your Reports**  
          Export the **Internal HTML** report as a CSV for both crawls.  
          Save each export as distinct files, e.g. `redirect_urls.csv` and `redirect_to_urls.csv`.

        - **Upload Your Files**  
          Under **Redirect data** upload `redirect_urls.csv`.  
          Under **Redirect to data** upload `redirect_to_urls.csv`.  
          _Tip: Filenames must differ; the tool will warn you if you upload the same file twice._

        - **Choose Your Matching Model**  
          Use **TF-IDF** for keyword overlap & speed, or **Embeddings** for semantic matches.

        - **Select Your Columns**  
          Expand the “Select Columns for Matching” panel to pick your fields.

        - **Adjust Confidence Threshold**  
          Use the slider (0–100%) to highlight mappings below your cutoff. Default 80%.

        - **Process & Review**  
          Click **Map redirects**, review the table, and download your CSV.
        """)
        st.markdown("---")
        st.markdown("## Matching Model")
        return st.selectbox(
            "Choose model", 
            ["TF-IDF", "Embeddings"],
            help="TF-IDF = keyword-based; Embeddings = semantic"
        )

# ------------------------------------------
# File Upload & Validation
# ------------------------------------------
def create_file_uploader_widget(label):
    return st.file_uploader(label, type=['csv'])

def validate_uploaded(f1, f2):
    if (not f1 or not f2) or (f1.getvalue() == f2.getvalue()):
        st.warning("🚨 Upload two distinct, non-empty CSV files.")
        return False
    return True

# ------------------------------------------
# File Reading & Preprocessing
# ------------------------------------------
def read_file(file):
    enc = chardet.detect(file.getvalue())['encoding']
    return pd.read_csv(file, dtype="str", encoding=enc, on_bad_lines='skip')

def preprocess_df(df):
    return df.apply(lambda col: col.str.lower() 
                    if col.dtype == 'object' else col)

# ------------------------------------------
# Column Selection (alphabetical, styled expander)
# ------------------------------------------
def select_columns_for_matching(df_live, df_staging):
    common = sorted(set(df_live.columns) & set(df_staging.columns))
    address_defaults = ['address','url','link']
    default_addr = next((c for c in common 
                         if c.lower() in address_defaults),
                        common[0])

    with st.expander("Select Columns for Matching", expanded=True):
        st.markdown("Use the search box to filter columns.", unsafe_allow_html=True)
        addr = st.selectbox(
            "Primary URL Column", 
            common, 
            index=common.index(default_addr),
            help="Search columns..."
        )
        additional = [c for c in common if c != addr]
        selected = st.multiselect(
            "Additional Columns (max 2)",
            additional,
            max_selections=2,
            help="Search columns..."
        )
    return addr, selected

# ------------------------------------------
# Matching Logic
# ------------------------------------------
def match_tfidf(df_live, df_staging, cols):
    model = PolyFuzz(TFIDF(min_similarity=0))
    matches = {}
    for col in cols:
        fl = df_live[col].fillna('').astype(str).tolist()
        tl = df_staging[col].fillna('').astype(str).tolist()
        model.match(fl, tl)
        matches[col] = model.get_matches()
    # find best per row
    rows = []
    for _, row in df_live.iterrows():
        best = {'Source': row[cols[0]], 'Match': None, 'Score': 0}
        for col in cols:
            mdf = matches[col]
            m = mdf[mdf['From'] == row[col]]
            if not m.empty and m.iloc[0]['Similarity'] > best['Score']:
                best.update({
                    'Match': m.iloc[0]['To'],
                    'Score': m.iloc[0]['Similarity']
                })
        rows.append(best)
    return pd.DataFrame(rows)

def match_embeddings(df_live, df_staging, cols):
    # combine selected cols into single text
    live_texts = df_live[cols].fillna('').agg(' '.join, axis=1).tolist()
    stag_texts = df_staging[cols].fillna('').agg(' '.join, axis=1).tolist()
    model = SentenceTransformer('all-mpnet-base-v2')
    live_emb = model.encode(live_texts, show_progress_bar=False)
    stag_emb = model.encode(stag_texts, show_progress_bar=False)
    # match best by cosine
    rows = []
    for idx, vec in enumerate(live_emb):
        sims = cosine_similarity([vec], stag_emb)[0]
        best_i = int(np.argmax(sims))
        score  = float(sims[best_i])
        match_url = (df_staging.iloc[best_i][cols[0]]
                     if score>0 else None)
        rows.append({
            'Source': df_live.iloc[idx][cols[0]],
            'Match':  match_url,
            'Score':  score
        })
    return pd.DataFrame(rows)

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    model_name = setup_streamlit_interface()

    threshold_pct = st.slider("Confidence threshold (%)", 0, 100, 80)
    threshold     = threshold_pct / 100.0
    include_low   = st.checkbox("Include URLs below threshold", True)

    col1, col2 = st.columns(2)
    with col1:
        live_file = create_file_uploader_widget("Redirect data")
    with col2:
        stag_file = create_file_uploader_widget("Redirect to data")

    if live_file and stag_file and validate_uploaded(live_file, stag_file):
        df_live   = preprocess_df(read_file(live_file))
        df_staging= preprocess_df(read_file(stag_file))
        addr_col, add_cols = select_columns_for_matching(df_live, df_staging)
        cols = [addr_col] + add_cols

        if st.button("Map redirects"):
            if model_name == 'TF-IDF':
                df_best = match_tfidf(df_live, df_staging, cols)
            else:
                df_best = match_embeddings(df_live, df_staging, cols)

            # Quality summary
            avg    = df_best['Score'].mean()
            median = df_best['Score'].median()
            pct    = (df_best['Score'] >= threshold).mean()*100
            st.markdown("## Mapping Quality Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Confidence", f"{avg:.2%}")
            c2.metric("Median Confidence", f"{median:.2%}")
            c3.metric(f"% ≥ {threshold_pct}%", f"{pct:.1f}%")

            # filter
            display_df = df_best if include_low else df_best[df_best['Score']>=threshold]

            st.markdown(f"### Top Matches ( < {threshold_pct}% highlighted )")
            styled = display_df.style.apply(
                lambda row: ['background-color: #fde2e2'
                             if row['Score']<threshold else ''
                             for _ in row],
                axis=1
            )
            st.dataframe(styled)

            # download CSV
            csv = display_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(
              f"<a href='data:text/csv;base64,{b64}' download='mapping.csv'>"
              "💾 Download CSV</a>", unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()










