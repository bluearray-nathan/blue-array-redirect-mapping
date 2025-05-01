# app.py

# requirements.txt should include:
# streamlit
# pandas
# numpy
# polyfuzz
# chardet
# sentence-transformers
# scikit-learn

import base64
import chardet
import streamlit as st
import pandas as pd
import numpy as np
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF

# ------------------------------------------
# Custom CSS Injection (Blue Array branding)
# ------------------------------------------
def inject_custom_css():
    st.markdown("""
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .stApp {background-color:#fff; color:#002f6c; font-family:'Helvetica Neue', Helvetica, Arial,sans-serif;}
      .stButton>button {background-color:#002f6c; color:#fff; border-radius:4px; font-weight:bold; padding:10px 20px;}
      .stButton>button:hover {background-color:#01447E;}
      div[data-testid="stFileUploadDropzone"] {
        position:relative; border:2px dashed #f48024; border-radius:4px;
        padding:20px; background-color:#f9f9f9;
      }
      div[data-testid="stFileUploadDropzone"]>p {visibility:hidden !important; margin:0; padding:0;}
      div[data-testid="stFileUploadDropzone"]::before {
        content:"Add your CSV file here"; position:absolute;
        top:50%; left:50%; transform:translate(-50%,-50%);
        color:#002f6c; font-size:16px; pointer-events:none;
      }
      .streamlit-expanderHeader {font-weight:bold; color:#002f6c;}
      .streamlit-expanderContent {background-color:#f0f4f8; border-left:4px solid #002f6c; padding:12px;}
      .css-1v0mbdj.e1fqkh3o3 {font-size:22px; color:#002f6c; font-weight:bold;}
      .sidebar .sidebar-content {background-color:#f0f4f8;}
      .stDataFrame thead tr th {background-color:#002f6c; color:#fff;}
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------
# Sidebar & Page Setup
# ------------------------------------------
def setup_streamlit_interface():
    st.set_page_config(page_title="Blue Array Redirect Mapping Tool",
                       layout="wide", initial_sidebar_state="expanded")
    inject_custom_css()

    st.markdown(
      "<h1 style='color:#002f6c; text-align:center;'>Blue Array Redirect Mapping Tool</h1>",
      unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown("## How to Use")
        st.markdown("""
        - **Crawl both sets of URLs** with Screaming Frog  
        - **Export Internal HTML** as CSV (`redirect_urls.csv`, `redirect_to_urls.csv`)  
        - **Upload** under **Redirect data** & **Redirect to data**  
        - **Choose model**: TF-IDF or Embeddings  
        - **Select columns**, **set threshold**, then **Map redirects**  
        """)
        st.markdown("---")
        return st.selectbox(
            "Choose model",
            ["TF-IDF", "Embeddings"],
            help="TF-IDF = keyword overlap; Embeddings = semantic"
        )

# ------------------------------------------
# File Upload Helpers
# ------------------------------------------
def create_file_uploader_widget(label):
    return st.file_uploader(label, type=['csv'])

def validate_uploaded(f1, f2):
    if (not f1 or not f2) or (f1.getvalue() == f2.getvalue()):
        st.warning("🚨 Please upload two distinct, non-empty CSVs.")
        return False
    return True

def read_file(f):
    enc = chardet.detect(f.getvalue())['encoding']
    return pd.read_csv(f, dtype="str", encoding=enc, on_bad_lines='skip')

def preprocess_df(df):
    return df.apply(lambda col: col.str.lower() if col.dtype=='object' else col)

# ------------------------------------------
# Column Selection (alphabetical + styled expander)
# ------------------------------------------
def select_columns_for_matching(df_live, df_staging):
    common = sorted(set(df_live.columns) & set(df_staging.columns))
    defaults = ['address','url','link']
    default_addr = next((c for c in common if c.lower() in defaults), common[0])

    with st.expander("Select Columns for Matching", True):
        st.markdown("Use the search box to filter columns.")
        addr = st.selectbox("Primary URL Column", common,
                            index=common.index(default_addr),
                            help="Search columns…")
        additional = [c for c in common if c != addr]
        selected = st.multiselect("Additional Columns (max 2)",
                                  additional, max_selections=2,
                                  help="Search columns…")
    return addr, selected

# ------------------------------------------
# TF-IDF Matcher (unchanged)
# ------------------------------------------
def match_tfidf(df_live, df_staging, cols):
    model = PolyFuzz(TFIDF(min_similarity=0))
    matches = {}
    for col in cols:
        fr = df_live[col].fillna('').astype(str).tolist()
        to = df_staging[col].fillna('').astype(str).tolist()
        model.match(fr, to)
        matches[col] = model.get_matches()

    # pick best per row
    rows = []
    for _, r in df_live.iterrows():
        best = {'Source': r[cols[0]], 'Match': None, 'Score': 0}
        for col in cols:
            mdf = matches[col]
            m = mdf[mdf['From']==r[col]]
            if not m.empty and m.iloc[0]['Similarity']>best['Score']:
                best.update({'Match': m.iloc[0]['To'],
                             'Score': m.iloc[0]['Similarity']})
        rows.append(best)
    return pd.DataFrame(rows)

# ------------------------------------------
# Embeddings Matcher (dynamic import)
# ------------------------------------------
def match_embeddings(df_live, df_staging, cols):
    # only now pull in Torch & SentenceTransformer
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    # combine selected cols into one text per row
    live_texts = df_live[cols].fillna('').agg(' '.join, axis=1).tolist()
    stag_texts = df_staging[cols].fillna('').agg(' '.join, axis=1).tolist()

    model = SentenceTransformer('all-mpnet-base-v2')
    live_emb = model.encode(live_texts, show_progress_bar=False)
    stag_emb = model.encode(stag_texts, show_progress_bar=False)

    rows = []
    for idx, vec in enumerate(live_emb):
        sims = cosine_similarity([vec], stag_emb)[0]
        best_i = int(np.argmax(sims))
        score  = float(sims[best_i])
        rows.append({
            'Source': df_live.iloc[idx][cols[0]],
            'Match':  df_staging.iloc[best_i][cols[0]] if score>0 else None,
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

    c1, c2 = st.columns(2)
    with c1:
        f_live = create_file_uploader_widget("Redirect data")
    with c2:
        f_stag = create_file_uploader_widget("Redirect to data")

    if f_live and f_stag and validate_uploaded(f_live, f_stag):
        df_live    = preprocess_df(read_file(f_live))
        df_staging = preprocess_df(read_file(f_stag))
        addr, adds = select_columns_for_matching(df_live, df_staging)
        cols = [addr] + adds

        if st.button("Map redirects"):
            if model_name == "Embeddings":
                df_best = match_embeddings(df_live, df_staging, cols)
            else:
                df_best = match_tfidf(df_live, df_staging, cols)

            # Quality summary
            avg    = df_best['Score'].mean()
            med    = df_best['Score'].median()
            pct    = (df_best['Score']>=threshold).mean()*100
            st.markdown("## Mapping Quality Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Confidence", f"{avg:.2%}")
            m2.metric("Median Confidence", f"{med:.2%}")
            m3.metric(f"% ≥ {threshold_pct}%", f"{pct:.1f}%")

            # filter
            df_show = df_best if include_low else df_best[df_best['Score']>=threshold]

            # results table
            st.markdown(f"### Top Matches ( < {threshold_pct}% highlighted )")
            styled = df_show.style.apply(
                lambda r: ['background-color:#fde2e2' if r['Score']<threshold else '' for _ in r],
                axis=1
            )
            st.dataframe(styled)

            # download CSV
            csv = df_show.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(
              f"<a href='data:text/csv;base64,{b64}' download='mapping.csv'>💾 Download CSV</a>",
              unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()











