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
      #MainMenu, footer, header { visibility: hidden; }
      .stApp { background-color: #ffffff; color: #002f6c; font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }
      .stButton > button { background-color: #002f6c; color: #ffffff; border-radius: 4px; font-weight: bold; padding: 10px 20px; }
      .stButton > button:hover { background-color: #01447E; }
      div[data-testid="stFileUploadDropzone"] {
        position: relative; border: 2px dashed #f48024; border-radius: 4px; padding: 20px; background-color: #f9f9f9;
      }
      div[data-testid="stFileUploadDropzone"] > p { visibility: hidden !important; margin: 0; padding: 0; }
      div[data-testid="stFileUploadDropzone"]::before {
        content: "Add your CSV file here"; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        color: #002f6c; font-size: 16px; pointer-events: none;
      }
      .streamlit-expanderHeader { font-weight: bold; color: #002f6c; }
      .streamlit-expanderContent { background-color: #f0f4f8; border-left: 4px solid #002f6c; padding: 12px; }
      .css-1v0mbdj.e1fqkh3o3 { font-size: 22px; color: #002f6c; font-weight: bold; }
      .sidebar .sidebar-content { background-color: #f0f4f8; }
      .stDataFrame thead tr th { background-color: #002f6c; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------
# Interface Setup
# ------------------------------------------
def setup_streamlit_interface():
    st.set_page_config(page_title="Blue Array Redirect Mapping Tool",
                       layout="wide", initial_sidebar_state="expanded")
    inject_custom_css()
    st.markdown(
        "<h1 style='color:#002f6c; text-align:center; margin-bottom:10px;'>"
        "Blue Array Redirect Mapping Tool</h1>",
        unsafe_allow_html=True
    )
    with st.sidebar:
        st.markdown("## How to Use")
        st.markdown("""
        1. **Crawl** both sets of URLs using Screaming Frog (live & target).  
        2. **Export** the *Internal HTML* reports as CSV (`redirect_urls.csv`, `redirect_to_urls.csv`).  
        3. **If you cannot crawl** (e.g., URLs not yet live or staging unavailable), prepare a CSV with your URLs in a column named **Address**.  
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
        if method == "TF-IDF":
            st.info("**TF-IDF** is fast, keyword-based matching. Works best when URLs/titles share common terms.")
            api_key = None
        else:
            st.info("**Embeddings** use OpenAI’s `text-embedding-ada-002` model to capture semantic similarity.")
            api_key = st.text_input("OpenAI API Key", type="password",
                                     help="Required for OpenAI embeddings")
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
# Column Selection
# ------------------------------------------
def select_columns_for_matching(df_live, df_staging):
    common = sorted(set(df_live.columns) & set(df_staging.columns))
    defaults = ['address', 'url', 'link']
    default_addr = next((c for c in common if c.lower() in defaults), common[0])
    with st.expander("Select Columns for Matching", expanded=True):
        st.markdown("Use the search box to filter columns quickly.")
        addr = st.selectbox("Primary URL Column", common,
                            index=common.index(default_addr), help="Start typing to search")
        additional = [c for c in common if c != addr]
        selected = st.multiselect("Additional Columns (max 2)", additional,
                                  max_selections=2, help="Start typing to search")
    return addr, selected

# ------------------------------------------
# Matching Functions
# ------------------------------------------
def match_tfidf(df_live, df_staging, cols):
    model = PolyFuzz(TFIDF(min_similarity=0))
    matches = {}
    for col in cols:
        fl = df_live[col].fillna('').tolist()
        tl = df_staging[col].fillna('').tolist()
        model.match(fl, tl)
        matches[col] = model.get_matches()
    primary = cols[0]
    rows = []
    for _, row in df_live.iterrows():
        best_score, best_match = 0, None
        for col in cols:
            mdf = matches[col]
            m = mdf[mdf['From'] == row[col]]
            if not m.empty and m.iloc[0]['Similarity'] > best_score:
                best_score = m.iloc[0]['Similarity']
                to_val = m.iloc[0]['To']
                if col == primary:
                    best_match = to_val
                else:
                    matched = df_staging[df_staging[col] == to_val]
                    if not matched.empty:
                        best_match = matched.iloc[0][primary]
        rows.append({'Source': row[primary], 'Match': best_match, 'Score': best_score})
    return pd.DataFrame(rows)

def match_openai(df_live, df_staging, cols, api_key):
    openai.api_key = api_key
    live_texts = df_live[cols].fillna('').agg(' '.join, axis=1).tolist()
    stag_texts = df_staging[cols].fillna('').agg(' '.join, axis=1).tolist()
    resp_live = openai.embeddings.create(model="text-embedding-ada-002", input=live_texts)
    live_emb = [d.embedding for d in resp_live.data]
    resp_stag = openai.embeddings.create(model="text-embedding-ada-002", input=stag_texts)
    stag_emb = [d.embedding for d in resp_stag.data]
    primary = cols[0]
    rows = []
    for i, vec in enumerate(live_emb):
        sims = cosine_similarity([vec], stag_emb)[0]
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        match_addr = df_staging.iloc[idx][primary] if score > 0 else None
        rows.append({'Source': df_live.iloc[i][primary], 'Match': match_addr, 'Score': score})
    return pd.DataFrame(rows)

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    method, api_key = setup_streamlit_interface()

    threshold_pct = st.slider("Confidence threshold (%)", 0, 100, 90)
    threshold = threshold_pct / 100.0
    include_low = st.checkbox("Include URLs below threshold", True)

    # Show appropriate guide
    if method == "Embeddings":
        st.markdown("#### Embeddings Threshold Guide")
        st.markdown("""
        - **1.00**: No change in meaning  
        - **0.95–0.99**: Minor update, content is still aligned  
        - **0.85–0.94**: Moderate shift, re-evaluation likely by Google  
        - **≤ 0.84**: Major drift, Google may treat it as new  
        """)
    else:
        st.markdown("#### TF-IDF Threshold Guide")
        st.markdown("""
        - **1.00**: No change in meaning  
        - **0.70–0.99**: Minor update, content is still aligned  
        - **0.50–0.69**: Moderate shift, re-evaluation likely by Google  
        - **< 0.50**: Major drift, Google may treat it as new  
        """)
    st.markdown("*Treat this as a guide; circumstances will vary. The more data you use for matching, the more accurate it will be.*")

    c1, c2 = st.columns(2)
    with c1:
        f_live = create_file_uploader_widget("Redirect data")
    with c2:
        f_stag = create_file_uploader_widget("Redirect to data")

    if f_live and f_stag and validate_uploaded(f_live, f_stag):
        df_live = preprocess_df(read_file(f_live))
        df_staging = preprocess_df(read_file(f_stag))
        addr, adds = select_columns_for_matching(df_live, df_staging)
        cols = [addr] + adds

        if st.button("Map redirects"):
            if method == "Embeddings":
                if not api_key:
                    st.error("Enter your OpenAI API key.")
                    return
                df_best = match_openai(df_live, df_staging, cols, api_key)
            else:
                df_best = match_tfidf(df_live, df_staging, cols)

            # Summary metrics
            avg = df_best['Score'].mean()
            med = df_best['Score'].median()
            pct = (df_best['Score'] >= threshold).mean() * 100
            st.markdown("## Mapping Quality Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Confidence", f"{avg:.2%}")
            m2.metric("Median Confidence", f"{med:.2%}")
            m3.metric(f"% ≥ {threshold_pct}%", f"{pct:.1f}%")

            # Interpretation counts
            total = len(df_best)
            if method == "Embeddings":
                cnt1 = (df_best['Score'] == 1.0).sum()
                cnt2 = ((df_best['Score'] >= 0.95) & (df_best['Score'] < 1.0)).sum()
                cnt3 = ((df_best['Score'] >= 0.85) & (df_best['Score'] < 0.95)).sum()
                cnt4 = (df_best['Score'] <= 0.84).sum()
                labels = ["1.00 (No change)", "0.95–0.99 (Minor)", "0.85–0.94 (Moderate)", "≤0.84 (Major)"]
                counts = [cnt1, cnt2, cnt3, cnt4]
            else:
                cnt1 = (df_best['Score'] == 1.0).sum()
                cnt2 = ((df_best['Score'] >= 0.70) & (df_best['Score'] < 1.0)).sum()
                cnt3 = ((df_best['Score'] >= 0.50) & (df_best['Score'] < 0.70)).sum()
                cnt4 = (df_best['Score'] < 0.50).sum()
                labels = ["1.00 (No change)", "0.70–0.99 (Minor)", "0.50–0.69 (Moderate)", "<0.50 (Major)"]
                counts = [cnt1, cnt2, cnt3, cnt4]

            st.markdown("## Score Interpretation")
            for label, count in zip(labels, counts):
                st.write(f"- **{label}**: {count} / {total}")

            # Filter & sort
            df_show = df_best if include_low else df_best[df_best['Score'] >= threshold]
            df_show = df_show.sort_values('Score', ascending=False).reset_index(drop=True)

            # Results
            st.markdown(f"### Top Matches (highest confidence first, < {threshold_pct}% shaded)")
            styled = df_show.style.apply(
                lambda r: ['background-color:#fde2e2' if r['Score'] < threshold else '' for _ in r],
                axis=1
            )
            st.dataframe(styled)

            # CSV download
            csv = df_show.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.markdown(f"<a href='data:text/csv;base64,{b64}' download='mapping.csv'>💾 Download CSV</a>",
                        unsafe_allow_html=True)

if __name__ == "__main__":
    main()

















