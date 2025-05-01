# app.py

# requirements.txt should include:
# streamlit
# pandas
# numpy
# matplotlib
# polyfuzz
# chardet

import base64
import chardet
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF

# Blue Array Redirect Mapping Tool | Styled for Blue Array Branding | 1st May 2025

# ------------------------------------------
# Custom CSS Injection for Blue Array Branding
# ------------------------------------------
def inject_custom_css():
    st.markdown(
        """
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

        /* File uploader styling & custom placeholder */
        .stFileUploader > div {
            position: relative;
            border: 2px dashed #f48024;
            border-radius: 4px;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .stFileUploader > div > p {
            visibility: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .stFileUploader > div::before {
            content: "Add your CSV file here";
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #002f6c;
            font-size: 16px;
            pointer-events: none;
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

        /* Expander */
        .streamlit-expanderHeader {
            font-weight: bold;
            color: #f48024;
        }
        .streamlit-expanderContent {
            background-color: #fcfcfc;
        }

        /* Dataframe header */
        .stDataFrame thead tr th {
            background-color: #002f6c;
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

    # Main title only
    st.markdown(
        "<h1 style='color:#002f6c; text-align:center; margin-bottom:5px;'>Blue Array Redirect Mapping Tool</h1>",
        unsafe_allow_html=True
    )

    # Sidebar: instructions & model settings
    with st.sidebar:
        st.markdown("## How to Use")
        st.markdown(
            """
            - **Crawl both sets of URLs using Screaming Frog**  
              
              Crawl the URLs you want to redirect, then repeat for the candidate URLs to redirect to.

            - **Export Your Reports**  
              
              Export the **Internal HTML** report as a CSV for both crawls.  
              Save each export as a distinct file, e.g. `redirect_urls.csv` and `redirect_to_urls.csv`.

            - **Upload Your Files**  
              
              Under **Redirect data** upload your `redirect_urls.csv`.  
              Under **Redirect to data** upload your `redirect_to_urls.csv`.  
              _Tip: Filenames must differ; the tool will warn you if you accidentally upload the same file twice._

            - **Choose Your Matching Model**  
              
              **TF-IDF**  
              Good for keyword-heavy URL and title matching.  
              Tends to be faster on large datasets.  

              **Embeddings**  
              Uses Sentence-Transformers to capture semantic similarity.  
              Better for pages with similar content but low keyword overlap.

            - **Select Your Columns**  
              
              **Primary URL Column:** choose `Address` (or whichever column contains your canonical URL).  
              **Additional Columns (up to 2):** pick among `Title 1`, `Meta Description`, `H1-1`, etc., to refine matching.  
              These fields will be used to compute similarity scores.

            - **Adjust Confidence Threshold**  
              
              Use the slider (0–100%) to highlight any mappings below your chosen confidence level.  
              Default is 80%; lower if you want a broader match set, raise if you need stricter one-to-one mappings.

            - **Process & Review**  
              
              Click **Map redirects**  
              View the **Matches** table:  
              - **Source** = your Live URL  
              - **Match** = suggested Staging URL  
              - **Score** = similarity (0–1)

              Rows below your threshold will be shaded red for easy spotting of low confidence score
            """
        )
        st.markdown("---")
        st.markdown("## Matching Model")
        st.markdown("**TF-IDF: Comprehensive keyword overlap via TF-IDF.**")

    return "TF-IDF"

# ------------------------------------------
# File Upload & Validation
# ------------------------------------------
def create_file_uploader_widget(label):
    return st.file_uploader(label, type=['csv'])

def validate_uploaded(file1, file2):
    if not file1 or not file2 or file1.getvalue() == file2.getvalue():
        st.warning("🚨 Please upload two distinct, non-empty CSV files.")
        return False
    return True

# ------------------------------------------
# File Reading & Preprocessing
# ------------------------------------------
def read_file(file):
    enc = chardet.detect(file.getvalue())['encoding']
    return pd.read_csv(file, dtype="str", encoding=enc, on_bad_lines='skip')

def preprocess_df(df):
    return df.apply(lambda c: c.str.lower() if c.dtype == 'object' else c)

# ------------------------------------------
# Column Selection
# ------------------------------------------
def select_columns_for_matching(df_live, df_staging):
    common = list(set(df_live.columns) & set(df_staging.columns))
    st.markdown("### Select Columns for Matching")
    address_defaults = ['address','url','link']
    default_addr = next((c for c in common if c.lower() in address_defaults), common[0])
    addr = st.selectbox("Primary URL Column", common, index=common.index(default_addr))
    additional = [c for c in common if c != addr]
    selected = st.multiselect("Additional Columns (max 2)", additional, max_selections=2)
    return addr, selected

# ------------------------------------------
# Matching Logic (TF-IDF only)
# ------------------------------------------
def setup_matching_model(name):
    return PolyFuzz(TFIDF(min_similarity=0))

def match_data(df_live, df_staging, cols, model_name):
    model = setup_matching_model(model_name)
    matches = {}
    for col in cols:
        from_list = df_live[col].fillna('').astype(str).tolist()
        to_list = df_staging[col].fillna('').astype(str).tolist()
        model.match(from_list, to_list)
        matches[col] = model.get_matches()
    return matches

def find_best_matches(df_live, df_staging, matches, cols):
    rows = []
    for _, row in df_live.iterrows():
        best = {'Source': row[cols[0]], 'Match': None, 'Score': 0}
        for col in cols:
            mdf = matches[col]
            m = mdf[mdf['From'] == row[col]]
            if not m.empty and m.iloc[0]['Similarity'] > best['Score']:
                best.update({'Match': m.iloc[0]['To'], 'Score': m.iloc[0]['Similarity']})
        rows.append(best)
    return pd.DataFrame(rows)

# ------------------------------------------
# Visualization
# ------------------------------------------
def plot_score_histogram(df):
    df['ScorePct'] = df['Score'] * 100
    brackets = pd.cut(df['ScorePct'], bins=range(0,110,10), right=False)
    counts = brackets.value_counts().sort_index()
    plt.figure(figsize=(6,3))
    ax = counts.plot(kind='bar', width=0.8, edgecolor='#002f6c')
    ax.set_xlabel('Score Bracket (%)')
    ax.set_ylabel('Count')
    st.pyplot(plt)

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    model_name = setup_streamlit_interface()

    threshold_pct = st.slider("Confidence threshold (%)", 0, 100, 80)
    threshold = threshold_pct / 100.0

    col1, col2 = st.columns(2)
    with col1:
        live_file = create_file_uploader_widget("Redirect data")
    with col2:
        stag_file = create_file_uploader_widget("Redirect to data")

    if live_file and stag_file and validate_uploaded(live_file, stag_file):
        df_live = preprocess_df(read_file(live_file))
        df_stag = preprocess_df(read_file(stag_file))
        addr_col, add_cols = select_columns_for_matching(df_live, df_stag)
        cols = [addr_col] + add_cols

        if st.button("Map redirects"):
            matches = match_data(df_live, df_stag, cols, model_name)
            df_best = find_best_matches(df_live, df_stag, matches, cols)

            st.markdown(f"### Top Matches (highlighting < {threshold_pct}% confidence)")
            styled_df = df_best.style.apply(
                lambda row: ['background-color: #fde2e2' if row['Score'] < threshold else '' for _ in row],
                axis=1
            )
            st.dataframe(styled_df)

            plot_score_histogram(df_best)

            # Download CSV only
            csv_data = df_best.to_csv(index=False)
            b64_csv = base64.b64encode(csv_data.encode()).decode()
            st.markdown(
                f"<a href='data:text/csv;base64,{b64_csv}' download='mapping.csv'>💾 Download CSV (mapping.csv)</a>",
                unsafe_allow_html=True
            )

            st.balloons()

if __name__ == '__main__':
    main()





