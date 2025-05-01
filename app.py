# app.py

# requirements.txt should now be:
# streamlit
# pandas
# numpy
# matplotlib
# plotly
# polyfuzz
# chardet
# xlsxwriter

import base64
import chardet
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from polyfuzz import PolyFuzz
from polyfuzz.models import TFIDF
import xlsxwriter

# Blue Array Migration Mapper | Styled for Blue Array Branding | 1st May 2025

# ------------------------------------------
# Custom CSS Injection for Blue Array Branding
# ------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Hide default Streamlit elements */
        #MainMenu, footer, header {visibility: hidden;}

        /* Page background and text */
        .stApp {
            background-color: #ffffff;
            color: #002f6c;
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        }

        /* Buttons */
        .stButton>button {
            background-color: #002f6c;
            color: #ffffff;
            border-radius: 4px;
            font-weight: bold;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #01447E;
        }

        /* File uploader */
        .stFileUploader>div {
            border: 2px dashed #f48024;
            border-radius: 4px;
            padding: 20px;
            background-color: #f9f9f9;
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
        """, unsafe_allow_html=True
    )

# ------------------------------------------
# Interface Setup
# ------------------------------------------
def setup_streamlit_interface():
    st.set_page_config(
        page_title="Blue Array Migration Mapper",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()

    # Main title
    st.markdown(
        "<h1 style='color:#002f6c; text-align:center; margin-bottom:5px;'>Blue Array Website Migration Mapper</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h4 style='color:#f48024; text-align:center; margin-top:0;'>Effortlessly map your live & staging URLs</h4>",
        unsafe_allow_html=True
    )

    # Sidebar: instructions & model settings
    with st.sidebar:
        st.markdown("## How to Use")
        st.markdown(
            """
            1. Crawl Live & Staging with Screaming Frog.  
            2. Export as CSV/XLSX.  
            3. Upload files below.  
            4. Use TF-IDF matching (only supported model).  
            5. Adjust confidence threshold slider if desired.  
            6. Click Map redirects and download the report.
            """
        )
        st.markdown("---")
        st.markdown("## Matching Model")
        st.markdown("**TF-IDF: Comprehensive keyword overlap via TF-IDF.**")

    return "TF-IDF"

# ------------------------------------------
# File Upload & Validation
# ------------------------------------------
def create_file_uploader_widget(label, types):
    return st.file_uploader(label, type=types)

def validate_uploaded(file1, file2):
    if not file1 or not file2 or file1.getvalue() == file2.getvalue():
        st.warning("🚨 Please upload two distinct, non-empty files.")
        return False
    return True

# ------------------------------------------
# File Reading & Preprocessing
# ------------------------------------------
def read_file(file):
    if file.name.lower().endswith('.csv'):
        enc = chardet.detect(file.getvalue())['encoding']
        return pd.read_csv(file, dtype="str", encoding=enc, on_bad_lines='skip')
    return pd.read_excel(file, dtype="str")

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
    # name is always "TF-IDF" in this version
    return PolyFuzz(TFIDF(min_similarity=0))

def match_data(df_live, df_staging, cols, model_name):
    model = setup_matching_model(model_name)
    matches = {}
    for col in cols:
        # Convert pandas Series to Python lists of strings
        from_list = df_live[col].fillna('').astype(str).tolist()
        to_list   = df_staging[col].fillna('').astype(str).tolist()

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
                best.update({
                    'Match': m.iloc[0]['To'],
                    'Score': m.iloc[0]['Similarity']
                })
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
# Excel Export
# ------------------------------------------
def create_excel(df, filename='mapping.xlsx'):
    score_df = pd.DataFrame({
        'Bracket': pd.cut(df['Score']*100, bins=range(0,110,10), right=False)
                         .value_counts().sort_index().index.astype(str),
        'Count': pd.cut(df['Score']*100, bins=range(0,110,10), right=False)
                       .value_counts().sort_index().values
    })
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Matches', index=False)
    score_df.to_excel(writer, sheet_name='Distribution', index=False)
    wb = writer.book
    ws = writer.sheets['Matches']
    fmt = wb.add_format({'num_format':'0.00%', 'align':'center'})
    ws.set_column('C:C', 12, fmt)
    ws2 = writer.sheets['Distribution']
    chart = wb.add_chart({'type':'column'})
    chart.add_series({
        'categories': f"=Distribution!$A$2:$A${len(score_df)+1}",
        'values': f"=Distribution!$B$2:$B${len(score_df)+1}",
        'fill': {'color': '#f48024'}
    })
    chart.set_title({'name':'Score Distribution'})
    ws2.insert_chart('D2', chart)
    writer.save()
    return filename

# ------------------------------------------
# Main
# ------------------------------------------
def main():
    model_name = setup_streamlit_interface()

    # Confidence threshold slider
    threshold_pct = st.slider("Confidence threshold (%)", 0, 100, 80)
    threshold = threshold_pct / 100.0

    col1, col2 = st.columns(2)
    with col1:
        live_file = create_file_uploader_widget("Redirect data", ['csv','xlsx','xls'])
    with col2:
        stag_file = create_file_uploader_widget("Redirect to data", ['csv','xlsx','xls'])

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

            filename = create_excel(df_best)
            with open(filename, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f"<a href='data:application/octet-stream;base64,{b64}' download='{filename}'>💾 Download {filename}</a>",
                unsafe_allow_html=True
            )
            st.balloons()

if __name__ == '__main__':
    main()



