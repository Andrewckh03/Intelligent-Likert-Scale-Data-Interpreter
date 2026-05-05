# app.py (Final Polish: Soft Card Theme, Modern Aesthetics, Friendly Jargon)

import streamlit as st
import pandas as pd
import numpy as np
import re
import tempfile
import os
from graphviz import Digraph
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from fpdf import FPDF
import google.generativeai as genai
import nltk
from nltk.stem import WordNetLemmatizer
from c45_logic import c45, pessimistic_prune, generate_rules

# --- Set Matplotlib to non-interactive mode ---
matplotlib.use('Agg')

# --- NLTK Setup ---
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

# =====================================================================
st.set_page_config(layout="wide", page_title="Survey Insights Assistant")
# =====================================================================

# --- MODERN, AIRY "CARD" CSS THEME ---
st.markdown("""
<style>
    /* 1. Global Background (Soft Light Blue/Gray) */
    .stApp {
        background-color: #F0F4F8; 
    }
    
    /* Global Typography */
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #2D3748; /* Dark Slate Text */
    }

    /* Main Headers */
    h1 {
        color: #1A365D !important; /* Deep Navy */
        font-weight: 800 !important;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }
    h2, h3 {
        color: #2B6CB0 !important; /* Soft Blue */
        font-weight: 600 !important;
        margin-top: 10px;
    }

    /* 2. Container Styling (Pure White Cards with Shadows) */
    /* This targets the containers we create with st.container() */
    [data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
        background-color: #FFFFFF !important; 
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04); /* Soft shadow */
    }

    /* 3. Style the Tabs */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #718096 !important;
        background-color: #E2E8F0 !important; /* Muted gray tab */
        padding: 12px 25px !important;
        margin-right: 5px !important;
        border-radius: 8px 8px 0 0 !important;
        border: none !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2B6CB0 !important;
        background-color: #FFFFFF !important; /* Active tab is white to match card */
        border-top: 4px solid #3182CE !important; /* Blue top highlight */
        box-shadow: 0 -2px 5px rgba(0,0,0,0.02);
    }

    /* General Text Size */
    .stMarkdown p, .stMarkdown li, label {
        font-size: 20px !important;
        line-height: 1.7 !important;
        color: #4A5568 !important;
    }

    /* Dropdowns and Inputs */
    div[data-baseweb="select"] > div, .stTextInput > div > div > input, .stTextArea > div > textarea {
        background-color: #F7FAFC !important;
        border: 1px solid #CBD5E0 !important;
        border-radius: 8px !important;
        color: #2D3748 !important;
    }
    div[data-baseweb="select"]:hover > div { border-color: #63B3ED !important; }

    /* Highlighted Feature Pills */
    .feature-pill {
        display: inline-block;
        background-color: #EBF8FF;
        color: #2B6CB0;
        border: 1px solid #90CDF4;
        padding: 8px 18px;
        border-radius: 20px;
        margin: 5px 10px 10px 0px;
        font-weight: 600;
        font-size: 15px;
    }

    /* Primary Action Buttons */
    .stButton > button {
        background-color: #3182CE !important; 
        color: white !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: none !important;
        transition: 0.3s;
        box-shadow: 0 2px 5px rgba(49, 130, 206, 0.3);
    }
    .stButton > button:hover { background-color: #2B6CB0 !important; transform: translateY(-2px); }

    /* Download Button Specific */
    [data-testid="stDownloadButton"] button {
        background-color: #38A169 !important; /* Soft Green */
        width: 100% !important;
        margin-top: 15px;
        box-shadow: 0 2px 5px rgba(56, 161, 105, 0.3);
    }
    [data-testid="stDownloadButton"] button:hover { background-color: #2F855A !important; }

    /* Hide anchor links next to headers */
    .css-15zrgzn {display: none}
    .css-1629p8f h1 a, .css-1629p8f h2 a, .css-1629p8f h3 a { display: none !important; }
    
    /* AI Result Text Formatting */
    .ai-result-text h2 {
        font-size: 28px !important;
        color: #2C5282 !important;
        border-bottom: 2px solid #EDF2F7;
        padding-bottom: 8px;
        margin-top: 25px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG: Custom Stop Words ---
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "should't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
    'eg', 'ie', 'etc', 'yes', 'no', 'maybe', 'na', 'none', 'nil', 'go', 'use', 'using', 'make', 'made', 'get', 'got'
])

# --- Helper Functions ---
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        for col in df.columns:
            # Check if the column contains mainly numbers before trying to convert
            if df[col].dtype == 'object':
                # Remove whitespace and check if the first non-null value looks like a number
                first_val = str(df[col].dropna().iloc[0]).strip() if not df[col].dropna().empty else ""
                if first_val.replace('.','',1).isdigit():
                    # If it looks like a number, convert safely
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"System Error: Unable to load file. Details: {e}")
        return None

def detect_identifier_columns(df):
    id_cols = []
    for col in df.columns:
        if len(df) > 0 and df[col].nunique() / len(df) > 0.95: id_cols.append(col); continue
        if df[col].dtype == 'object' and df[col].str.contains(r'@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', na=False).any(): id_cols.append(col); continue
        if re.search(r'id|name|email|phone|timestamp', col.lower()): id_cols.append(col)
    return list(set(id_cols))

def analyze_text_responses(df):
    text_columns_to_analyze = []
    potential_text_columns = df.select_dtypes(include=['object']).columns
    for col in potential_text_columns:
        if df[col].nunique() > 10 and df[col].dropna().astype(str).str.split().str.len().mean() > 2:
            text_columns_to_analyze.append(col)
    if not text_columns_to_analyze: return None, None, []
    
    all_words = []
    for col in text_columns_to_analyze:
        text_data = df[col].dropna().astype(str).tolist()
        for text in text_data:
            clean = re.sub(r'[^a-z\s]', '', text.lower())
            words = clean.split()
            meaningful_words = [lemmatizer.lemmatize(w) for w in words if w not in STOP_WORDS and len(w) > 2]
            all_words.extend(meaningful_words)
            
    if not all_words: return None, None, []
    
    word_counts = Counter(all_words)
    most_common = word_counts.most_common(15)
    summary_df = pd.DataFrame(most_common, columns=['Keyword / Theme', 'Mentions'])
    llm_summary_text = ", ".join([f"{word} ({count})" for word, count in most_common])
    return summary_df, llm_summary_text, text_columns_to_analyze

def get_llm_interpretation(api_key, context, chi_results, rules, text_themes=None):
    try:
        genai.configure(api_key=api_key)
        generation_config = {"temperature": 0.2}
        model = genai.GenerativeModel('gemini-flash-latest', generation_config=generation_config)
        
        prompt = f"""
        You are an expert data analyst writing a professional report for a researcher.
        
        **Data Provided:**
        1. Survey Context: {context if context else "None provided."}
        2. Top Predictive Features: {chi_results}
        3. Decision Tree Logic Rules (Format: Total Instances / Errors): {rules}
        """
        if text_themes:
            prompt += f"4. Qualitative Text Themes: {text_themes}\n"

        prompt += """
        **REQUIRED OUTPUT FORMAT:**
        Format your response strictly using these two exact Markdown headers. Do not use A, B, C lettering. Ensure the tone is professional, objective, and analytical.
        
        ## Executive Summary
        Write a comprehensive narrative paragraph synthesizing the findings. 
        - Identify the primary survey question that drives the outcome.
        - Explain key interactions (e.g., 'If X is high, the outcome depends heavily on Y').
        - If Text Themes were provided, explain how they validate or contextualize the mathematical rules.
        - Explicitly note any logical pathways that are less reliable due to high error rates.

        ## Actionable Suggestions
        Give bullet points. 
        Provide bullet points of practical, data-driven recommendations based on the highest-confidence rules discovered by the model.
        Must be bullet points.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        #return f"Interpretation Engine Error: Please verify your API key connection. Details: {e}"
        return str(e)

# --- PDF Report ---
def create_pdf_report(df_raw, df_clean, eda_stats, selected_features, tree_image_bytes, text_stats_df, rules, interpretation):
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16); self.cell(0, 10, 'Survey Analysis & Insights Report', 0, 1, 'C'); self.ln(5)
        def footer(self):
            self.set_y(-15); self.set_font('Arial', 'I', 8); self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    def clean_text(text): return text.encode('latin-1', 'replace').decode('latin-1')
    pdf = PDF(); pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page(); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "1. Data Processing Log", 0, 1); pdf.set_font("Arial", '', 10)
    pdf.cell(0, 6, f"Raw Data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns", 0, 1)
    pdf.cell(0, 6, f"Cleaned Data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns", 0, 1)
    pdf.ln(5); pdf.set_font("Arial", 'B', 11); pdf.cell(0, 8, "Key Drivers Identified (Statistical Selection):", 0, 1); pdf.set_font("Courier", size=9)
    for i, feat in enumerate(selected_features, 1): pdf.cell(0, 5, f"{i}. {clean_text(feat)}", 0, 1)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "2. Statistical Summary", 0, 1)
    if eda_stats is not None:
        w_var = 50; w_num = 15; pdf.set_font("Arial", 'B', 7); pdf.set_fill_color(240, 248, 255)
        headers = ["Variable", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Miss"]
        pdf.cell(w_var, 6, headers[0], 1, 0, 'L', 1)
        for h in headers[1:]: pdf.cell(w_num, 6, h, 1, 0, 'C', 1)
        pdf.ln(); pdf.set_font("Arial", '', 7)
        for index, row in eda_stats.iterrows():
            var_name = str(index); var_name = var_name[:32] + "..." if len(var_name) > 35 else var_name
            pdf.cell(w_var, 6, clean_text(var_name), 1); pdf.cell(w_num, 6, f"{row['count']:.0f}", 1, 0, 'C'); pdf.cell(w_num, 6, f"{row['mean']:.2f}", 1, 0, 'C')
            pdf.cell(w_num, 6, f"{row['std']:.2f}", 1, 0, 'C'); pdf.cell(w_num, 6, f"{row['min']:.1f}", 1, 0, 'C'); pdf.cell(w_num, 6, f"{row['25%']:.1f}", 1, 0, 'C')
            pdf.cell(w_num, 6, f"{row['50%']:.1f}", 1, 0, 'C'); pdf.cell(w_num, 6, f"{row['75%']:.1f}", 1, 0, 'C'); pdf.cell(w_num, 6, f"{row['max']:.1f}", 1, 0, 'C')
            pdf.cell(w_num, 6, str(int(row['missing'])), 1, 1, 'C')
    else: pdf.cell(0, 10, "No numeric stats.", 0, 1)

    pdf.add_page(); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "3. Variable Distributions", 0, 1); pdf.ln(5)
    for col in df_clean.columns:
        if df_clean[col].nunique() > 50 and not pd.api.types.is_numeric_dtype(df_clean[col]): continue 
        fig, ax = plt.subplots(figsize=(6, 3))
        try:
            if pd.api.types.is_numeric_dtype(df_clean[col]) and df_clean[col].nunique() > 15:
                ax.hist(df_clean[col].dropna(), bins=15, color='#4299E1', edgecolor='white', alpha=0.9)
            else:
                df_clean[col].value_counts().sort_index().plot(kind='bar', ax=ax, color='#4299E1', edgecolor='white', rot=0)
            ax.set_title(f"{col[:40]}")
            ax.spines[['top', 'right']].set_visible(False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                plt.tight_layout(); plt.savefig(tmp.name, format='png', dpi=90); tpath = tmp.name
            if pdf.get_y() > 220: pdf.add_page()
            pdf.image(tpath, x=25, w=160); pdf.ln(5); plt.close(fig); os.remove(tpath)
        except: continue

    if text_stats_df is not None and not text_stats_df.empty:
        pdf.add_page(); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "4. Qualitative Feedback Themes", 0, 1); pdf.set_font("Arial", '', 10)
        pdf.set_fill_color(240, 248, 255); pdf.set_font("Arial", 'B', 10); pdf.cell(100, 8, "Keyword / Theme", 1, 0, 'L', 1); pdf.cell(40, 8, "Mentions", 1, 1, 'C', 1)
        pdf.set_font("Arial", '', 10)
        for _, row in text_stats_df.iterrows(): pdf.cell(100, 8, clean_text(str(row['Keyword / Theme'])), 1); pdf.cell(40, 8, str(row['Mentions']), 1, 1, 'C')
        pdf.ln(10)

    pdf.add_page(); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "5. Model Interpretation & Insights", 0, 1)
    if tree_image_bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(tree_image_bytes); tpath = tmp.name
        try: pdf.image(tpath, x=10, w=190)
        except: pass
        finally: 
            if os.path.exists(tpath): os.remove(tpath)
    pdf.ln(10); pdf.set_font("Arial", '', 10)
    clean_interp = interpretation.replace('**', '').replace('__', '')
    pdf.multi_cell(0, 6, clean_text(clean_interp))
    pdf.ln(10); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "6. Extracted Decision Rules", 0, 1)
    pdf.set_font("Courier", size=8); pdf.multi_cell(0, 5, clean_text(rules))
    return pdf.output(dest='S').encode('latin-1')

# --- Streamlit App UI ---
st.title("Intelligent Likert Scale Data Interpreter")

try:
    MY_API_KEY = st.secrets["GOOGLE_API_KEY"]

except:
    MY_API_KEY = None #Fallback if not found

st.sidebar.header("Configuration")
user_provided_key = st.sidebar.text_input(
    "Use your own Google AI Studio API Key (if developer quota is reached)", 
    type="password",
)
st.write("Get a free API key from https://aistudio.google.com/")
# Prioritize user key if provided, otherwise use developer key
api_key = user_provided_key if user_provided_key else MY_API_KEY

# --- State Management ---
if 'df_raw' not in st.session_state: st.session_state.df_raw = None
if 'df_clean' not in st.session_state: st.session_state.df_clean = None
if 'eda_stats' not in st.session_state: st.session_state.eda_stats = None
if 'tree_image' not in st.session_state: st.session_state.tree_image = None
if 'text_stats' not in st.session_state: st.session_state.text_stats = None
if 'rules_text' not in st.session_state: st.session_state.rules_text = None
if 'llm_interpretation' not in st.session_state: st.session_state.llm_interpretation = None
if 'selected_features' not in st.session_state: st.session_state.selected_features = []
if 'current_file_name' not in st.session_state: st.session_state.current_file_name = None

# --- User-Friendly Tab Names ---
tab1, tab2, tab3, tab4 = st.tabs(["1. Upload Survey", "2. Data Overview", "3. Configure Analysis", "4. Insights & Report"])

with tab1:
    with st.container():
        st.subheader("Data Upload")
        st.write("Please upload your survey data file (.csv) to begin.")
        uploaded_file = st.file_uploader("", type=["csv"])
        
        if uploaded_file is not None and uploaded_file.name != st.session_state.current_file_name:
            st.session_state.df_raw = load_data(uploaded_file)
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.df_clean = None 
            st.session_state.tree_image = None
            st.success("Dataset loaded successfully. Please proceed to 'Data Overview'.")

with tab2:
    if st.session_state.df_raw is None: 
        st.info("A dataset must be uploaded to proceed.")
    else:
        df_raw = st.session_state.df_raw
        
        with st.container():
            st.subheader("Data Summary")
            st.write("Review the summary of numeric responses below.")
            df_numeric = df_raw.select_dtypes(include=np.number)
            if not df_numeric.empty:
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Total Survey Responses", df_raw.shape[0])
                col_m2.metric("Numeric Variables", df_numeric.shape[1])
                
                stats = df_numeric.describe().transpose()
                stats = stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                stats['missing'] = df_numeric.isnull().sum()
                stats.index.name = "Variable"
                
                st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)
                st.session_state.eda_stats = stats 
                
                csv = stats.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Statistics as CSV",
                    data=csv,
                    file_name="survey_statistics.csv",
                    mime="text/csv",
                )
            else: 
                st.info("No numeric columns detected in the dataset.")
        
        with st.container():
            st.subheader("Visual Data Distribution")
            st.write("Select a specific question to view how respondents answered.")
            col_to_plot = st.selectbox("Select Survey Question:", df_raw.columns)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if col_to_plot:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fig.patch.set_facecolor('#FFFFFF')
                    ax.set_facecolor('#FFFFFF')
                    is_num = pd.api.types.is_numeric_dtype(df_raw[col_to_plot])
                    uniq = df_raw[col_to_plot].nunique()
                    
                    if is_num and uniq > 15:
                        ax.hist(df_raw[col_to_plot].dropna(), bins=15, color='#4299E1', edgecolor='white', alpha=0.9)
                    else:
                        df_raw[col_to_plot].value_counts().sort_index().plot(kind='bar', ax=ax, color='#4299E1', edgecolor='white', rot=0)
                    
                    ax.spines[['top', 'right']].set_visible(False)
                    ax.spines['bottom'].set_color('#CBD5E0')
                    ax.spines['left'].set_color('#CBD5E0')
                    ax.tick_params(colors='#4A5568')
                    
                    st.pyplot(fig, use_container_width=True)

with tab3:
    if st.session_state.df_raw is None: 
        st.info("A dataset must be uploaded to proceed.")
    else:
        with st.container():
            st.subheader("Step 1: Clean Your Data")
            st.write("The system has identified potential administrative columns (e.g., Names, Timestamps). You may remove any columns you feel that are not needed for the analysis.")
            if st.session_state.df_clean is None: st.session_state.df_clean = st.session_state.df_raw.copy()
            suggested = detect_identifier_columns(st.session_state.df_raw)
            to_drop = st.multiselect("Review columns to remove:", st.session_state.df_raw.columns, default=suggested)
            st.session_state.df_clean = st.session_state.df_raw.drop(columns=to_drop)

        with st.container():
            st.subheader("Step 2: Analyze Text Comments")
            st.write("If your survey includes open-ended text fields, the system will extract the most common themes to provide context for the final report.")
            text_df, text_summary_str, text_cols = analyze_text_responses(st.session_state.df_clean)
            if text_cols:
                st.success(f"Feedback fields processed: {', '.join(text_cols)}")
                st.dataframe(text_df, use_container_width=False)
                st.session_state.text_stats = text_df
            else:
                st.info("No long-form text responses detected.")
                st.session_state.text_stats = None
                text_summary_str = None

        with st.container():
            st.subheader("Step 3: Setup Predictions")
            st.write("What is the primary outcome you want to predict? (Target Variable):")
            target = st.selectbox("Choose your target variable :", st.session_state.df_clean.columns)
            features_available = []
            for c in st.session_state.df_clean.columns:
                if c == target: continue
                if text_cols and c in text_cols: continue
    
                # Skip columns that aren't numbers and have more than 15 unique answers
                # This prevents the "too complex" tree error
                is_numeric = pd.api.types.is_numeric_dtype(st.session_state.df_clean[c])
                if not is_numeric and st.session_state.df_clean[c].nunique() > 15:
                    continue
    
                features_available.append(c)
            max_f = len(features_available)
            
            if max_f > 0:
                st.write("How many top key drivers should the system display ?")
                k_feats = st.slider("Number of top features having the strongest relationship with the target variable : ", 1, max_f, min(5, max_f))
            else:
                k_feats = 0; st.error("Insufficient variables for modeling.")
            st.write("Research Context (Highly Recommended)")
            st.session_state.user_context = st.text_area("Describe what is the survey about : ", placeholder="Briefly describe the survey's purpose to help the AI write a better report...")

            if st.button("Run AI Analysis Pipeline") and k_feats > 0:
                with st.spinner("Processing data, finding patterns, and generating insights..."):
                    try:
                        df_proc = st.session_state.df_clean.dropna(subset=[target])
                        df_enc = df_proc.copy()
                        for c in df_enc.columns:
                            if not pd.api.types.is_numeric_dtype(df_enc[c]):
                                df_enc[c] = LabelEncoder().fit_transform(df_enc[c].astype(str))
                        
                        X = df_enc[features_available]
                        y = df_enc[target]
                        
                        sel = SelectKBest(chi2, k=k_feats).fit(X, y)
                        cols = X.columns[sel.get_support(indices=True)]
                        final_df = df_proc[list(cols) + [target]]
                        st.session_state.selected_features = list(cols)

                        attrs = final_df.columns.drop(target).tolist()
                        tree = c45(final_df, target, attrs)
                        pruned = pessimistic_prune(tree, target)

                        dot = Digraph(graph_attr={'rankdir': 'TB', 'nodesep': '0.5', 'ranksep': '0.8'})
                        pruned.add_to_dot(dot)
                        
                        st.session_state.tree_image = dot.pipe(format='png')
                        st.session_state.rules_text = "\n".join(generate_rules(pruned))

                        interpretation_result = get_llm_interpretation(
                            api_key, st.session_state.user_context, 
                            ", ".join(cols), 
                            st.session_state.rules_text, 
                            text_summary_str
                    )
                        
                        st.session_state.llm_interpretation = get_llm_interpretation(
                            api_key, st.session_state.user_context, ", ".join(cols), st.session_state.rules_text, text_summary_str
                        )
                        

                        if "429" in str(interpretation_result) or "quota" in str(interpretation_result).lower():
                            st.error("⚠️ Developer API Quota Reached.")
                            st.info("The system is currently out of free requests. Please enter your own Google AI Studio API Key in the left sidebar to continue.")
                            st.session_state.llm_interpretation = None
                        else:
                            st.session_state.llm_interpretation = interpretation_result
                            st.success("Analysis complete. Navigate to 'Insights & Reporting'.")

                    except Exception as e:
                        st.error(f"System Error: {e}")

with tab4:
    if st.session_state.tree_image is None:
        st.info("Please configure and run the model in Tab 3 to view insights.")
    else:
        with st.container():
            st.subheader("Key Drivers Identified")
            st.write("The system statistically identified the following variables as having the strongest relationship with your target outcome:")
            pills_html = "".join([f"<span class='feature-pill'>{f}</span>" for f in st.session_state.selected_features])
            st.markdown(pills_html, unsafe_allow_html=True)

        with st.container():
            st.subheader("Visual Decision Pathway")
            st.info("How to read this chart: Follow the arrows down. The format (Total Instances / Misclassifications) shows how reliable a specific path is. The model automatically removes weak or noisy variables to improve reliability. ")
            st.image(st.session_state.tree_image, use_column_width=False) 
            
            with st.expander(" Expand IF-THEN Rules"):
                st.text_area("", st.session_state.rules_text, height=200)

        if st.session_state.llm_interpretation:
            interp = st.session_state.llm_interpretation
            
            # 1. Logic to handle splitting and bullet formatting
            if "## Actionable Suggestions" in interp:
                parts = interp.split("## Actionable Suggestions")
                summary = parts[0].replace("## Executive Summary", "").strip()
                suggestions = parts[1].strip()

                # --- FIX: Force newlines before bullets so they don't smash together ---
                # This ensures every "* **" or " * " starts on a fresh line
                suggestions = suggestions.replace("* **", "\n* **").replace(" * ", "\n* ")

                with st.container():
                    # Executive Summary Section
                    st.markdown('<div class="ai-result-text"><h2>Executive Summary & Logic</h2></div>', unsafe_allow_html=True)
                    st.markdown(summary)
                    
                    st.markdown("<br>", unsafe_allow_html=True) # Add a little breathing room

                    # Suggestions Section
                    st.markdown('<div class="ai-result-text"><h2>Actionable Suggestions</h2></div>', unsafe_allow_html=True)
                    st.markdown(suggestions)
            else:
                # Fallback if AI doesn't use the specific headers
                with st.container():
                    st.markdown(f'<div class="ai-result-text">{interp}</div>', unsafe_allow_html=True)

        with st.container():
            st.subheader("Export Documentation")
            if st.session_state.eda_stats is not None:
                pdf_bytes = create_pdf_report(
                    st.session_state.df_raw,
                    st.session_state.df_clean,
                    st.session_state.eda_stats,
                    st.session_state.selected_features,
                    st.session_state.tree_image,
                    st.session_state.text_stats, 
                    st.session_state.rules_text,
                    st.session_state.llm_interpretation
                )
                st.download_button("Download Consolidated Report (PDF)", pdf_bytes, "Analysis_Report.pdf", "application/pdf")
