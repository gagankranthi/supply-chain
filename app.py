import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import math
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# -------------------------
# Styling (Design preserved)
# -------------------------
page_bg = """
<style>
:root{
  --card-bg: rgba(255,255,255,0.03);
  --accent: #0ea5a4;
  --muted: rgba(230,246,245,0.9);
}
html, body, .stApp { height:100%; }
.stApp {
  background: linear-gradient(135deg, #071023 0%, #0f172a 50%, #052f2e 100%);
  color: var(--muted);
}
.card { background: var(--card-bg); padding: 18px; border-radius: 10px; margin-bottom: 20px; }
.stButton>button { background-color: var(--accent); color: white; border-radius: 6px; width: 100%; }
.big-emoji {font-size: 32px}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------
# Paths & Environment
# -------------------------
LOCAL_EXCEL = "E:\\supply_chain_project\\Supply chain logistics problem.xlsx"
EXCEL_PATH = LOCAL_EXCEL if os.path.exists(LOCAL_EXCEL) else "Supply chain logistics problem.xlsx"

# -------------------------
# Data & Model Logic
# -------------------------
@st.cache_data
def load_excel():
    if not os.path.exists(EXCEL_PATH):
        st.error(f"Dataset not found at {EXCEL_PATH}. Please check your file path.")
        st.stop()
    
    df = pd.read_excel(EXCEL_PATH)
    
    # --- FIX: LargeUtf8 Frontend Error ---
    # We convert all object/text columns to standard strings to avoid Arrow decoding errors
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == 'string':
            df[col] = df[col].astype(str)
    return df

@st.cache_resource
def train_internal_model(df_input, target_col):
    """Trains the model in RAM. Zero reliance on external .joblib files."""
    df = df_input.copy().dropna()
    
    # Feature Engineering
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert Categorical features and handle mixed types
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.factorize(X[col].astype(str))[0]
    
    # Determine if Regression or Classification
    is_regression = pd.api.types.is_numeric_dtype(y) and len(y.unique()) > 20
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y.astype(str))[0]

    model = RandomForestRegressor(n_estimators=50, random_state=42) if is_regression else RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist(), is_regression

# -------------------------
# Sidebar Navigation
# -------------------------
st.sidebar.markdown('# 📦 Supply Chain App')
st.sidebar.write('MSc Data Science Project')
page_options = {'Home': '🏠 Home', 'Dataset': '📥 Dataset', 'Insights': '🔗 Insights', 'Predict': '🚀 Predict'}
choice = st.sidebar.radio('Navigate', list(page_options.values()))
page = [k for k,v in page_options.items() if v==choice][0]

# -------------------------
# Page: Home
# -------------------------
if page == 'Home':
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h1>📊 Supply Chain Dashboard</h1>', unsafe_allow_html=True)
    
    df = load_excel()
    qty_col = next((c for c in df.columns if any(x in c.lower() for x in ['qty', 'quantity', 'weight'])), df.columns[0])
    
    k1, k2, k3 = st.columns(3)
    k1.metric('📦 Total orders', len(df))
    k2.metric('⚖️ Total Vol/Qty', f"{df[qty_col].sum():,.0f}" if pd.api.types.is_numeric_dtype(df[qty_col]) else "N/A")
    k3.metric('👥 Unique Nodes', df.iloc[:, 0].nunique())

    # Sankey Flow
    st.subheader("Network Flow Overview")
    possible_origin = [c for c in df.columns if any(x in c.lower() for x in ['origin', 'from'])]
    possible_dest = [c for c in df.columns if any(x in c.lower() for x in ['destination', 'to'])]
    
    if possible_origin and possible_dest:
        agg = df.groupby([possible_origin[0], possible_dest[0]]).size().reset_index(name='value')
        agg.columns = ['source', 'target', 'value']
        agg['source'], agg['target'] = agg['source'].astype(str), agg['target'].astype(str)
        
        labels = pd.concat([agg['source'], agg['target']]).unique().tolist()
        label_map = {l: i for i, l in enumerate(labels)}
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels, color="#0ea5a4", thickness=20),
            link=dict(source=agg['source'].map(label_map), target=agg['target'].map(label_map), value=agg['value'])
        )])
        fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Page: Dataset
# -------------------------
if page == 'Dataset':
    st.header('📥 Dataset Explorer')
    df = load_excel()
    st.write(f"Showing first 100 rows of {len(df)} total records.")
    st.dataframe(df.head(100))
    st.download_button("Download Clean Data", df.to_csv(index=False), "cleaned_data.csv", "text/csv")

# -------------------------
# Page: Insights
# -------------------------
if page == 'Insights':
    st.header('🔗 Supply Chain Insights')
    df = load_excel()
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 Origins")
        st.bar_chart(df.iloc[:, 0].value_counts().head(10))
    with c2:
        st.subheader("Top 10 Destinations")
        st.bar_chart(df.iloc[:, 1].value_counts().head(10))

# -------------------------
# Page: Predict (Live Engine)
# -------------------------
if page == 'Predict':
    st.header('🚀 Live ML Prediction')
    st.info("This engine trains a Random Forest model in real-time using your current dataset.")
    
    df = load_excel()
    target_col = st.selectbox("What would you like to predict?", options=df.columns, index=len(df.columns)-1)
    
    with st.spinner("Training model in RAM..."):
        model, feature_names, is_regression = train_internal_model(df, target_col)
    
    st.success("Model trained successfully for this session!")
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Input Parameters")
    user_inputs = []
    # Display up to 9 features for input
    display_feats = feature_names[:9]
    cols = st.columns(3)
    for i, feat in enumerate(display_feats):
        with cols[i % 3]:
            # Use median as default to make it easy for the user
            default_val = 0.0
            if pd.api.types.is_numeric_dtype(df[feat]):
                default_val = float(df[feat].median())
            
            val = st.number_input(f"{feat}", value=default_val)
            user_inputs.append(val)
    
    if st.button("Generate Prediction"):
        # Construct full feature vector (fill remaining with zeros if needed)
        full_vector = np.zeros(len(feature_names))
        full_vector[:len(user_inputs)] = user_inputs
        
        prediction = model.predict(full_vector.reshape(1, -1))
        
        st.markdown("---")
        if is_regression:
            st.metric("Predicted Value", f"{prediction[0]:,.2f}")
        else:
            st.metric("Predicted Category", f"{prediction[0]}")
        st.balloons()
    st.markdown('</div>', unsafe_allow_html=True)
