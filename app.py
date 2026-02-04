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
# Styling (Preserved Design)
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
.card { background: var(--card-bg); padding: 18px; border-radius: 10px; }
.stButton>button { background-color: var(--accent); color: white; border-radius: 6px; }
.big-emoji {font-size: 32px}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------
# Paths & constants
# -------------------------
LOCAL_EXCEL = "E:\\supply_chain_project\\Supply chain logistics problem.xlsx"
EXCEL_PATH = LOCAL_EXCEL if os.path.exists(LOCAL_EXCEL) else "Supply chain logistics problem.xlsx"
SETTINGS_PATH = '.dashboard_settings.json'

# -------------------------
# Logic: The "No-File" Model Engine
# -------------------------
@st.cache_data
def load_excel():
    if not os.path.exists(EXCEL_PATH):
        st.error("Dataset not found. Please ensure 'Supply chain logistics problem.xlsx' is in the folder.")
        st.stop()
    
    df = pd.read_excel(EXCEL_PATH)
    
    # --- FIX FOR "LargeUtf8" ERROR ---
    # This converts "LargeUtf8" and "Object" types to standard strings
    # which prevents the frontend decoding error.
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) == 'string':
            df[col] = df[col].astype(str)
            
    # Additionally, handle any potential mixed types that cause Arrow issues
    df = df.infer_objects()
    
    return df
@st.cache_resource
def train_internal_model(df_input, target_col):
    """Trains the model in memory. No .joblib file is created."""
    # Automated Preprocessing
    df = df_input.copy()
    
    # Identify features
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col not in numeric_cols and len(df[target_col].unique()) > 20:
         # If target is text but has many values, try converting to numeric
         df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Simple cleaning for internal training
    df = df.dropna()
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert Categorical features to codes
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.factorize(X[col])[0]
    
    # Handle target encoding if text
    if not pd.api.types.is_numeric_dtype(y):
        y = pd.factorize(y)[0]

    # Model Selection
    if pd.api.types.is_numeric_dtype(y) and len(np.unique(y)) > 20:
        model = RandomForestRegressor(n_estimators=50, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
    model.fit(X, y)
    return model, X.columns.tolist()

# -------------------------
# Sidebar / Navigation
# -------------------------
st.sidebar.markdown('# 📦 Supply Chain App')
page_options = {'Home': '🏠 Home', 'Dataset': '📥 Dataset', 'Insights': '🔗 Insights', 'Predict': '🚀 Predict'}
choice = st.sidebar.radio('Navigate', list(page_options.values()))
page = [k for k,v in page_options.items() if v==choice][0]

# -------------------------
# Home (Design Preserved)
# -------------------------
if page == 'Home':
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h1>📊 Supply Chain Dashboard</h1>', unsafe_allow_html=True)
    df = load_excel()
    
    qty_col = next((c for c in df.columns if 'qty' in c.lower() or 'quantity' in c.lower()), df.columns[0])
    k1, k2, k3 = st.columns(3)
    k1.metric('📦 Total orders', len(df))
    k2.metric('⚖️ Total weight', f"{df[qty_col].sum():.0f}" if pd.api.types.is_numeric_dtype(df[qty_col]) else "N/A")
    k3.metric('👥 Unique Items', df.iloc[:, 0].nunique())
    
    # Robust Sankey
    possible_origin = [c for c in df.columns if 'origin' in c.lower() or 'from' in c.lower()]
    possible_dest = [c for c in df.columns if 'destination' in c.lower() or 'to' in c.lower()]
    if possible_origin and possible_dest:
        agg = df.groupby([possible_origin[0], possible_dest[0]]).size().reset_index(name='value')
        agg.columns = ['source', 'target', 'value']
        agg['source'] = agg['source'].astype(str)
        agg['target'] = agg['target'].astype(str)
        labels = pd.concat([agg['source'], agg['target']]).unique().tolist()
        label_map = {l: i for i, l in enumerate(labels)}
        fig = go.Figure(data=[go.Sankey(node=dict(label=labels), link=dict(source=agg['source'].map(label_map), target=agg['target'].map(label_map), value=agg['value']))])
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Predict (The "Live Training" Section)
# -------------------------
if page == 'Predict':
    st.header('🚀 Live Prediction Engine')
    st.info("Note: This app trains the model in RAM every session to avoid external file dependencies.")
    
    df = load_excel()
    target_options = df.columns.tolist()
    target_col = st.selectbox("Select Target Variable to Predict", options=target_options, index=len(target_options)-1)
    
    # Train the model in memory
    with st.spinner("Training model from dataset... please wait."):
        model, feature_names = train_internal_model(df, target_col)
    
    st.success("Model ready for this session!")
    
    # Inputs
    st.subheader("Enter values for prediction:")
    user_inputs = []
    cols = st.columns(3)
    for i, feat in enumerate(feature_names[:6]): # Show first 6 features for clean UI
        with cols[i % 3]:
            val = st.number_input(f"{feat}", value=0.0)
            user_inputs.append(val)
            
    if st.button("Calculate Prediction"):
        # Match input dimensions
        full_input = np.zeros(len(feature_names))
        full_input[:len(user_inputs)] = user_inputs
        prediction = model.predict(full_input.reshape(1, -1))
        st.balloons()
        st.metric("Predicted Value", f"{prediction[0]:.2f}")

# -------------------------
# Other sections (Dataset, Insights)
# -------------------------
if page == 'Dataset':
    st.header('📥 Dataset')
    st.dataframe(load_excel().head(100))

if page == 'Insights':
    st.header('🔗 Insights')
    df = load_excel()
    st.bar_chart(df.iloc[:, 1].value_counts().head(10))
