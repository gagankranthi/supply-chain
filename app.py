import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np
import os
import json
import math
import urllib.request
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# -------------------------
# Styling (gradient background + card)
# -------------------------
page_bg = """
<style>
:root{
  --card-bg: rgba(255,255,255,0.03);
  --accent: #0ea5a4;
  --muted: rgba(230,246,245,0.9);
}
html, body, .stApp {
  height:100%;
}
.stApp {
  background: linear-gradient(135deg, #071023 0%, #0f172a 50%, #052f2e 100%);
  color: var(--muted);
}
.streamlit-expanderHeader {
  color: var(--muted) !important;
}
.stBlock {
  background: transparent;
}
.st-beta {
  background: transparent;
}
/* card */
.card {
  background: var(--card-bg);
  padding: 18px;
  border-radius: 10px;
}
.stButton>button {
  background-color: var(--accent);
  color: white;
  border-radius: 6px;
}
/* header emoji larger */
.big-emoji {font-size: 32px}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------
# Paths & constants (Updated for Cloud Compatibility)
# -------------------------
LOCAL_EXCEL = "E:\\supply_chain_project\\Supply chain logistics problem.xlsx"
# If local path doesn't exist, look in the current folder (for Streamlit Cloud)
EXCEL_PATH = LOCAL_EXCEL if os.path.exists(LOCAL_EXCEL) else "Supply chain logistics problem.xlsx"

DB_PATH = "supply_chain.db"
MODEL_PATH = "model.joblib"
SETTINGS_PATH = '.dashboard_settings.json'

# -------------------------
# Helper Functions (Cached)
# -------------------------
@st.cache_data
def load_excel():
    return pd.read_excel(EXCEL_PATH)

@st.cache_resource
def load_trained_model(path):
    """Loads the model once and keeps it in memory."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_settings():
    if os.path.exists(SETTINGS_PATH):
        try:
            return json.load(open(SETTINGS_PATH, 'r', encoding='utf-8'))
        except Exception:
            return {}
    return {}

def save_settings(d):
    json.dump(d, open(SETTINGS_PATH, 'w', encoding='utf-8'), indent=2)

settings = load_settings()

# -------------------------
# Sidebar / Navigation
# -------------------------
st.sidebar.markdown('# 📦 Supply Chain App')
st.sidebar.write('A compact EDA + ML')
st.sidebar.markdown('### ✨ Legend')
st.sidebar.markdown('📦 = Shipments  ')
st.sidebar.markdown('🚚 = Carriers  ')
st.sidebar.markdown('🧾 = Orders  ')
st.sidebar.markdown('📈 = Forecasts')

page_options = {
    'Home': '🏠 Home',
    'Dataset': '📥 Dataset',
    'Preprocess': '🧹 Preprocess',
    'Cluster': '🧭 Cluster',
    'Insights': '🔗 Insights',
    'Performance': '📈 Performance',
    'Train': '⚙️ Train',
    'Predict': '🚀 Predict'
}
choice = st.sidebar.radio('Navigate', list(page_options.values()))
page = [k for k,v in page_options.items() if v==choice][0]

# -------------------------
# Home
# -------------------------
if page == 'Home':
    st.markdown('<div class="card">', unsafe_allow_html=True)
    style = st.radio('', options=['Classic','Supply Chain Insights (visual)','Logistics Illustration'], index=0, horizontal=True)
    st.session_state['front_style'] = style
    if style == 'Classic':
        st.markdown('<h1>📊 Supply Chain Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="big-emoji">🔎 Explore · 🔧 Model · 🚀 Predict</p>', unsafe_allow_html=True)
    else:
        svg_path = 'assets/Supply_chain_insights.svg' if 'Supply Chain Insights' in style else 'assets/Logistics.svg'
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                svg = f.read()
            st.markdown(svg, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f'Could not load hero image: {e}')
    st.write('This small app demonstrates dataset loading, preprocessing, model training and prediction for the supplied Excel dataset.')

    df = load_excel()
    qty_col = None
    for c in df.columns:
        if any(x in c.lower() for x in ['weight', 'unit', 'qty', 'quantity']):
            qty_col = c
            break
    
    total_orders = len(df)
    total_weight = df[qty_col].sum() if qty_col and pd.api.types.is_numeric_dtype(df[qty_col]) else None
    unique_customers = next((df[c].nunique() for c in df.columns if 'customer' in c.lower()), None)

    k1, k2, k3 = st.columns(3)
    k1.metric('📦 Total orders', total_orders)
    k2.metric('⚖️ Total weight' if total_weight is not None else '⚖️ Quantity (na)', f'{total_weight:.0f}' if total_weight is not None else 'N/A')
    k3.metric('👥 Unique customers', unique_customers if unique_customers is not None else 'N/A')

    # Sankey Logic (Simplified for Home)
    possible_origin = [c for c in df.columns if 'origin' in c.lower() or 'from' in c.lower()]
    possible_dest = [c for c in df.columns if 'destination' in c.lower() or 'to' in c.lower()]
    if possible_origin and possible_dest:
        origin_col, dest_col = possible_origin[0], possible_dest[0]
        agg = df.groupby([origin_col, dest_col])[qty_col if qty_col else origin_col].sum().reset_index()
        agg.columns = ['source','target','value']
        
        labels = list(pd.unique(agg['source'].tolist() + agg['target'].tolist()))
        label_to_idx = {l:i for i,l in enumerate(labels)}
        
        import plotly.graph_objects as go
        mini = go.Figure(data=[go.Sankey(
            node=dict(label=labels), 
            link=dict(source=agg['source'].map(label_to_idx), target=agg['target'].map(label_to_idx), value=agg['value'].head(30))
        )])
        mini.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(mini, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('---')
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        st.line_chart(df.set_index('Order Date').resample('M').size())

# -------------------------
# Train (Updated with Compression)
# -------------------------
if page == 'Train':
    st.header('⚙️ Train Model')
    df = st.session_state.get('preprocessed') if 'preprocessed' in st.session_state else load_excel()
    if df is None or df.shape[1] < 2:
        st.error('Need at least 2 columns (features + target). Go to Preprocess to choose.')
    else:
        cols = df.columns.tolist()
        target = st.selectbox('Choose target (y)', options=cols, index=len(cols)-1)
        features = [c for c in cols if c != target]
        test_size = st.slider('Test size (%)', 10, 50, 20)
        
        if st.button('Train model'):
            with st.spinner("Training and compressing..."):
                X_df = df[features].copy()
                for col in X_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(X_df[col]):
                        X_df[col] = pd.to_numeric(X_df[col])
                    elif not pd.api.types.is_numeric_dtype(X_df[col]):
                        X_df[col] = pd.factorize(X_df[col].fillna(''))[0]
                
                X = X_df.fillna(0).values
                y_ser = df[target].copy()
                y = pd.factorize(y_ser.fillna(''))[0] if not pd.api.types.is_numeric_dtype(y_ser) else y_ser.values
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
                
                if len(np.unique(y)) > 20:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    st.success(f"RMSE: {math.sqrt(mean_squared_error(y_test, model.predict(X_test))):.3f}")
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    st.success(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
                
                # --- THE BIG CHANGE: Compression ---
                # compress=3 reduces size without losing any data
                joblib.dump(model, MODEL_PATH, compress=3)
                st.write(f'Model saved & compressed to {MODEL_PATH}')
                # Clear cache so the new model is loaded in 'Predict'
                st.cache_resource.clear()

# -------------------------
# Predict (Updated with Caching)
# -------------------------
if page == 'Predict':
    st.header('🚀 Predict')
    # Use the cached loader
    model = load_trained_model(MODEL_PATH)
    
    if model is None:
        st.error('Model file not found. Please run the Train tab first.')
    else:
        df = load_excel()
        num = df.select_dtypes(include=['number'])
        if num.shape[1] == 0:
            st.error('No numeric columns available.')
        else:
            features = num.columns.tolist()[:-1]
            st.write('Enter feature values:')
            inputs = []
            cols = st.columns(len(features))
            for i, f in enumerate(features):
                with cols[i]:
                    val = st.number_input(f, value=float(num[f].median()))
                    inputs.append(val)
            
            if st.button('Predict'):
                pred = model.predict(np.array(inputs).reshape(1, -1))
                st.success(f'Prediction: {pred[0]}')

# (Note: Other sections like Dataset, Preprocess, Cluster, etc. remain the same as your original code)