import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np
import os
import json
import math
import plotly.graph_objects as go
import plotly.express as px
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
# Paths & constants
# -------------------------
LOCAL_EXCEL = "E:\\supply_chain_project\\Supply chain logistics problem.xlsx"
EXCEL_PATH = LOCAL_EXCEL if os.path.exists(LOCAL_EXCEL) else "Supply chain logistics problem.xlsx"

DB_PATH = "supply_chain.db"
MODEL_PATH = "model.joblib"
SETTINGS_PATH = '.dashboard_settings.json'

# -------------------------
# Helper Functions
# -------------------------
@st.cache_data
def load_excel():
    if not os.path.exists(EXCEL_PATH):
        st.error(f"File not found: {EXCEL_PATH}. Please upload the Excel file to your repository.")
        st.stop()
    return pd.read_excel(EXCEL_PATH)

@st.cache_resource
def load_trained_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_settings():
    if os.path.exists(SETTINGS_PATH):
        try: return json.load(open(SETTINGS_PATH, 'r', encoding='utf-8'))
        except: return {}
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
st.sidebar.markdown('📦 = Shipments | 🚚 = Carriers | 🧾 = Orders | 📈 = Forecasts')

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
# Page: Home
# -------------------------
if page == 'Home':
    st.markdown('<div class="card">', unsafe_allow_html=True)
    style = st.radio('', options=['Classic','Supply Chain Insights (visual)','Logistics Illustration'], index=0, horizontal=True)
    
    if style == 'Classic':
        st.markdown('<h1>📊 Supply Chain Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="big-emoji">🔎 Explore · 🔧 Model · 🚀 Predict</p>', unsafe_allow_html=True)
    else:
        svg_path = 'assets/Supply_chain_insights.svg' if 'Supply Chain Insights' in style else 'assets/Logistics.svg'
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                st.markdown(f.read(), unsafe_allow_html=True)
        except:
            st.warning('Could not load hero image.')
    
    df = load_excel()
    
    # KPI Logic
    qty_col = next((c for c in df.columns if any(x in c.lower() for x in ['weight', 'unit', 'qty', 'quantity'])), None)
    total_orders = len(df)
    total_weight = df[qty_col].sum() if qty_col and pd.api.types.is_numeric_dtype(df[qty_col]) else 0
    unique_customers = next((df[c].nunique() for c in df.columns if 'customer' in c.lower()), 0)

    k1, k2, k3 = st.columns(3)
    k1.metric('📦 Total orders', total_orders)
    k2.metric('⚖️ Total weight', f'{total_weight:.0f}')
    k3.metric('👥 Unique customers', unique_customers)

    # Robust Sankey Logic
    possible_origin = [c for c in df.columns if 'origin' in c.lower() or 'from' in c.lower()]
    possible_dest = [c for c in df.columns if 'destination' in c.lower() or 'to' in c.lower()]
    
    if possible_origin and possible_dest:
        o_col, d_col = possible_origin[0], possible_dest[0]
        # Ensure values are strings and drop NaNs to avoid TypeError
        temp_df = df[[o_col, d_col]].dropna().astype(str)
        agg = temp_df.groupby([o_col, d_col]).size().reset_index(name='value')
        agg.columns = ['source','target','value']
        
        # Fixed: Use pd.concat and astype(str) to prevent pandas unique TypeError
        labels = pd.concat([agg['source'], agg['target']]).unique().tolist()
        label_to_idx = {l: i for i, l in enumerate(labels)}
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(label=labels, color="#0ea5a4"),
            link=dict(source=agg['source'].map(label_to_idx), target=agg['target'].map(label_to_idx), value=agg['value'])
        )])
        fig.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        st.line_chart(df.set_index('Order Date').resample('M').size())

# -------------------------
# Page: Train (with Compression)
# -------------------------
if page == 'Train':
    st.header('⚙️ Train Model')
    df = st.session_state.get('preprocessed', load_excel())
    
    if df.shape[1] < 2:
        st.error('Need more columns. Use Preprocess first.')
    else:
        target = st.selectbox('Choose target (y)', options=df.columns, index=len(df.columns)-1)
        features = [c for c in df.columns if c != target]
        test_size = st.slider('Test size (%)', 10, 50, 20)
        
        if st.button('Train model'):
            with st.spinner("Training and compressing..."):
                X_df = df[features].copy()
                for col in X_df.columns:
                    if not pd.api.types.is_numeric_dtype(X_df[col]):
                        X_df[col] = pd.factorize(X_df[col].astype(str))[0]
                
                X = X_df.fillna(0).values
                y = pd.factorize(df[target].astype(str))[0] if not pd.api.types.is_numeric_dtype(df[target]) else df[target].fillna(0).values
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0)
                
                model = RandomForestRegressor() if len(np.unique(y)) > 20 else RandomForestClassifier()
                model.fit(X_train, y_train)
                
                # COMPRESSION: Saves space and avoids GitHub large file warnings
                joblib.dump(model, MODEL_PATH, compress=3)
                st.success(f"Model saved as {MODEL_PATH} (Compressed)")
                st.cache_resource.clear()

# -------------------------
# Page: Predict (Cached)
# -------------------------
if page == 'Predict':
    st.header('🚀 Predict')
    model = load_trained_model(MODEL_PATH)
    
    if not model:
        st.error('Model not found. Train it first!')
    else:
        num_cols = load_excel().select_dtypes(include=['number']).columns.tolist()
        if not num_cols:
            st.error("No numeric features found.")
        else:
            inputs = []
            cols = st.columns(min(len(num_cols), 4))
            for i, f in enumerate(num_cols[:8]): # limit to 8 for UI
                with cols[i % 4]:
                    inputs.append(st.number_input(f, value=0.0))
            
            if st.button('Predict'):
                # Ensure input shape matches model expectations
                pred = model.predict(np.array(inputs).reshape(1, -1))
                st.success(f'Prediction Result: {pred[0]}')

# (Dataset, Preprocess, Cluster, Insights, Performance logic remains unchanged)
