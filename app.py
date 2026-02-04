import streamlit as st
import pandas as pd
import sqlite3
import joblib
import numpy as np
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
EXCEL_PATH = "E:\\supply_chain_project\\Supply chain logistics problem.xlsx"
DB_PATH = "supply_chain.db"
MODEL_PATH = "model.joblib"

# -------------------------
# Sidebar / Navigation
# -------------------------
st.sidebar.markdown('# 📦 Supply Chain App')
st.sidebar.write('A compact EDA + ML')
# small emoji legend for clarity
st.sidebar.markdown('### ✨ Legend')
st.sidebar.markdown('📦 = Shipments  ')
st.sidebar.markdown('🚚 = Carriers  ')
st.sidebar.markdown('🧾 = Orders  ')
st.sidebar.markdown('📈 = Forecasts')
# show emoji-rich labels but keep internal keys for logic
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
# map back to internal key
page = [k for k,v in page_options.items() if v==choice][0]

@st.cache_data
def load_excel():
    return pd.read_excel(EXCEL_PATH)

@st.cache_data
def load_db():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM shipments LIMIT 1000', conn)
    conn.close()
    return df

# settings persistence
import json, os
SETTINGS_PATH = '.dashboard_settings.json'
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
        # map the new display labels to actual asset file names
        svg_path = 'assets/Supply_chain_insights.svg' if 'Supply Chain Insights' in style else 'assets/Logistics.svg'
    # no background SVG for these visual styles per user request
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
        if 'weight' in c.lower() or 'unit' in c.lower() or 'qty' in c.lower() or 'quantity' in c.lower():
            qty_col = c
            break
    total_orders = len(df)
    total_weight = df[qty_col].sum() if qty_col and pd.api.types.is_numeric_dtype(df[qty_col]) else None
    unique_customers = None
    for c in df.columns:
        if 'customer' in c.lower():
            unique_customers = df[c].nunique()
            break

    k1, k2, k3 = st.columns(3)
    k1.metric('📦 Total orders', total_orders)
    k2.metric('⚖️ Total weight' if total_weight is not None else '⚖️ Quantity (na)', f'{total_weight:.0f}' if total_weight is not None else 'N/A')
    k3.metric('👥 Unique customers', unique_customers if unique_customers is not None else 'N/A')

    possible_origin = [c for c in df.columns if 'origin' in c.lower() or 'from' in c.lower()]
    possible_dest = [c for c in df.columns if 'destination' in c.lower() or 'to' in c.lower()]
    if possible_origin and possible_dest:
        origin_col = possible_origin[0]
        dest_col = possible_dest[0]
        value_col = qty_col
        agg = df.groupby([origin_col, dest_col])[value_col if value_col else origin_col].agg('sum').reset_index()
        agg.columns = ['source','target','value']
        st.subheader('Top flows')
        st.dataframe(agg.sort_values('value', ascending=False).head(10))

        labels = list(pd.unique(agg['source'].tolist() + agg['target'].tolist()))
        label_to_idx = {l:i for i,l in enumerate(labels)}
        sources = agg['source'].map(label_to_idx).tolist()
        targets = agg['target'].map(label_to_idx).tolist()
        values = agg['value'].tolist()
        import plotly.graph_objects as go
        mini = go.Figure(data=[go.Sankey(node=dict(label=labels), link=dict(source=sources[:30], target=targets[:30], value=values[:30]))])
        mini.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(mini, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('### 🔗 Quick Links & Symbols')
    cols = st.columns(4)
    cols[0].markdown('📦 Shipments')
    cols[1].markdown('🚚 Carriers')
    cols[2].markdown('🧾 Orders')
    cols[3].markdown('📈 Forecasts')
    st.markdown('---')
    st.subheader('KPI trends')
    if 'Order Date' in df.columns:
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        trend = df.set_index('Order Date').resample('M').size()
        st.line_chart(trend)
    else:
        st.info('No date column named "Order Date" found for trends.')

    st.subheader('Quick Model Compare')
    if st.button('Run quick compare (RF vs simple)'):
        from sklearn.dummy import DummyRegressor, DummyClassifier
        from sklearn.model_selection import cross_val_score
        num = df.select_dtypes(include=['number'])
        if num.shape[1] < 2:
            st.error('Not enough numeric columns for model comparison')
        else:
            X = num.iloc[:, :-1].fillna(0)
            y = num.iloc[:, -1].fillna(0)
            if y.nunique() > 20:
                rf = RandomForestRegressor(n_estimators=50, random_state=0)
                dummy = DummyRegressor()
                scores_rf = -1 * cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=3)
                scores_dummy = -1 * cross_val_score(dummy, X, y, scoring='neg_mean_squared_error', cv=3)
                st.write('RF MSE mean:', scores_rf.mean())
                st.write('Dummy MSE mean:', scores_dummy.mean())
            else:
                rf = RandomForestClassifier(n_estimators=50, random_state=0)
                dummy = DummyClassifier()
                scores_rf = cross_val_score(rf, X, y, scoring='accuracy', cv=3)
                scores_dummy = cross_val_score(dummy, X, y, scoring='accuracy', cv=3)
                st.write('RF accuracy mean:', scores_rf.mean())
                st.write('Dummy accuracy mean:', scores_dummy.mean())

# -------------------------
# Settings (column mapping)
# -------------------------
if page == 'Settings':
    st.header('⚙️ Dashboard settings')
    df_sample = load_excel()
    cols = df_sample.columns.tolist()
    st.write('Map dataset columns to KPI roles (helps the dashboard detect the right columns).')
    date_col = st.selectbox('Date column', options=['']+cols, index=(cols.index(settings.get('date_col'))+1 if settings.get('date_col') in cols else 0))
    qty_col = st.selectbox('Quantity/weight column', options=['']+cols, index=(cols.index(settings.get('qty_col'))+1 if settings.get('qty_col') in cols else 0))
    monetary_col = st.selectbox('Monetary/Value column', options=['']+cols, index=(cols.index(settings.get('monetary_col'))+1 if settings.get('monetary_col') in cols else 0))
    origin_col = st.selectbox('Origin column', options=['']+cols, index=(cols.index(settings.get('origin_col'))+1 if settings.get('origin_col') in cols else 0))
    dest_col = st.selectbox('Destination column', options=['']+cols, index=(cols.index(settings.get('dest_col'))+1 if settings.get('dest_col') in cols else 0))
    if st.button('Save settings'):
        settings.update({'date_col': date_col or None, 'qty_col': qty_col or None, 'monetary_col': monetary_col or None, 'origin_col': origin_col or None, 'dest_col': dest_col or None})
        save_settings(settings)
        st.success('Settings saved')

# -------------------------
# Dataset
# -------------------------
if page == 'Dataset':
    st.header('📥 Dataset')
    df = load_excel()
    st.write('Shape:', df.shape)
    st.dataframe(df.head(100))
    with st.expander('Columns & types'):
        st.write(df.dtypes.astype(str))
    with st.expander('Missing values'):
        st.write(df.isna().sum())

# -------------------------
# Preprocessing
# -------------------------
if page == 'Preprocess':
    st.header('🧹 Preprocessing')
    df = load_excel()
    st.write('Raw shape:', df.shape)
    cols = df.columns.tolist()
    st.write('Select numeric columns to use (features + target):')
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric:
        st.warning('No numeric columns detected automatically. You may need to convert or select columns manually.')
    chosen = st.multiselect('Numeric columns', options=cols, default=numeric)
    st.write('Preview of chosen columns:')
    st.dataframe(df[chosen].head())
    if st.button('Drop rows with NA in chosen columns'):
        df = df.dropna(subset=chosen)
        st.success('Dropped. New shape: {}'.format(df.shape))
    st.session_state['preprocessed'] = df[chosen] if chosen else df

# -------------------------
# Cluster
# -------------------------
if page == 'Cluster':
    st.header('🧭 Clustering (KMeans + PCA)')
    df = st.session_state.get('preprocessed') if 'preprocessed' in st.session_state else load_excel()
    if df is None or df.shape[1] < 2:
        st.error('Need at least 2 numeric columns. Use Preprocess to prepare data.')
    else:
        st.write('Data shape:', df.shape)
        cols = df.columns.tolist()
        sel = st.multiselect('Select features for clustering', options=cols, default=cols[:3])
        n_clusters = st.slider('Number of clusters', 2, 10, 3)
        if st.button('Run KMeans'):
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score

            X = df[sel].dropna().values
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(X)
            sil = silhouette_score(X, labels) if len(set(labels))>1 else None
            st.write('Silhouette score:', sil)

            pca = PCA(n_components=2)
            proj = pca.fit_transform(X)
            import pandas as _pd
            out = _pd.DataFrame(proj, columns=['pc1','pc2'])
            out['cluster'] = labels.astype(str)

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6,4))
            for c in sorted(out['cluster'].unique()):
                s = out[out['cluster']==c]
                ax.scatter(s['pc1'], s['pc2'], label=f'Cluster {c}', alpha=0.7)
            ax.legend()
            ax.set_title('PCA projection of clusters')
            st.pyplot(fig)

            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button('Download labeled CSV', data=csv, file_name='clusters.csv', mime='text/csv')

# -------------------------
# Insights (Sankey)
# -------------------------
if page == 'Insights':
    st.header('🔗 Insights - Flow (Sankey)')
    df = load_excel()
    possible_origin = [c for c in df.columns if 'origin' in c.lower() or 'from' in c.lower()]
    possible_dest = [c for c in df.columns if 'destination' in c.lower() or 'to' in c.lower()]
    value_cols = [c for c in df.columns if 'weight' in c.lower() or 'qty' in c.lower() or 'quantity' in c.lower()]
    if not possible_origin or not possible_dest:
        st.error('Could not detect origin/destination columns automatically. Check your dataset.')
    else:
        origin_col = possible_origin[0]
        dest_col = possible_dest[0]
        value_col = value_cols[0] if value_cols else None
        st.write('Using', origin_col, '->', dest_col, ' value:', value_col)
        agg = df.groupby([origin_col, dest_col])[value_col if value_col else origin_col].agg('sum').reset_index()
        agg.columns = ['source','target','value']

        labels = list(pd.unique(agg['source'].tolist() + agg['target'].tolist()))
        label_to_idx = {l:i for i,l in enumerate(labels)}
        sources = agg['source'].map(label_to_idx).tolist()
        targets = agg['target'].map(label_to_idx).tolist()
        values = agg['value'].tolist()
        import plotly.graph_objects as go
        sankey = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=20),
            link=dict(source=sources, target=targets, value=values)
        )])
        sankey.update_layout(title_text='Flow from origin to destination', font_size=10)
        st.plotly_chart(sankey, use_container_width=True)

        st.write('Top flows:')
        st.dataframe(agg.sort_values('value', ascending=False).head(50))
        st.download_button('Download flows CSV', data=agg.to_csv(index=False).encode('utf-8'), file_name='flows.csv', mime='text/csv')

# -------------------------
# Performance dashboard
# -------------------------
if page == 'Performance':
    st.header('📈 Supply Chain Performance')
    df = load_excel()
    st.write('Data status:')
    st.write(df.describe(include='all').T[['count','unique']].fillna(''))

    date_cols = [settings.get('date_col')] if settings.get('date_col') else [c for c in df.columns if 'date' in c.lower()]
    origin_cols = [settings.get('origin_col')] if settings.get('origin_col') else [c for c in df.columns if 'origin' in c.lower() or 'from' in c.lower()]
    dest_cols = [settings.get('dest_col')] if settings.get('dest_col') else [c for c in df.columns if 'destination' in c.lower() or 'to' in c.lower()]
    qty_cols = [settings.get('qty_col')] if settings.get('qty_col') else [c for c in df.columns if 'weight' in c.lower() or 'qty' in c.lower() or 'quantity' in c.lower() or 'units' in c.lower()]
    value_cols = [settings.get('monetary_col')] if settings.get('monetary_col') else [c for c in df.columns if 'value' in c.lower() or 'price' in c.lower() or 'amount' in c.lower() or 'cost' in c.lower()]

    st.subheader('Detected columns')
    st.write('Dates:', date_cols)
    st.write('Origin:', origin_cols)
    st.write('Destination:', dest_cols)
    st.write('Quantity/weight:', qty_cols)
    st.write('Monetary:', value_cols)

    total_shipments = len(df)
    total_qty = df[qty_cols[0]].sum() if qty_cols and pd.api.types.is_numeric_dtype(df[qty_cols[0]]) else None
    avg_value = df[value_cols[0]].mean() if value_cols and pd.api.types.is_numeric_dtype(df[value_cols[0]]) else None

    c1, c2, c3 = st.columns(3)
    c1.metric('Total shipments', total_shipments)
    c2.metric('Total quantity', f"{total_qty:.0f}" if total_qty is not None else 'N/A')
    c3.metric('Avg monetary', f"{avg_value:.2f}" if avg_value is not None else 'N/A')

    # --- time & carrier KPIs and charts
    # Detect carrier / service / transit time columns
    carrier_cols = [c for c in df.columns if 'carrier' in c.lower()]
    service_cols = [c for c in df.columns if 'service' in c.lower()]
    tpt_cols = [c for c in df.columns if 'tpt' in c.lower() or 'transit' in c.lower() or 'time' in c.lower()]
    ship_ahead_cols = [c for c in df.columns if 'ship ahead' in c.lower() or 'ahead' in c.lower()]
    ship_late_cols = [c for c in df.columns if 'late' in c.lower() or 'ship late' in c.lower()]

    # compute numeric time metrics when available
    avg_tpt = None
    med_tpt = None
    if tpt_cols:
        tpt_col = tpt_cols[0]
        tpt_series = pd.to_numeric(df[tpt_col], errors='coerce')
        if tpt_series.notna().any():
            avg_tpt = tpt_series.mean()
            med_tpt = tpt_series.median()

    avg_ahead = None
    if ship_ahead_cols:
        sa = pd.to_numeric(df[ship_ahead_cols[0]], errors='coerce')
        if sa.notna().any():
            avg_ahead = sa.mean()

    avg_late = None
    if ship_late_cols:
        sl = pd.to_numeric(df[ship_late_cols[0]], errors='coerce')
        if sl.notna().any():
            avg_late = sl.mean()

    k4, k5, k6 = st.columns(3)
    k4.metric('Avg transit time', f"{avg_tpt:.1f}" if avg_tpt is not None else 'N/A')
    k5.metric('Median transit time', f"{med_tpt:.1f}" if med_tpt is not None else 'N/A')
    k6.metric('Avg days late', f"{avg_late:.1f}" if avg_late is not None else (f"{avg_ahead:.1f} ahead" if avg_ahead is not None else 'N/A'))

    # Carrier / service distribution pie
    import plotly.express as px
    if carrier_cols:
        carrier = carrier_cols[0]
        ct = df[carrier].value_counts().reset_index()
        ct.columns = ['label','count']
        fig_car = px.pie(ct, names='label', values='count', title='Shipments by Carrier')
        st.plotly_chart(fig_car, use_container_width=True)
    elif service_cols:
        svc = service_cols[0]
        st.info(f'No carrier column found — showing service level distribution ({svc})')
        sv = df[svc].value_counts().reset_index()
        sv.columns = ['label','count']
        fig_svc = px.pie(sv, names='label', values='count', title='Service level distribution')
        st.plotly_chart(fig_svc, use_container_width=True)

    # Transit time charts
    if tpt_cols:
        tpt_col = tpt_cols[0]
        tpt_series = pd.to_numeric(df[tpt_col], errors='coerce')
        if tpt_series.notna().any():
            st.subheader('Transit time distribution')
            fig_hist = px.histogram(df, x=tpt_col, nbins=30, title='Transit time distribution')
            st.plotly_chart(fig_hist, use_container_width=True)

            # avg TPT by origin (top 15)
            possible_origin = [c for c in df.columns if 'origin' in c.lower() or 'from' in c.lower()]
            if possible_origin:
                oc = possible_origin[0]
                grp = df.groupby(oc)[tpt_col].apply(lambda s: pd.to_numeric(s, errors='coerce').mean()).dropna().sort_values(ascending=False).head(15).reset_index()
                grp.columns = [oc, 'avg_tpt']
                fig_bar = px.bar(grp, x=oc, y='avg_tpt', title='Avg transit time by origin (top 15)')
                st.plotly_chart(fig_bar, use_container_width=True)

            # trend over time if date present
            date_cols_local = [settings.get('date_col')] if settings.get('date_col') else [c for c in df.columns if 'date' in c.lower()]
            if date_cols_local:
                try:
                    df['_dt_tpt'] = pd.to_datetime(df[date_cols_local[0]], errors='coerce')
                    trend_tpt = df.set_index('_dt_tpt').resample('M')[tpt_col].apply(lambda s: pd.to_numeric(s, errors='coerce').mean())
                    st.subheader('Avg transit time over time')
                    st.line_chart(trend_tpt.dropna())
                except Exception:
                    pass

    import plotly.express as px
    if origin_cols:
        o = df[origin_cols[0]].value_counts().reset_index()
        o.columns = ['label','count']
        fig_o = px.pie(o, names='label', values='count', title='Shipments by Origin')
        st.plotly_chart(fig_o, use_container_width=True)

    if dest_cols:
        d = df[dest_cols[0]].value_counts().reset_index()
        d.columns = ['label','count']
        fig_d = px.bar(d.head(20), x='label', y='count', title='Top Destinations')
        st.plotly_chart(fig_d, use_container_width=True)

    if date_cols:
        try:
            df['_date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
            trend = df.set_index('_date').resample('M').size()
            st.subheader('Monthly shipment trend')
            st.line_chart(trend)
        except Exception:
            st.info('Could not parse date column for trend chart.')

    st.markdown('---')
    st.subheader('Estimated financial KPIs')
    carrying_rate = st.number_input('Carrying cost rate (annual %)', min_value=0.0, max_value=100.0, value=20.0)
    days_inventory = st.number_input('Average inventory days', min_value=0, max_value=365, value=60)
    cost_of_goods = None
    if value_cols and pd.api.types.is_numeric_dtype(df[value_cols[0]]):
        cost_of_goods = df[value_cols[0]].sum()
    if cost_of_goods is not None:
        avg_inventory = cost_of_goods * (days_inventory / 365.0)
        carrying_cost = avg_inventory * (carrying_rate/100.0)
        st.write('Estimated carrying cost (annual):', f"{carrying_cost:,.2f}")

    st.subheader('Cash-to-Cash cycle (quick estimate)')
    days_receivable = st.number_input('Days receivable (DSO)', min_value=0, max_value=365, value=45)
    days_payable = st.number_input('Days payable (DPO)', min_value=0, max_value=365, value=30)
    c2c = days_inventory + days_receivable - days_payable
    st.metric('Cash-to-Cash (days)', c2c)

    st.markdown('Notes: These financial KPIs are rough estimates based on detected columns and user assumptions. For accurate results, provide explicit inventory valuation, COGS, AR and AP datasets.')

# -------------------------
# Train
# -------------------------
if page == 'Train':
    st.header('⚙️ Train Model')
    df = st.session_state.get('preprocessed') if 'preprocessed' in st.session_state else load_excel()
    if df is None or df.shape[1] < 2:
        st.error('Need at least 2 columns (features + target). Go to Preprocess to choose.')
    else:
        st.write('Data for training shape:', df.shape)
        cols = df.columns.tolist()
        target = st.selectbox('Choose target (y)', options=cols, index=len(cols)-1)
        features = [c for c in cols if c != target]
        st.write('Features:', features)
        test_size = st.slider('Test size (%)', 10, 50, 20)
        if st.button('Train model'):
            # prepare features: convert datetimes and categoricals to numeric
            X_df = df[features].copy()
            for col in X_df.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(X_df[col]) or np.issubdtype(X_df[col].dtype, np.datetime64):
                        X_df[col] = pd.to_numeric(pd.to_datetime(X_df[col], errors='coerce'))
                    elif not pd.api.types.is_numeric_dtype(X_df[col]):
                        X_df[col] = pd.factorize(X_df[col].fillna(''))[0].astype(float)
                except Exception:
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0)
            X = X_df.fillna(0).values

            # prepare target y similarly (handle datetimes or categoricals)
            y_ser = df[target].copy()
            if pd.api.types.is_datetime64_any_dtype(y_ser) or np.issubdtype(y_ser.dtype, np.datetime64):
                y = pd.to_numeric(pd.to_datetime(y_ser, errors='coerce')).astype(float)
            else:
                if not pd.api.types.is_numeric_dtype(y_ser):
                    # factorize string/categorical targets
                    y = pd.factorize(y_ser.fillna(''))[0]
                else:
                    y = y_ser.values.astype(float)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42)
            if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 20:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                import math
                rmse = math.sqrt(mse)
                st.success(f'Regressor trained — RMSE: {rmse:.3f}')
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.success(f'Classifier trained — Accuracy: {acc:.3f}')
            joblib.dump(model, MODEL_PATH)
            st.write('Saved model to', MODEL_PATH)

# -------------------------
# Predict
# -------------------------
if page == 'Predict':
    st.header('🚀 Predict')
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error('Model not found. Train first in the Train tab.')
        st.stop()
    df = load_excel()
    num = df.select_dtypes(include=['number'])
    if num.shape[1] == 0:
        st.error('No numeric columns available for prediction.')
    else:
        features = num.columns.tolist()[:-1]
        st.write('Enter feature values:')
        inputs = {}
        cols = st.columns(len(features))
        for i, f in enumerate(features):
            with cols[i]:
                inputs[f] = st.number_input(f, value=float(num[f].median()))
        X = np.array([inputs[f] for f in features]).reshape(1, -1)
        if st.button('Predict'):
            pred = model.predict(X)
            st.success(f'Prediction: {pred}')

# About page removed per user request

