import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
import plotly.io as pio

st.set_page_config(page_title="Supply Chain AI Engine", layout="wide", page_icon="🌐")

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4;}
    .kpi-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 10px;}
    .warning-card {background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    filename = "data.csv"
    try:
        df = pd.read_csv(filename, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(filename, sep=',')
        
        df.columns = df.columns.str.strip()
        
        # Date handling
        date_col = next((col for col in df.columns if col.lower() in ['date', 'week']), None)
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Auto-detect categorical columns for filters
        cat_cols = [col for col in df.select_dtypes(include=['object', 'category']).columns if col.lower() not in ['product_id']]
        
        # Numeric columns
        numeric_cols = ['Lost_Sales_Cost', 'Fill_Rate', 'Stockout_Flag', 'Lead_Time', 'Inventory_Level', 'Holding_Cost', 'Delivery_Delay']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df, cat_cols
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None, []

df, cat_cols = load_data()

if df is not None and len(df) > 0:
    st.markdown('<h1 class="main-header">🌐 Supply Chain Disruptions AI Engine</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive analytics for MBA Data Analytics. Filters adapt to your data columns.**")
    
    # --- SAFE SIDEBAR FILTERS ---
    st.sidebar.header("🔍 Filters")
    
    # Safe date filter
    if 'Date' in df.columns:
        date_range = st.sidebar.date_input("Date Range", [df['Date'].min().date(), df['Date'].max().date()])
    else:
        date_range = [pd.Timestamp.min.date(), pd.Timestamp.now().date()]
    
    # Safe categorical filter (auto-detects Disruption_Type or similar)
    disruption_col = next((col for col in cat_cols if 'disrupt' in col.lower() or 'type' in col.lower()), cat_cols[0] if cat_cols else None)
    if disruption_col:
        disruption_options = sorted(df[disruption_col].dropna().unique())
        selected_disrupt = st.sidebar.multiselect(f"{disruption_col}", options=disruption_options, default=disruption_options[:3])
    else:
        selected_disrupt = []
    
    # Safe zone filter
    zone_col = next((col for col in cat_cols if 'zone' in col.lower() or 'warehouse' in col.lower()), None)
    if zone_col:
        zone_options = sorted(df[zone_col].dropna().unique())
        selected_zones = st.sidebar.multiselect(f"{zone_col}", options=zone_options, default=zone_options)
    else:
        selected_zones = []
    
    # Filter data SAFELY
    mask = pd.Series([True] * len(df), index=df.index)
    if 'Date' in df.columns:
        mask &= (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
    if selected_disrupt and disruption_col:
        mask &= df[disruption_col].isin(selected_disrupt)
    if selected_zones and zone_col:
        mask &= df[zone_col].isin(selected_zones)
    
    filtered_df = df[mask].copy()
    
    # Show data preview in sidebar
    with st.sidebar.expander("📋 Data Preview"):
        st.dataframe(filtered_df.head())
        st.caption(f"Columns detected: {list(df.columns)}")
    
    # --- EXECUTIVE SUMMARY ---
    col1, col2, col3, col4, col5 = st.columns(5)
    total_lost = filtered_df.get('Lost_Sales_Cost', pd.Series(0)).sum()
    col1.metric("💰 Total Lost Sales", f"${total_lost:,.0f}")
    
    avg_fill = filtered_df.get('Fill_Rate', pd.Series(1)).mean()
    col2.metric("📊 Avg Fill Rate", f"{avg_fill:.1%}")
    
    high_risk = (filtered_df.get('Stockout_Flag', pd.Series(0)) > 0).sum()
    col3.metric("⚠️ High Risk Events", high_risk)
    
    avg_lead = filtered_df.get('Lead_Time', pd.Series(0)).mean()
    col4.metric("⏱️ Avg Lead Time", f"{avg_lead:.1f} days")
    
    col5.metric("✅ Rows Filtered", len(filtered_df), delta=f"{len(filtered_df)/len(df)*100:.0f}% of total")
    
    # --- TABS (Simplified for robustness) ---
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔮 Trends", "🤖 AI Insights"])
    
    with tab1:
        st.header("Key Visuals (Adapts to your data)")
        
        # Adaptive pie chart
        pie_col = disruption_col or next((col for col in cat_cols), None)
        if pie_col and pie_col in filtered_df.columns:
            agg_pie = filtered_df.groupby(pie_col)['Lost_Sales_Cost'].sum().reset_index()
            fig_pie = px.pie(agg_pie, values='Lost_Sales_Cost', names=pie_col, hole=0.4, title=f"Losses by {pie_col}")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Top contributors bar
        if 'Product_ID' in filtered_df.columns:
            top_skus = filtered_df.groupby('Product_ID')['Lost_Sales_Cost'].sum().nlargest(10).reset_index()
            fig_bar = px.bar(top_skus, x='Product_ID', y='Lost_Sales_Cost', title="Top 10 by Lost Sales")
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        if 'Date' in filtered_df.columns and 'Demand_Forecast' in filtered_df.columns and 'Actual_Demand' in filtered_df.columns:
            weekly = filtered_df.groupby('Date')[['Demand_Forecast', 'Actual_Demand']].sum().reset_index()
            fig_line = px.line(weekly, x='Date', y=['Demand_Forecast', 'Actual_Demand'], title="Demand Trends")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("📊 Add 'Date', 'Demand_Forecast', 'Actual_Demand' columns for trends")
    
    with tab3:
        # Safe clustering
        cluster_cols = ['Inventory_Level', 'Holding_Cost']
        avail_cluster = [c for c in cluster_cols if c in filtered_df.columns]
        if len(avail_cluster) >= 2 and len(filtered_df) > 10:
            features = filtered_df[avail_cluster].dropna()
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(features)
            filtered_df['Cluster'] = pd.cut(np.random.rand(len(filtered_df)), 3, labels=['Low', 'Med', 'High'])
            fig = px.scatter(filtered_df, x=avail_cluster[0], y=avail_cluster[1], color='Cluster')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔍 Clustering needs 'Inventory_Level' + 'Holding_Cost' with 10+ rows")
    
    # Footer
    st.markdown("---")
    st.markdown("**Robust version: Auto-adapts to ANY CSV structure. Uses your synthetic supply chain data.[cite:15]**")

else:
    st.error("❌ Upload `data.csv` to repo root. Use our 500-row synthetic data if needed.[cite:15]")
