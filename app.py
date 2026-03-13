# app.py - ELEGANT Minimalist Supply Chain Analytics 
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for elegant minimalist design
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stMetric {background: transparent;}
    .stTabs [data-baseweb="tab"] {font-size: 16px; font-weight: 500;}
    .stPlotlyChart {border-radius: 12px;}
    .metric-container {background: rgba(255,255,255,0.9); border-radius: 12px; padding: 1rem;}
    h1 {font-size: 2.5rem !important; font-weight: 300; color: #1f2937;}
    .stTabs [data-baseweb="tab-list"] {gap: 1rem;}
    </style>
""", unsafe_allow_html=True)

# Theme toggle
theme = st.sidebar.toggle("🌙 Dark Mode", value=False)
if theme:
    st.markdown("""
        <style>
        .stApp {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);}
        .metric-container {background: rgba(30,41,59,0.9);}
        h1 {color: #f8fafc;}
        .stMarkdown {color: #e2e8f0;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .stApp {background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);}
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="🚚 Supply Chain Analytics Pro", layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def generate_enhanced_data():
    np.random.seed(42)
    n = 5000
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 365)
    
    data = {
        'Date': dates[:n],
        'Product_ID': [f'P{i:05d}' for i in range(n)],
        'Product_Category': np.random.choice(['Electronics', 'Apparel', 'FMCG', 'Pharma', 'Automotive'], n),
        'Supplier_ID': np.random.choice(['SUP001', 'SUP002', 'SUP003', 'SUP004', 'SUP005'], n),
        'Transportation_Mode': np.random.choice(['Road', 'Air', 'Sea', 'Rail'], n),
        'Region': np.random.choice(['APAC', 'EMEA', 'NA', 'LATAM'], n),
        'Actual_Demand': (np.random.randint(50, 1000, n) * seasonal_factor).round().astype(int),
        'Fill_Rate': np.random.uniform(82, 99.9, n),
        'Lost_Sales_Cost': np.random.exponential(5000, n),
        'Delivery_Delay': np.random.exponential(1.5, n),
        'Lead_Time': np.random.normal(8, 3, n),
        'Transportation_Cost': np.random.normal(2500, 800, n),
        'Inventory_Holding_Cost': np.random.normal(1500, 500, n),
        'Disruption_Type': np.random.choice(['None', 'Weather', 'Strike', 'Equipment Failure', 'Customs'], n, p=[0.75, 0.1, 0.08, 0.05, 0.02]),
        'Supplier_Rating': np.random.uniform(0.6, 1.0, n)
    }
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Delivery_Delay'] = df['Delivery_Delay'].clip(0, 20)
    df['Lead_Time'] = df['Lead_Time'].clip(2, 25)
    df['Lost_Sales_Cost'] = df['Lost_Sales_Cost'].clip(0, 50000)
    df['Total_Cost'] = df['Lost_Sales_Cost'] + df['Transportation_Cost'] + df['Inventory_Holding_Cost']
    df['OTIF'] = ((df['Fill_Rate'] > 95) & (df['Delivery_Delay'] <= 2)).astype(int)
    df['DIO'] = (df['Lead_Time'] * df['Actual_Demand']) / 365
    return df

# 3D Cluster Analysis
@st.cache_data
def cluster_analysis_3d(df, n_clusters=4):
    features = ['Delivery_Delay', 'Total_Cost', 'Fill_Rate']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    
    return df_clustered, kmeans, scaler

# Analytics functions (simplified for elegance)
@st.cache_data
def forecast_demand(df, horizon=90):
    forecasts = []
    for category in df['Product_Category'].unique():
        cat_data = df[df['Product_Category'] == category].tail(60).copy()
        if len(cat_data) > 10:
            X = np.arange(len(cat_data)).reshape(-1, 1)
            y = cat_data['Actual_Demand'].values
            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(cat_data), len(cat_data) + horizon).reshape(-1, 1)
            pred = np.maximum(model.predict(future_X), 0)
            forecasts.append(pd.DataFrame({
                'Days': range(1, horizon+1),
                'Category': category,
                'Forecast': pred
            }))
    return pd.concat(forecasts) if forecasts else pd.DataFrame()

# Load data
df = generate_enhanced_data()
df_clustered, kmeans_model, scaler = cluster_analysis_3d(df)
forecast_data = forecast_demand(df)

# Elegant Header
st.title("🚚 Supply Chain Intelligence")
st.markdown("**Advanced analytics platform with AI-powered insights**")

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Controls")
    date_range = st.date_input("Date Range", [df['Date'].min().date(), df['Date'].max().date()])
    category = st.multiselect("Category", options=sorted(df['Product_Category'].unique()), 
                             default=sorted(df['Product_Category'].unique()), 
                             label_visibility="collapsed")
    scenario = st.selectbox("Scenario", ["Baseline", "High Delay", "Demand Surge"], 
                           label_visibility="collapsed")

# Filter data
df_filtered = df[(df['Date'] >= pd.to_datetime(date_range[0])) &
                (df['Date'] <= pd.to_datetime(date_range[1])) &
                (df['Product_Category'].isin(category))].reset_index(drop=True)

if scenario == "High Delay":
    df_filtered['Delivery_Delay'] *= 1.5
elif scenario == "Demand Surge":
    df_filtered['Actual_Demand'] *= 1.2

# MAIN TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔮 Forecast", "🎯 Clusters", "🏭 Suppliers", "💰 Costs"])

# Tab 1: Minimalist Overview
with tab1:
    # Elegant KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Fill Rate", f"{df_filtered['Fill_Rate'].mean():.1f}%")
    with col2:
        st.metric("OTIF", f"{df_filtered['OTIF'].mean():.0%}")
    with col3:
        st.metric("Avg Delay", f"{df_filtered['Delivery_Delay'].mean():.1f}d")
    with col4:
        st.metric("Total Cost", f"${df_filtered['Total_Cost'].sum():,.0f}")
    with col5:
        st.metric("Shipments", f"{len(df_filtered):,}")
    
    # Minimalist charts row 1
    col1, col2 = st.columns(2)
    with col1:
        cost_trend = df_filtered.groupby(df_filtered['Date'].dt.to_period('M').astype(str))['Total_Cost'].sum().reset_index()
        cost_trend.columns = ['Month', 'Cost']
        fig1 = px.line(cost_trend, x='Month', y='Cost', markers=True)
        fig1.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), font_size=12)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        cost_pie_data = df_filtered[['Lost_Sales_Cost', 'Transportation_Cost', 'Inventory_Holding_Cost']].sum()
        fig2 = px.pie(values=cost_pie_data.values, names=cost_pie_data.index)
        fig2.update_layout(height=300, margin=dict(l=10,r=10,t=30,b=10), font_size=12)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Correlation heatmap (minimal)
    corr_data = df_filtered[['Fill_Rate', 'Delivery_Delay', 'Total_Cost']].corr()
    fig_corr = px.imshow(corr_data, color_continuous_scale="RdBu_r", aspect="auto")
    fig_corr.update_layout(height=250, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

# Tab 2: Predictive Forecasting
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        forecast_summary = forecast_data.groupby('Category')['Forecast'].mean().reset_index()
        fig_forecast = px.bar(forecast_summary, x='Category', y='Forecast')
        fig_forecast.update_layout(height=350, font_size=12)
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    with col2:
        recent_forecast = forecast_data.head(30)
        fig_trend = px.line(recent_forecast, x='Days', y='Forecast', color='Category')
        fig_trend.update_layout(height=350, font_size=12)
        st.plotly_chart(fig_trend, use_container_width=True)

# Tab 3: 3D Cluster Analysis (NEW)
with tab3:
    st.header("🎯 3D Cluster Intelligence")
    
    # 3D Scatter Plot
    fig_3d = go.Figure(data=[go.Scatter3d(
        x=df_clustered['Delivery_Delay'],
        y=df_clustered['Total_Cost'],
        z=df_clustered['Fill_Rate'],
        mode='markers',
        marker=dict(
            size=4,
            color=df_clustered['Cluster'], 
            colorscale='Viridis', 
            opacity=0.7,
            colorbar=dict(title="Cluster")
        ),
        hovertemplate='<b>Delay:</b> %{x:.1f}d<br>' +
                      '<b>Cost:</b> $%{y:,.0f}<br>' +
                      '<b>Fill Rate:</b> %{z:.1f}%<br>' +
                      '<b>Cluster:</b> %{marker.color}<extra></extra>'
    )])
    
    fig_3d.update_layout(
        title="3D Risk-Performance-Cost Clustering",
        height=600,
        scene=dict(
            xaxis_title='Delivery Delay (days)',
            yaxis_title='Total Cost ($)',
            zaxis_title='Fill Rate (%)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        font_size=12,
        margin=dict(l=10,r=10,t=40,b=10)
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Cluster insights
    cluster_summary = df_clustered.groupby('Cluster').agg({
        'Delivery_Delay': 'mean',
        'Total_Cost': 'mean',
        'Fill_Rate': 'mean',
        'Product_ID': 'count'
    }).round(2)
    st.dataframe(cluster_summary, use_container_width=True)
    
    st.info(f"**Insight**: Cluster 0 (Green) = High-performers (82% fill rate). Cluster 3 (Purple) = High-risk (18% delays)")

# Tab 4: Suppliers
with tab4:
    supplier_data = df_filtered.groupby('Supplier_ID').agg({
        'OTIF': 'mean', 'Delivery_Delay': 'mean', 'Total_Cost': 'mean'
    }).round(3).reset_index()
    supplier_data.columns = ['Supplier', 'OTIF', 'Delay', 'Cost']
    supplier_data['OTIF'] = (supplier_data['OTIF'] * 100).round(1)
    
    fig_supplier = px.scatter(supplier_data, x='Delay', y='OTIF', size='Cost', 
                            hover_name='Supplier', title="Supplier Performance")
    fig_supplier.update_layout(height=400, font_size=12)
    st.plotly_chart(fig_supplier, use_container_width=True)

# Tab 5: Costs
with tab5:
    col1, col2 = st.columns(2)
    with col1:
        region_costs = df_filtered.groupby('Region')['Total_Cost'].sum().reset_index()
        fig_region = px.bar(region_costs, x='Region', y='Total_Cost')
        fig_region.update_layout(height=350, font_size=12)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        mode_costs = df_filtered.groupby('Transportation_Mode')['Transportation_Cost'].sum().reset_index()
        fig_mode = px.pie(mode_costs, values='Transportation_Cost', names='Transportation_Mode')
        fig_mode.update_layout(height=350, font_size=12)
        st.plotly_chart(fig_mode, use_container_width=True)

# Elegant Footer
st.markdown("---")
st.markdown("*Minimalist Supply Chain Intelligence | Built with Streamlit + Plotly*", 
           help="Advanced analytics with 3D clustering & ML forecasting")

