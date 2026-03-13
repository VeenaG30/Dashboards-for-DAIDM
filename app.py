# app.py - ULTIMATE Supply Chain Analytics Platform (ALL Analytics Types)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="🚚 Supply Chain Analytics Pro", layout="wide", initial_sidebar_state="expanded")

# Generate comprehensive sample data with seasonality
@st.cache_data
def generate_enhanced_data():
    np.random.seed(42)
    n = 8000
    dates = pd.date_range('2022-01-01', periods=n, freq='D')
    
    # Seasonal demand patterns
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 365)
    
    data = {
        'Date': dates,
        'Product_ID': [f'P{i:05d}' for i in range(n)],
        'Product_Category': np.random.choice(['Electronics', 'Apparel', 'FMCG', 'Pharma', 'Automotive'], n),
        'Supplier_ID': np.random.choice(['SUP001', 'SUP002', 'SUP003', 'SUP004', 'SUP005'], n),
        'Transportation_Mode': np.random.choice(['Road', 'Air', 'Sea', 'Rail'], n),
        'Region': np.random.choice(['APAC', 'EMEA', 'NA', 'LATAM'], n),
        'Actual_Demand': (np.random.randint(50, 1000, n) * seasonal_factor).round().astype(int),
        'Forecast_Demand': 0,  # Will be calculated
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
    df['Cost_Per_Unit'] = df['Total_Cost'] / (df['Actual_Demand'] + 1)
    
    # Generate forecast demand (simple trend + seasonality)
    for cat in df['Product_Category'].unique():
        mask = df['Product_Category'] == cat
        trend = np.arange(len(df[mask])) * 0.1
        df.loc[mask, 'Forecast_Demand'] = df.loc[mask, 'Actual_Demand'].rolling(30, min_periods=1).mean() + trend
    
    return df

# PREDICTIVE: Demand forecasting
@st.cache_data
def forecast_demand(df, horizon=90):
    forecasts = []
    for category in df['Product_Category'].unique():
        cat_data = df[df['Product_Category'] == category].copy()
        if len(cat_data) > 30:
            X = np.arange(len(cat_data)).reshape(-1, 1)
            y = cat_data['Actual_Demand'].values
            model = LinearRegression().fit(X, y)
            
            future_X = np.arange(len(cat_data), len(cat_data) + horizon).reshape(-1, 1)
            pred = model.predict(future_X)
            forecasts.append(pd.DataFrame({
                'Date': pd.date_range(start=cat_data['Date'].max() + timedelta(days=1), periods=horizon),
                'Product_Category': category,
                'Forecast_Demand': pred
            }))
    return pd.concat(forecasts) if forecasts else pd.DataFrame()

# DIAGNOSTIC: Root cause analysis
@st.cache_data
def root_cause_analysis(df):
    disruptions = df[df['Disruption_Type'] != 'None']
    if len(disruptions) == 0:
        return pd.DataFrame()
    
    causes = disruptions.groupby(['Disruption_Type', 'Transportation_Mode', 'Region']).agg({
        'Total_Cost': 'sum',
        'Delivery_Delay': 'mean',
        'Product_ID': 'count'
    }).round(2).reset_index()
    causes.columns = ['Disruption_Type', 'Transport_Mode', 'Region', 'Total_Cost', 'Avg_Delay', 'Frequency']
    return causes.sort_values('Total_Cost', ascending=False).head(10)

# PRESCRIPTIVE: Optimization recommendations
@st.cache_data
def prescriptive_analytics(df):
    # Optimal transport mode by region/category
    opt_transport = df.groupby(['Region', 'Product_Category', 'Transportation_Mode']).agg({
        'Total_Cost': 'mean',
        'Delivery_Delay': 'mean'
    }).reset_index()
    
    # Find best mode per region/category combo
    best_modes = []
    for region in opt_transport['Region'].unique():
        for category in opt_transport['Product_Category'].unique():
            subset = opt_transport[(opt_transport['Region'] == region) & 
                                 (opt_transport['Product_Category'] == category)]
            if not subset.empty:
                best_mode = subset.loc[subset['Total_Cost'].idxmin()]
                best_modes.append(best_mode)
    
    return pd.DataFrame(best_modes).sort_values('Total_Cost')

# Load data and analytics
df = generate_enhanced_data()
demand_forecast = forecast_demand(df)
root_causes = root_cause_analysis(df)
opt_recommendations = prescriptive_analytics(df)

# === ENHANCED SIDEBAR ===
st.sidebar.header("🔍 **Filters**")
date_range = st.sidebar.date_input("Date Range", 
                                  [df['Date'].min().date(), df['Date'].max().date()])
category = st.sidebar.multiselect("Category", options=sorted(df['Product_Category'].unique()),
                                 default=sorted(df['Product_Category'].unique()))
mode = st.sidebar.multiselect("Transport Mode", options=sorted(df['Transportation_Mode'].unique()),
                             default=sorted(df['Transportation_Mode'].unique()))
region = st.sidebar.multiselect("Region", options=sorted(df['Region'].unique()),
                               default=sorted(df['Region'].unique()))

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Scenarios")
scenario = st.sidebar.selectbox("Scenario", ["Baseline", "High Delay", "Demand Surge", "Cost Crisis"])

# Apply filters
df_filtered = df[
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Product_Category'].isin(category)) &
    (df['Transportation_Mode'].isin(mode)) &
    (df['Region'].isin(region))
].reset_index(drop=True)

df_scenario = df_filtered.copy()
if scenario == "High Delay":
    df_scenario['Delivery_Delay'] *= 2
elif scenario == "Demand Surge":
    df_scenario['Actual_Demand'] *= 1.3
elif scenario == "Cost Crisis":
    df_scenario['Total_Cost'] *= 1.5

# === COMPREHENSIVE DASHBOARD ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview", "🔍 EDA", "🚨 Diagnostics", "🔮 Predictive", 
    "💡 Prescriptive", "🏭 Suppliers", "📦 Inventory", "💰 Costs"
])

# Tab 1: Executive Overview
with tab1:
    st.subheader("**Executive KPIs**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("Fill Rate", f"{df_scenario['Fill_Rate'].mean():.1f}%")
    col2.metric("OTIF", f"{df_scenario['OTIF'].mean():.1%}")
    col3.metric("Total Cost", f"${df_scenario['Total_Cost'].sum():,.0f}")
    col4.metric("Avg Delay", f"{df_scenario['Delivery_Delay'].mean():.1f} days")
    col5.metric("DIO", f"{df_scenario['DIO'].mean():.0f} days")
    col6.metric("Shipments", f"{len(df_scenario):,}")

# Tab 2: EXPLORATORY DATA ANALYSIS (NEW)
with tab2:
    st.header("🔍 **Exploratory Data Analysis**")
    
    col1, col2 = st.columns(2)
    with col1:
        # Correlation heatmap
        corr_cols = ['Fill_Rate', 'Delivery_Delay', 'Total_Cost', 'OTIF', 'DIO']
        corr_matrix = df_scenario[corr_cols].corr()
        fig_heatmap = px.imshow(corr_matrix, title="Correlation Matrix", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Distribution of key metrics
        fig_dist = px.histogram(df_scenario, x='Delivery_Delay', marginal="box", 
                               title="Delay Distribution", nbins=50)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    col3, col4 = st.columns(2)
    with col3:
        # Demand seasonality
        monthly_demand = df_scenario.groupby(df_scenario['Date'].dt.month)['Actual_Demand'].mean().reset_index()
        fig_seasonal = px.line(monthly_demand, x='Date', y='Actual_Demand', 
                              title="Monthly Demand Seasonality")
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col4:
        # Cost vs Performance scatter
        fig_scatter = px.scatter(df_scenario.head(1000), x='Total_Cost', y='Fill_Rate', 
                               color='Transportation_Mode', title="Cost vs Performance")
        st.plotly_chart(fig_scatter, use_container_width=True)

# Tab 3: DIAGNOSTIC ANALYTICS (NEW)
with tab3:
    st.header("🚨 **Diagnostic Analytics - Root Cause Analysis**")
    
    if not root_causes.empty:
        st.dataframe(root_causes, use_container_width=True)
        
        fig_cause = px.treemap(root_causes, path=['Disruption_Type', 'Transport_Mode'], 
                              values='Total_Cost', title="Root Cause Impact")
        st.plotly_chart(fig_cause, use_container_width=True)
    else:
        st.info("No disruptions found in filtered data")

# Tab 4: PREDICTIVE ANALYTICS (NEW)
with tab4:
    st.header("🔮 **Predictive Analytics - Demand Forecasting**")
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_summary = demand_forecast.groupby('Product_Category')['Forecast_Demand'].mean().reset_index()
        fig_forecast = px.bar(forecast_summary, x='Product_Category', y='Forecast_Demand',
                            title="90-Day Demand Forecast by Category")
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    with col2:
        if not demand_forecast.empty:
            fig_forecast_trend = px.line(demand_forecast.head(30), x='Date', y='Forecast_Demand',
                                       color='Product_Category', title="Next 30 Days Forecast")
            st.plotly_chart(fig_forecast_trend, use_container_width=True)

# Tab 5: PRESCRIPTIVE ANALYTICS (NEW)
with tab5:
    st.header("💡 **Prescriptive Analytics - Optimization**")
    
    if not opt_recommendations.empty:
        st.dataframe(opt_recommendations[['Region', 'Product_Category', 'Transportation_Mode', 'Total_Cost']].head(10), use_container_width=True)
        
        fig_opt = px.scatter(opt_recommendations.head(20), x='Delivery_Delay', y='Total_Cost',
                           size='Total_Cost', color='Transportation_Mode',
                           hover_name='Region', title="Optimal Transport Recommendations")
        st.plotly_chart(fig_opt, use_container_width=True)
        
        st.success("💡 **Action**: Switch to recommended transport modes by region/product")

# Tab 6-8: Previous tabs (Suppliers, Inventory, Costs) - same as before
with tab6:
    st.subheader("🏭 **Supplier Performance**")
    # ... supplier code here (same as previous)

with tab7:
    st.subheader("📦 **Inventory Analytics**")
    # ... inventory code here

with tab8:
    st.subheader("💰 **Cost Analysis**")
    # ... cost code here

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🚚 Complete Supply Chain Analytics Platform**")
with col2:
    st.markdown("*EDA • Diagnostic • Predictive • Prescriptive*")
with col3:
    st.markdown(f"**{len(df_scenario):,} records | {scenario} scenario**")
