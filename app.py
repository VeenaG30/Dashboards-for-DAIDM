# app.py - Enhanced Supply Chain Analytics Dashboard
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="🚚 Supply Chain Analytics Pro", layout="wide", initial_sidebar_state="expanded")

# Generate comprehensive sample data
@st.cache_data
def generate_enhanced_data():
    np.random.seed(42)
    n = 5000
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    
    data = {
        'Date': np.random.choice(dates, n),
        'Product_ID': [f'P{i:05d}' for i in range(1, n+1)],
        'Product_Category': np.random.choice(['Electronics', 'Apparel', 'FMCG', 'Pharma', 'Automotive'], n),
        'Supplier_ID': np.random.choice(['SUP001', 'SUP002', 'SUP003', 'SUP004', 'SUP005'], n),
        'Transportation_Mode': np.random.choice(['Road', 'Air', 'Sea', 'Rail'], n),
        'Region': np.random.choice(['APAC', 'EMEA', 'NA', 'LATAM'], n),
        'Actual_Demand': np.random.randint(50, 1000, n),
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
    df['Delivery_Delay'] = df['Delivery_Delay'].clip(0, 20)
    df['Lead_Time'] = df['Lead_Time'].clip(2, 25)
    df['Lost_Sales_Cost'] = df['Lost_Sales_Cost'].clip(0, 50000)
    df['Total_Cost'] = df['Lost_Sales_Cost'] + df['Transportation_Cost'] + df['Inventory_Holding_Cost']
    df['OTIF'] = ((df['Fill_Rate'] > 95) & (df['Delivery_Delay'] <= 2)).astype(int)
    df['DIO'] = (df['Lead_Time'] * df['Actual_Demand']) / 365
    
    return df

# Advanced analytics functions
@st.cache_data
def detect_anomalies(df):
    anomalies = []
    for col in ['Delivery_Delay', 'Lost_Sales_Cost', 'Total_Cost']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] > upper) & (df['Disruption_Type'] != 'None')]
        anomalies.append(outliers)
    return pd.concat(anomalies).drop_duplicates().sort_values('Date').head(15)

@st.cache_data
def supplier_performance(df):
    perf = df.groupby('Supplier_ID').agg({
        'OTIF': 'mean',
        'Delivery_Delay': 'mean',
        'Total_Cost': 'mean',
        'Product_ID': 'count'
    }).round(3)
    perf.columns = ['OTIF_Rate', 'Avg_Delay', 'Avg_Cost', 'Shipments']
    perf['OTIF_Rate_%'] = (perf['OTIF_Rate'] * 100).round(1)
    return perf.sort_values('OTIF_Rate', ascending=False)

@st.cache_data
def abc_analysis(df):
    sku_demand = df.groupby('Product_ID')['Actual_Demand'].sum().sort_values(ascending=False)
    sku_demand = sku_demand / sku_demand.sum()
    sku_demand_cum = sku_demand.cumsum()
    sku_demand['ABC'] = np.where(sku_demand_cum <= 0.8, 'A', 
                                np.where(sku_demand_cum <= 0.95, 'B', 'C'))
    return sku_demand.reset_index()

# Load data
df = generate_enhanced_data()

# Enhanced Sidebar
st.sidebar.header("🔍 **Global Filters**")
date_range = st.sidebar.date_input("Date Range", 
                                  [df['Date'].min().date(), df['Date'].max().date()],
                                  format="YYYY-MM-DD")
category = st.sidebar.multiselect("Category", options=df['Product_Category'].unique(),
                                 default=df['Product_Category'].unique())
mode = st.sidebar.multiselect("Transport Mode", options=df['Transportation_Mode'].unique(),
                             default=df['Transportation_Mode'].unique())
region = st.sidebar.multiselect("Region", options=df['Region'].unique(),
                               default=df['Region'].unique())

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

# Apply scenario
if scenario == "High Delay":
    df_filtered['Delivery_Delay'] *= 2
elif scenario == "Demand Surge":
    df_filtered['Actual_Demand'] *= 1.3
elif scenario == "Cost Crisis":
    df_filtered['Total_Cost'] *= 1.5

# Main Dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🚨 Risk Alerts", "🏭 Supplier Perf", "📦 Inventory", "💰 Cost Analysis"])

# Tab 1: Executive Overview
with tab1:
    st.subheader("**Executive KPIs**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("Fill Rate", f"{df_filtered['Fill_Rate'].mean():.1f}%", 
                delta=f"{df_filtered['Fill_Rate'].std():.1f}% σ")
    col2.metric("OTIF", f"{df_filtered['OTIF'].mean():.1%}", 
                delta=f"+{((df_filtered['OTIF'].mean()*100)-95):+.1f}% vs target")
    col3.metric("Total Cost", f"${df_filtered['Total_Cost'].sum():,.0f}", 
                f"${df_filtered['Total_Cost'].mean():,.0f} avg")
    col4.metric("Avg Delay", f"{df_filtered['Delivery_Delay'].mean():.1f} days")
    col5.metric("DIO", f"{df_filtered['DIO'].mean():.0f} days")
    col6.metric("Shipments", f"{len(df_filtered):,}")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        trend_data = df_filtered.groupby(df_filtered['Date'].dt.to_period('M').astype(str))['Total_Cost'].sum()
        fig_trend = px.line(trend_data.reset_index(), x='Date', y='Total_Cost',
                           title="Monthly Cost Trend", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_b:
        cost_pie = df_filtered[['Lost_Sales_Cost', 'Transportation_Cost', 'Inventory_Holding_Cost']].sum()
        fig_pie = px.pie(values=cost_pie.values, names=cost_pie.index,
                        title="Cost Breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)

# Tab 2: Risk & Anomalies
with tab2:
    st.subheader("🚨 **High-Risk Events & Anomalies**")
    anomalies = detect_anomalies(df_filtered)
    
    if not anomalies.empty:
        col1, col2 = st.columns([3,1])
        with col1:
            st.dataframe(anomalies[['Date', 'Product_ID', 'Supplier_ID', 'Delivery_Delay', 
                                  'Total_Cost', 'Disruption_Type']], use_container_width=True)
        with col2:
            st.metric("🚨 Anomalies", len(anomalies), delta=f"{len(anomalies)/len(df_filtered)*100:.1f}% of shipments")
    else:
        st.success("✅ No significant anomalies detected")

# Tab 3: Supplier Performance
with tab3:
    st.subheader("🏭 **Supplier Performance Ranking**")
    supplier_perf = supplier_performance(df_filtered)
    
    st.dataframe(supplier_perf, use_container_width=True)
    
    fig_supplier = px.scatter(supplier_perf.reset_index(), 
                            x='Avg_Delay', y='OTIF_Rate_%', size='Shipments', color='Avg_Cost',
                            hover_name='Supplier_ID', title="Supplier Performance Matrix")
    st.plotly_chart(fig_supplier, use_container_width=True)

# Tab 4: Inventory & ABC Analysis
with tab4:
    st.subheader("📦 **Inventory Analytics**")
    
    col1, col2 = st.columns(2)
    with col1:
        abc = abc_analysis(df_filtered)
        fig_abc = px.bar(abc.head(15), x='Product_ID', y='Actual_Demand',
                        title="Top 15 SKUs - ABC Analysis", color='ABC')
        st.plotly_chart(fig_abc, use_container_width=True)
    
    with col2:
        dio_by_category = df_filtered.groupby('Product_Category')['DIO'].mean().sort_values()
        fig_dio = px.bar(dio_by_category.reset_index(), x='Product_Category', y='DIO',
                        title="Days Inventory Outstanding by Category")
        st.plotly_chart(fig_dio, use_container_width=True)

# Tab 5: Cost Analysis
with tab5:
    st.subheader("💰 **Cost Structure Analysis**")
    
    col1, col2 = st.columns(2)
    with col1:
        cost_by_region = df_filtered.groupby('Region')['Total_Cost'].sum().sort_values()
        fig_region = px.bar(cost_by_region.reset_index(), x='Region', y='Total_Cost',
                          title="Total Cost by Region", color='Total_Cost')
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        cost_by_mode = df_filtered.groupby('Transportation_Mode')['Transportation_Cost'].sum()
        fig_mode = px.pie(values=cost_by_mode.values, names=cost_by_mode.index,
                         title="Transport Cost by Mode")
        st.plotly_chart(fig_mode, use_container_width=True)

# Footer
st.markdown("---")
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🚚 Supply Chain Pro Analytics**")
    with col2:
        st.markdown("*Enhanced with time intelligence, supplier tracking, ABC analysis*")
    with col3:
        st.markdown(f"**Data: {len(df_filtered):,} records | {scenario} scenario**")

st.markdown("---")
