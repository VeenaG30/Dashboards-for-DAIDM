import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# Configure page
st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

# Generate sample data (for demo purposes - replace with your data.csv)
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    
    data = {
        'Date': np.random.choice(dates, n),
        'Product_ID': [f'P{i:04d}' for i in range(1, n+1)],
        'Product_Category': np.random.choice(['Electronics', 'Apparel', 'FMCG', 'Pharma'], n),
        'Transportation_Mode': np.random.choice(['Road', 'Air', 'Sea', 'Rail'], n),
        'Actual_Demand': np.random.randint(50, 500, n),
        'Fill_Rate': np.random.uniform(85, 99.9, n),
        'Lost_Sales_Cost': np.random.randint(1000, 50000, n),
        'Delivery_Delay': np.random.exponential(2, n),  # Most delays small
        'Lead_Time': np.random.normal(7, 2, n),
        'Disruption_Type': np.random.choice(['None', 'Weather', 'Strike', 'Equipment Failure'], n, p=[0.8, 0.1, 0.06, 0.04])
    }
    df = pd.DataFrame(data)
    df['Delivery_Delay'] = df['Delivery_Delay'].clip(0, 15)
    df['Lead_Time'] = df['Lead_Time'].clip(1, 20)
    return df

# Anomaly detection (simple statistical method)
@st.cache_data
def detect_anomalies(df):
    anomalies = []
    for col in ['Delivery_Delay', 'Lost_Sales_Cost']:
        mean = df[col].mean()
        std = df[col].std()
        threshold = 3  # 3-sigma rule
        high_risk = df[(df[col] > mean + threshold * std) & (df['Disruption_Type'] != 'None')]
        anomalies.append(high_risk)
    return pd.concat(anomalies).drop_duplicates().sort_values('Date').head(10)

# Simple association rules (delay patterns by transport mode)
@st.cache_data
def get_association_rules(df):
    disruptions = df[df['Disruption_Type'] != 'None']
    rules = disruptions.groupby(['Transportation_Mode', 'Disruption_Type']).agg({
        'Product_ID': 'count',
        'Delivery_Delay': 'mean'
    }).round(2)
    rules.columns = ['Frequency', 'Avg_Delay_Days']
    rules = rules.reset_index()
    rules = rules[rules['Frequency'] > 1].sort_values('Frequency', ascending=False)
    return rules.head(10)

# Load/generate data
df = generate_sample_data()

# Sidebar filters
st.sidebar.header("🔍 Filters")
category = st.sidebar.multiselect("Product Category", 
                                 options=df['Product_Category'].unique(),
                                 default=df['Product_Category'].unique())
mode = st.sidebar.multiselect("Transport Mode", 
                             options=df['Transportation_Mode'].unique(),
                             default=df['Transportation_Mode'].unique())

# Apply filters
if not category:
    category = df['Product_Category'].unique()
if not mode:
    mode = df['Transportation_Mode'].unique()

df_filtered = df[df['Product_Category'].isin(category) & 
                 df['Transportation_Mode'].isin(mode)].reset_index(drop=True)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚨 Anomaly & Risk", "🔗 Patterns", "🎯 Forecasting"])

# Tab 1: Overview
with tab1:
    st.subheader("Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Fill Rate", f"{df_filtered['Fill_Rate'].mean():.1f}%", 
                delta=f"{df_filtered['Fill_Rate'].std():.1f}% σ")
    col2.metric("Total Lost Sales", f"${df_filtered['Lost_Sales_Cost'].sum():,.0f}", 
                delta=f"${df_filtered['Lost_Sales_Cost'].sum()/len(df_filtered):,.0f} avg")
    col3.metric("Avg Delivery Delay", f"{df_filtered['Delivery_Delay'].mean():.1f} days",
                delta=f"{df_filtered['Delivery_Delay'].std():.1f} σ")
    col4.metric("Avg Lead Time", f"{df_filtered['Lead_Time'].mean():.1f} days")
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        demand_by_category = df_filtered.groupby('Product_Category')['Actual_Demand'].sum().reset_index()
        fig1 = px.bar(demand_by_category, x='Product_Category', y='Actual_Demand',
                      title="Demand by Category", color='Actual_Demand')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_b:
        delay_by_mode = df_filtered.groupby('Transportation_Mode')['Delivery_Delay'].mean().reset_index()
        fig2 = px.bar(delay_by_mode, x='Transportation_Mode', y='Delivery_Delay',
                      title="Avg Delay by Transport Mode", color='Delivery_Delay')
        st.plotly_chart(fig2, use_container_width=True)

# Tab 2: Anomaly Detection
with tab2:
    st.subheader("🚨 High-Risk Disruption Events")
    anomalies = detect_anomalies(df_filtered)
    
    if not anomalies.empty:
        st.dataframe(anomalies[['Date', 'Product_ID', 'Delivery_Delay', 'Disruption_Type', 'Lost_Sales_Cost']],
                    use_container_width=True)
        st.metric("Anomalies Detected", len(anomalies))
    else:
        st.info("✅ No significant anomalies detected in filtered data")

# Tab 3: Association Rules
with tab3:
    st.subheader("🔗 Disruption Patterns")
    rules = get_association_rules(df_filtered)
    
    if not rules.empty:
        st.dataframe(rules, use_container_width=True)
        
        fig3 = px.scatter(rules, x='Avg_Delay_Days', y='Frequency', 
                         size='Frequency', color='Transportation_Mode',
                         hover_name='Disruption_Type',
                         title="Disruption Patterns: Frequency vs Impact")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No significant patterns found in filtered data")

# Tab 4: Forecasting/Scenario
with tab4:
    st.subheader("🎯 Scenario Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        delay_increase = st.slider("Lead Time Increase (Days)", 0, 15, 0)
        demand_drop = st.slider("Demand Drop (%)", 0, 50, 0)
    
    # Simulate impact
    simulation_df = df_filtered.copy()
    simulation_df['Sim_Lead_Time'] = simulation_df['Lead_Time'] + delay_increase
    simulation_df['Sim_Demand'] = simulation_df['Actual_Demand'] * (1 - demand_drop/100)
    
    orig_lead_time = df_filtered['Lead_Time'].mean()
    orig_demand = df_filtered['Actual_Demand'].sum()
    
    st.metric("Original Lead Time", f"{orig_lead_time:.1f} days")
    st.metric("New Lead Time", f"{simulation_df['Sim_Lead_Time'].mean():.1f} days", 
              delta=f"+{delay_increase} days")
    st.metric("Original Demand", f"{orig_demand:,.0f} units")
    st.metric("New Demand", f"{simulation_df['Sim_Demand'].sum():,.0f} units", 
              delta=f"-{demand_drop}%")

# Footer
st.markdown("---")
st.markdown("**Supply Chain Analytics Dashboard** | Built with Streamlit & Plotly")

# Instructions for deployment
with st.expander("📋 Deployment Instructions"):
    st.markdown("""
    1. Save as `app.py`
    2. Create `requirements.txt`:
    ```
    streamlit
    pandas
    plotly
    numpy
    ```
    3. Push to GitHub repo
    4. Deploy on Streamlit Cloud ✅
    """)
