import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Page Configuration
st.set_page_config(page_title="Supply Chain AI Engine", layout="wide", page_icon="🌐")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Ensure this matches your GitHub filename exactly
    filename = "Supply Chain Disruptions Inventory.csv"
    try:
        # Read with semicolon separator
        df = pd.read_csv(filename, sep=';')
        
        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()
        
        # Convert Date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

df = load_data()

# --- APP UI ---
st.title("🌐 Supply Chain Disruptions AI Engine")
st.markdown("**MBA Data Analytics Dashboard**")

if df.empty:
    st.warning("No data found. Please check if 'Supply Chain Disruptions Inventory.csv' exists in the root folder.")
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filters")

# Date Filter
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

# Dynamic Multiselects
disruption_types = st.sidebar.multiselect("Disruption Type", df['Disruption_Type'].unique())
zones = st.sidebar.multiselect("Warehouse Zone", df['Warehouse_Zone'].unique())

# --- APPLY FILTERS ---
mask = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
if disruption_types:
    mask &= df['Disruption_Type'].isin(disruption_types)
if zones:
    mask &= df['Warehouse_Zone'].isin(zones)

filtered_df = df[mask].copy()

# --- KPI METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Lost Sales", f"${filtered_df['Lost_Sales_Cost'].sum():,.0f}")
col2.metric("📊 Avg Fill Rate", f"{filtered_df['Fill_Rate'].mean():.1%}")
col3.metric("⚠️ Stockouts", int(filtered_df['Stockout_Flag'].sum()))
col4.metric("⏱️ Avg Lead Time", f"{filtered_df['Lead_Time'].mean():.1f} days")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Trends", "📋 Data Explorer"])

with tab1:
    st.subheader("Financial Impact by Disruption")
    fig = px.pie(filtered_df, values='Lost_Sales_Cost', names='Disruption_Type', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Demand Forecast vs Actual")
    # Grouping by date for the line chart
    trend_data = filtered_df.groupby('Date')[['Demand_Forecast', 'Actual_Demand']].sum().reset_index()
    fig2 = px.line(trend_data, x='Date', y=['Demand_Forecast', 'Actual_Demand'], markers=True)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Raw Data View")
    st.dataframe(filtered_df)
    
    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Filtered Data", csv, "filtered_data.csv", "text/csv")
