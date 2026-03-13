import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Supply Chain AI Engine", layout="wide", page_icon="🌐")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv", sep=';')
        df.columns = [col.strip() for col in df.columns]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except:
        return pd.DataFrame()

df = load_data()

st.title("🌐 Supply Chain Disruptions AI Engine")
st.markdown("**MBA Data Analytics Dashboard**")

if len(df) == 0:
    st.error("❌ Put data.csv in repo root (semicolon separated)")
    st.stop()

# SAFE FILTERS
st.sidebar.header("🔍 Filters")
date_range = st.sidebar.date_input("Date", [df['Date'].min().date(), df['Date'].max().date()])

# Safe multiselects with error handling
try:
    disruption_types = st.sidebar.multiselect("Disruption Type", 
        df['Disruption_Type'].dropna().unique().tolist()[:5])
except:
    disruption_types = []

try:
    zones = st.sidebar.multiselect("Warehouse Zone", 
        df['Warehouse_Zone'].dropna().unique().tolist())
except:
    zones = []

# SAFE FILTER
mask = pd.Series([True] * len(df), index=df.index)
mask &= (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
if disruption_types:
    mask &= df['Disruption_Type'].isin(disruption_types)
if zones:
    mask &= df['Warehouse_Zone'].isin(zones)

filtered_df = df[mask].copy()

# SAFE KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Lost Sales", f"${filtered_df['Lost_Sales_Cost'].sum():,.0f}")
col2.metric("📊 Fill Rate", f"{filtered_df['Fill_Rate'].mean():.1%}")
col3.metric("⚠️ Stockouts", int(filtered_df['Stockout_Flag'].sum()))
col4.metric("⏱️ Lead Time", f"{filtered_df['Lead_Time'].mean():.1f} days")

# SAFE TABS
tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Trends", "📋 Data"])

with tab1:
    # Safe pie chart
    if 'Disruption_Type' in filtered_df.columns and 'Lost_Sales_Cost' in filtered_df.columns:
        pie_data = filtered_df.groupby('Disruption_Type')['Lost_Sales_Cost'].sum().reset_index()
        fig = px.pie(pie_data, values='Lost_Sales_Cost', names='Disruption_Type', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    if all(col in filtered_df.columns for col in ['Date', 'Demand_Forecast', 'Actual_Demand']):
        weekly = filtered_df.groupby('Date')[['Demand_Forecast', 'Actual_Demand']].sum().reset_index()
        fig = px.line(weekly, x='Date', y=['Demand_Forecast', 'Actual_Demand'])
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Your Data")
    st.dataframe(filtered_df[['Date', 'Product_ID', 'Disruption_Type', 'Warehouse_Zone', 'Lost_Sales_Cost']].head(10))
    st.caption(f"Total rows: {len(filtered_df)} | Columns: {list(df.columns)}")

