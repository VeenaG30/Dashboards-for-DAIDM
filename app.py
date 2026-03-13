import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
import plotly.io as pio

st.set_page_config(page_title="Supply Chain AI Engine", layout="wide", page_icon="🌐")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv", sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv("data.csv", sep=',')
        df.columns = df.columns.str.strip()
        
        # Safe date parsing
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'week' in col.lower()]
        if date_cols:
            df['Date'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        
        # Safe numeric conversion
        for col in df.columns:
            if col not in ['Product_ID', 'Disruption_Type', 'Warehouse_Zone']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except:
        return pd.DataFrame()

df = load_data()

st.markdown("# 🌐 Supply Chain Disruptions AI Engine")
st.markdown("**MBA Data Analytics Dashboard - Robust Edition**")

if len(df) == 0:
    st.error("❌ No data.csv found. Upload to repo root.")
    st.info("Use our synthetic data from earlier chat!")
    st.stop()

# Sidebar - ULTRA SAFE
st.sidebar.header("🔍 Filters")
try:
    if 'Date' in df.columns:
        min_date, max_date = df['Date'].min(), df['Date'].max()
        date_range = st.sidebar.date_input("Date", [min_date.date(), max_date.date()])
    else:
        date_range = [pd.Timestamp.now().date(), pd.Timestamp.now().date()]
except:
    date_range = [pd.Timestamp.now().date(), pd.Timestamp.now().date()]

# SAFE FILTERING
mask = pd.Series([True] * len(df), index=df.index)
if 'Date' in df.columns:
    mask &= (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])

filtered_df = df[mask].copy()

# SAFE KPIs
col1, col2, col3, col4 = st.columns(4)
try:
    col1.metric("💰 Lost Sales", f"${filtered_df.get('Lost_Sales_Cost', 0).sum():,.0f}")
    col2.metric("📊 Fill Rate", f"{filtered_df.get('Fill_Rate', 1).mean():.1%}")
    col3.metric("⚠️ Stockouts", int(filtered_df.get('Stockout_Flag', 0).sum()))
    col4.metric("⏱️ Lead Time", f"{filtered_df.get('Lead_Time', 0).mean():.1f} days")
except:
    col1.metric("💰 Lost Sales", "$0")
    col2.metric("📊 Fill Rate", "100%")
    col3.metric("⚠️ Stockouts", "0")
    col4.metric("⏱️ Lead Time", "0 days")

# TABS - CRASH-PROOF
tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Trends", "🎯 Insights"])

with tab1:
    # Safe pie chart
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0 and 'Lost_Sales_Cost' in df.columns:
        pie_data = filtered_df.groupby(cat_cols[0])['Lost_Sales_Cost'].sum().reset_index()
        fig = px.pie(pie_data, values='Lost_Sales_Cost', names=cat_cols[0], hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("➕ Add categorical column + Lost_Sales_Cost for pie chart")

with tab2:
    if all(col in filtered_df.columns for col in ['Date', 'Demand_Forecast', 'Actual_Demand']):
        weekly = filtered_df.groupby('Date')[['Demand_Forecast', 'Actual_Demand']].sum().reset_index()
        fig = px.line(weekly,
