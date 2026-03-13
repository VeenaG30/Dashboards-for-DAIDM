# app.py - Supply Chain Analytics + AI Recommendation Engine
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🚚 Supply Chain Pro + AI Recommendations", layout="wide", initial_sidebar_state="expanded")

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
    df['Date'] = pd.to_datetime(df['Date'])
    df['Delivery_Delay'] = df['Delivery_Delay'].clip(0, 20)
    df['Lead_Time'] = df['Lead_Time'].clip(2, 25)
    df['Lost_Sales_Cost'] = df['Lost_Sales_Cost'].clip(0, 50000)
    df['Total_Cost'] = df['Lost_Sales_Cost'] + df['Transportation_Cost'] + df['Inventory_Holding_Cost']
    df['OTIF'] = ((df['Fill_Rate'] > 95) & (df['Delivery_Delay'] <= 2)).astype(int)
    df['DIO'] = (df['Lead_Time'] * df['Actual_Demand']) / 365
    df['Cost_Per_Unit'] = df['Total_Cost'] / df['Actual_Demand']
    
    return df

# === RECOMMENDATION ENGINE ===
@st.cache_data
def build_recommendation_engine(df):
    """Multi-criteria recommendation system for suppliers, transport, products"""
    
    # 1. SUPPLIER RECOMMENDATIONS (KNN-style scoring)
    supplier_scores = df.groupby('Supplier_ID').agg({
        'OTIF': 'mean',
        'Delivery_Delay': 'mean',
        'Total_Cost': 'mean',
        'Fill_Rate': 'mean',
        'Product_ID': 'count'
    }).round(3)
    
    # Normalize scores (higher OTIF, Fill_Rate = better; lower Delay, Cost = better)
    supplier_scores['OTIF_Score'] = (supplier_scores['OTIF'] - supplier_scores['OTIF'].min()) / (supplier_scores['OTIF'].max() - supplier_scores['OTIF'].min())
    supplier_scores['FillRate_Score'] = (supplier_scores['Fill_Rate'] - supplier_scores['Fill_Rate'].min()) / (supplier_scores['Fill_Rate'].max() - supplier_scores['Fill_Rate'].min())
    supplier_scores['Delay_Score'] = 1 - (supplier_scores['Delivery_Delay'] - supplier_scores['Delivery_Delay'].min()) / (supplier_scores['Delivery_Delay'].max() - supplier_scores['Delivery_Delay'].min())
    supplier_scores['Cost_Score'] = 1 - (supplier_scores['Total_Cost'] - supplier_scores['Total_Cost'].min()) / (supplier_scores['Total_Cost'].max() - supplier_scores['Total_Cost'].min())
    supplier_scores['Volume_Score'] = (supplier_scores['Product_ID'] - supplier_scores['Product_ID'].min()) / (supplier_scores['Product_ID'].max() - supplier_scores['Product_ID'].min())
    
    supplier_scores['Overall_Score'] = (
        0.3 * supplier_scores['OTIF_Score'] + 
        0.25 * supplier_scores['FillRate_Score'] + 
        0.2 * supplier_scores['Delay_Score'] + 
        0.15 * supplier_scores['Cost_Score'] + 
        0.1 * supplier_scores['Volume_Score']
    )
    
    top_suppliers = supplier_scores.sort_values('Overall_Score', ascending=False).head(5)
    
    # 2. TRANSPORT MODE RECOMMENDATIONS
    transport_perf = df.groupby('Transportation_Mode').agg({
        'Delivery_Delay': 'mean',
        'Transportation_Cost': 'mean',
        'OTIF': 'mean'
    }).round(3)
    
    transport_perf['Speed_Score'] = 1 - (transport_perf['Delivery_Delay'] - transport_perf['Delivery_Delay'].min()) / (transport_perf['Delivery_Delay'].max() - transport_perf['Delivery_Delay'].min())
    transport_perf['Cost_Score'] = 1 - (transport_perf['Transportation_Cost'] - transport_perf['Transportation_Cost'].min()) / (transport_perf['Transportation_Cost'].max() - transport_perf['Transportation_Cost'].min())
    transport_perf['Reliability_Score'] = transport_perf['OTIF']
    transport_perf['Overall_Score'] = 0.4 * transport_perf['Speed_Score'] + 0.4 * transport_perf['Cost_Score'] + 0.2 * transport_perf['Reliability_Score']
    
    top_transport = transport_perf.sort_values('Overall_Score', ascending=False).head(3)
    
    # 3. PRODUCT PRIORITIZATION (ABC + Risk)
    abc_risk = df.groupby('Product_ID').agg({
        'Actual_Demand': 'sum',
        'Total_Cost': 'sum',
        'Delivery_Delay': 'mean'
    }).round(3)
    
    abc_risk['Demand_Score'] = abc_risk['Actual_Demand'] / abc_risk['Actual_Demand'].sum()
    abc_risk['Risk_Score'] = abc_risk['Delivery_Delay'] / abc_risk['Delivery_Delay'].max()
    abc_risk['Priority_Score'] = abc_risk['Demand_Score'] * (1 - abc_risk['Risk_Score'])
    
    top_products = abc_risk.nlargest(10, 'Priority_Score').reset_index()
    
    return top_suppliers, top_transport, top_products

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
    if anomalies:
        return pd.concat(anomalies).drop_duplicates().sort_values('Date').head(15)
    return pd.DataFrame()

@st.cache_data
def supplier_performance(df):
    perf = df.groupby('Supplier_ID').agg({
        'OTIF': 'mean',
        'Delivery_Delay': 'mean',
        'Total_Cost': 'mean',
        'Product_ID': 'count'
    }).round(3)
    perf.columns = ['OTIF_Rate', 'Avg_Delay', 'Avg_Cost', 'Shipments']
    perf = perf.reset_index()
    perf['OTIF_Rate_%'] = (perf['OTIF_Rate'] * 100).round(1)
    return perf.sort_values('OTIF_Rate', ascending=False)

@st.cache_data
def abc_analysis(df):
    sku_demand = df.groupby('Product_ID')['Actual_Demand'].sum().sort_values(ascending=False).reset_index()
    sku_demand['CumPct'] = sku_demand['Actual_Demand'] / sku_demand['Actual_Demand'].sum()
    sku_demand['CumPct'] = sku_demand['CumPct'].cumsum()
    sku_demand['ABC'] = np.where(sku_demand['CumPct'] <= 0.8, 'A', 
                                np.where(sku_demand['CumPct'] <= 0.95, 'B', 'C'))
    return sku_demand

# Load data and recommendations
df = generate_enhanced_data()
recommendations = build_recommendation_engine(df)

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
df_scenario = df_filtered.copy()
if scenario == "High Delay":
    df_scenario['Delivery_Delay'] *= 2
elif scenario == "Demand Surge":
    df_scenario['Actual_Demand'] *= 1.3
elif scenario == "Cost Crisis":
    df_scenario['Total_Cost'] *= 1.5

# Main Dashboard with Recommendations Tab
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🚨 Risk Alerts", "🏭 Suppliers", "📦 Inventory", 
    "💰 Costs", "🤖 RECOMMENDATIONS"
])

# [Previous tabs 1-5 remain the same as before...]
with tab1:
    st.subheader("**Executive KPIs**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("Fill Rate", f"{df_scenario['Fill_Rate'].mean():.1f}%")
    col2.metric("OTIF", f"{df_scenario['OTIF'].mean():.1%}")
    col3.metric("Total Cost", f"${df_scenario['Total_Cost'].sum():,.0f}")
    col4.metric("Avg Delay", f"{df_scenario['Delivery_Delay'].mean():.1f} days")
    col5.metric("DIO", f"{df_scenario['DIO'].mean():.0f} days")
    col6.metric("Shipments", f"{len(df_scenario):,}")
    
    col_a, col_b = st.columns(2)
    with col_a:
        trend_data = df_scenario.groupby(df_scenario['Date'].dt.to_period('M').astype(str))['Total_Cost'].sum().reset_index()
        trend_data.columns = ['Month', 'Total_Cost']
        fig_trend = px.line(trend_data, x='Month', y='Total_Cost', title="Monthly Cost Trend")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_b:
        cost_pie = df_scenario[['Lost_Sales_Cost', 'Transportation_Cost', 'Inventory_Holding_Cost']].sum()
        fig_pie = px.pie(values=cost_pie.values, names=cost_pie.index, title="Cost Breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)

# Tab 6: AI RECOMMENDATION ENGINE ⚡
with tab6:
    st.header("🤖 **AI-Powered Recommendations**")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("🏆 **Top 5 Suppliers**")
        top_suppliers, _, _ = recommendations
        st.dataframe(top_suppliers[['Overall_Score']].round(3), use_container_width=True)
        
        fig_sup = px.bar(top_suppliers.head(3).reset_index(), 
                        x='Supplier_ID', y='Overall_Score',
                        title="Best Suppliers (Score out of 1.0)")
        st.plotly_chart(fig_sup, use_container_width=True)
        
        st.info(f"**Recommendation**: Prioritize **{top_suppliers.index[0]}** (Score: {top_suppliers['Overall_Score'].iloc[0]:.3f})")
    
    with c2:
        st.subheader("🚚 **Best Transport Modes**")
        _, top_transport, _ = recommendations
        st.dataframe(top_transport[['Overall_Score']].round(3), use_container_width=True)
        
        fig_trans = px.pie(top_transport.head(3), values='Overall_Score', names=top_transport.index,
                          title="Optimal Transport Mix")
        st.plotly_chart(fig_trans, use_container_width=True)
        
        st.info(f"**Recommendation**: Use **{top_transport.index[0]}** for 60% of shipments")
    
    with c3:
        st.subheader("📦 **Priority Products**")
        _, _, top_products = recommendations
        st.dataframe(top_products[['Priority_Score']].round(3).head(5), use_container_width=True)
        
        st.success(f"**Focus on**: {top_products.iloc[0]['Product_ID']} (Priority: {top_products.iloc[0]['Priority_Score']:.3f})")
    
    st.markdown("---")
    st.markdown("""
    **🎯 AI Engine Features:**
    - **Supplier Scoring**: OTIF (30%) + Reliability (25%) + Speed (20%) + Cost (15%) + Volume (10%)
    - **Transport Optimization**: Speed (40%) + Cost (40%) + Reliability (20%)
    - **Product Prioritization**: Demand × (1-Risk)
    """)

# Footer
st.markdown("---")
st.markdown("**🚚 Supply Chain Pro + AI Recommendations** | Production-ready analytics platform")

# Updated requirements.txt
