# app.py - COMPLETE Supply Chain Analytics + AI Recommendation Engine (FIXED)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚚 Supply Chain Pro + AI Recommendations", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

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
    df['Cost_Per_Unit'] = df['Total_Cost'] / (df['Actual_Demand'] + 1)  # +1 to avoid division by zero
    
    return df

# AI Recommendation Engine
@st.cache_data
def build_recommendation_engine(df):
    """Multi-criteria recommendation system"""
    
    # Supplier recommendations
    supplier_scores = df.groupby('Supplier_ID').agg({
        'OTIF': 'mean',
        'Delivery_Delay': 'mean',
        'Total_Cost': 'mean',
        'Fill_Rate': 'mean',
        'Product_ID': 'count'
    }).round(3)
    
    # Normalize and score suppliers
    supplier_scores['OTIF_Score'] = (supplier_scores['OTIF'] - supplier_scores['OTIF'].min()) / (supplier_scores['OTIF'].max() - supplier_scores['OTIF'].min() + 0.001)
    supplier_scores['FillRate_Score'] = (supplier_scores['Fill_Rate'] - supplier_scores['Fill_Rate'].min()) / (supplier_scores['Fill_Rate'].max() - supplier_scores['Fill_Rate'].min() + 0.001)
    supplier_scores['Delay_Score'] = 1 - (supplier_scores['Delivery_Delay'] - supplier_scores['Delivery_Delay'].min()) / (supplier_scores['Delivery_Delay'].max() - supplier_scores['Delivery_Delay'].min() + 0.001)
    supplier_scores['Cost_Score'] = 1 - (supplier_scores['Total_Cost'] - supplier_scores['Total_Cost'].min()) / (supplier_scores['Total_Cost'].max() - supplier_scores['Total_Cost'].min() + 0.001)
    supplier_scores['Volume_Score'] = (supplier_scores['Product_ID'] - supplier_scores['Product_ID'].min()) / (supplier_scores['Product_ID'].max() - supplier_scores['Product_ID'].min() + 0.001)
    
    supplier_scores['Overall_Score'] = (
        0.3 * supplier_scores['OTIF_Score'] + 
        0.25 * supplier_scores['FillRate_Score'] + 
        0.2 * supplier_scores['Delay_Score'] + 
        0.15 * supplier_scores['Cost_Score'] + 
        0.1 * supplier_scores['Volume_Score']
    )
    top_suppliers = supplier_scores.nlargest(5, 'Overall_Score').reset_index()
    
    # Transport recommendations
    transport_perf = df.groupby('Transportation_Mode').agg({
        'Delivery_Delay': 'mean',
        'Transportation_Cost': 'mean',
        'OTIF': 'mean'
    }).round(3).reset_index()
    
    transport_perf['Speed_Score'] = 1 - (transport_perf['Delivery_Delay'] - transport_perf['Delivery_Delay'].min()) / (transport_perf['Delivery_Delay'].max() - transport_perf['Delivery_Delay'].min() + 0.001)
    transport_perf['Cost_Score'] = 1 - (transport_perf['Transportation_Cost'] - transport_perf['Transportation_Cost'].min()) / (transport_perf['Transportation_Cost'].max() - transport_perf['Transportation_Cost'].min() + 0.001)
    transport_perf['Reliability_Score'] = transport_perf['OTIF']
    transport_perf['Overall_Score'] = 0.4 * transport_perf['Speed_Score'] + 0.4 * transport_perf['Cost_Score'] + 0.2 * transport_perf['Reliability_Score']
    top_transport = transport_perf.nlargest(3, 'Overall_Score')
    
    # Product prioritization
    abc_risk = df.groupby('Product_ID').agg({
        'Actual_Demand': 'sum',
        'Total_Cost': 'sum',
        'Delivery_Delay': 'mean'
    }).round(3).reset_index()
    
    abc_risk['Demand_Score'] = abc_risk['Actual_Demand'] / abc_risk['Actual_Demand'].sum()
    abc_risk['Risk_Score'] = abc_risk['Delivery_Delay'] / abc_risk['Delivery_Delay'].max()
    abc_risk['Priority_Score'] = abc_risk['Demand_Score'] * (1 - abc_risk['Risk_Score'])
    top_products = abc_risk.nlargest(10, 'Priority_Score')
    
    return top_suppliers, top_transport, top_products

# Analytics functions (FIXED)
@st.cache_data
def detect_anomalies(df):
    anomalies = []
    for col in ['Delivery_Delay', 'Lost_Sales_Cost', 'Total_Cost']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] > (Q3 + 1.5 * IQR)) & (df['Disruption_Type'] != 'None')]
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
    }).round(3).reset_index()
    perf.columns = ['Supplier_ID', 'OTIF_Rate', 'Avg_Delay', 'Avg_Cost', 'Shipments']
    perf['OTIF_Rate_%'] = (perf['OTIF_Rate'] * 100).round(1)
    return perf.sort_values('OTIF_Rate', ascending=False)

@st.cache_data
def abc_analysis(df):
    sku_demand = df.groupby('Product_ID')['Actual_Demand'].sum().sort_values(ascending=False).reset_index()
    sku_demand.columns = ['Product_ID', 'Actual_Demand']
    sku_demand['CumPct'] = sku_demand['Actual_Demand'] / sku_demand['Actual_Demand'].sum()
    sku_demand['CumPct'] = sku_demand['CumPct'].cumsum()
    sku_demand['ABC'] = np.where(sku_demand['CumPct'] <= 0.8, 'A', 
                                np.where(sku_demand['CumPct'] <= 0.95, 'B', 'C'))
    return sku_demand

# Load data
df = generate_enhanced_data()
recommendations = build_recommendation_engine(df)

# === SIDEBAR ===
st.sidebar.header("🔍 **Global Filters**")
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

# Apply scenario
df_scenario = df_filtered.copy()
if scenario == "High Delay":
    df_scenario['Delivery_Delay'] *= 2
elif scenario == "Demand Surge":
    df_scenario['Actual_Demand'] *= 1.3
elif scenario == "Cost Crisis":
    df_scenario['Total_Cost'] *= 1.5

# === MAIN DASHBOARD ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🚨 Risk Alerts", "🏭 Suppliers", "📦 Inventory", 
    "💰 Costs", "🤖 RECOMMENDATIONS"
])

# Tab 1: Overview
with tab1:
    st.subheader("**Executive KPIs**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("Fill Rate", f"{df_scenario['Fill_Rate'].mean():.1f}%", f"{df_scenario['Fill_Rate'].std():.1f}% σ")
    col2.metric("OTIF", f"{df_scenario['OTIF'].mean():.1%}", f"{((df_scenario['OTIF'].mean()*100)-95):+.1f}% vs target")
    col3.metric("Total Cost", f"${df_scenario['Total_Cost'].sum():,.0f}", f"${df_scenario['Total_Cost'].mean():,.0f} avg")
    col4.metric("Avg Delay", f"{df_scenario['Delivery_Delay'].mean():.1f} days")
    col5.metric("DIO", f"{df_scenario['DIO'].mean():.0f} days")
    col6.metric("Shipments", f"{len(df_scenario):,}")
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        trend_data = df_scenario.groupby(df_scenario['Date'].dt.to_period('M').astype(str))['Total_Cost'].sum().reset_index()
        trend_data.columns = ['Month', 'Total_Cost']
        fig_trend = px.line(trend_data, x='Month', y='Total_Cost', title="Monthly Cost Trend", markers=True)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col_b:
        cost_pie = df_scenario[['Lost_Sales_Cost', 'Transportation_Cost', 'Inventory_Holding_Cost']].sum()
        fig_pie = px.pie(values=cost_pie.values, names=cost_pie.index, title="Cost Breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)

# Tab 2: Risk Alerts
with tab2:
    st.subheader("🚨 **High-Risk Events & Anomalies**")
    anomalies = detect_anomalies(df_scenario)
    
    if not anomalies.empty:
        col1, col2 = st.columns([3,1])
        with col1:
            st.dataframe(anomalies[['Date', 'Product_ID', 'Supplier_ID', 'Delivery_Delay', 
                                  'Total_Cost', 'Disruption_Type']], use_container_width=True)
        with col2:
            st.metric("🚨 Anomalies", len(anomalies), delta=f"{len(anomalies)/len(df_scenario)*100:.1f}% of shipments")
    else:
        st.success("✅ No significant anomalies detected")

# Tab 3: Suppliers
with tab3:
    st.subheader("🏭 **Supplier Performance**")
    supplier_perf = supplier_performance(df_scenario)
    st.dataframe(supplier_perf, use_container_width=True)
    
    fig_supplier = px.scatter(supplier_perf.head(10), x='Avg_Delay', y='OTIF_Rate_%', 
                            size='Shipments', color='Avg_Cost',
                            hover_name='Supplier_ID', title="Supplier Performance Matrix")
    st.plotly_chart(fig_supplier, use_container_width=True)

# Tab 4: Inventory
with tab4:
    st.subheader("📦 **Inventory Analytics**")
    col1, col2 = st.columns(2)
    
    with col1:
        abc = abc_analysis(df_scenario)
        abc_top = abc.head(15)
        fig_abc = px.bar(abc_top, x='Product_ID', y='Actual_Demand',
                        title="Top 15 SKUs - ABC Analysis", color='ABC')
        st.plotly_chart(fig_abc, use_container_width=True)
    
    with col2:
        dio_by_category = df_scenario.groupby('Product_Category')['DIO'].mean().reset_index()
        fig_dio = px.bar(dio_by_category, x='Product_Category', y='DIO',
                        title="Days Inventory by Category")
        st.plotly_chart(fig_dio, use_container_width=True)

# Tab 5: Costs
with tab5:
    st.subheader("💰 **Cost Analysis**")
    col1, col2 = st.columns(2)
    
    with col1:
        cost_by_region = df_scenario.groupby('Region')['Total_Cost'].sum().reset_index()
        fig_region = px.bar(cost_by_region, x='Region', y='Total_Cost',
                          title="Total Cost by Region", color='Total_Cost')
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        cost_by_mode = df_scenario.groupby('Transportation_Mode')['Transportation_Cost'].sum().reset_index()
        fig_mode = px.pie(cost_by_mode, values='Transportation_Cost', names='Transportation_Mode',
                         title="Transport Cost by Mode")
        st.plotly_chart(fig_mode, use_container_width=True)

# Tab 6: AI Recommendations
with tab6:
    st.header("🤖 **AI-Powered Recommendations**")
    
    top_suppliers, top_transport, top_products = recommendations
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.subheader("🏆 **Top 5 Suppliers**")
        st.dataframe(top_suppliers[['Supplier_ID', 'Overall_Score']].round(3), use_container_width=True)
        st.success(f"**#1 Choice**: {top_suppliers.iloc[0]['Supplier_ID']} (Score: {top_suppliers.iloc[0]['Overall_Score']:.3f})")
    
    with c2:
        st.subheader("🚚 **Best Transport Modes**")
        st.dataframe(top_transport[['Transportation_Mode', 'Overall_Score']].round(3), use_container_width=True)
        st.info(f"**Primary**: {top_transport.iloc[0]['Transportation_Mode']} ({top_transport.iloc[0]['Overall_Score']:.3f})")
    
    with c3:
        st.subheader("📦 **Priority Products**")
        st.dataframe(top_products[['Product_ID', 'Priority_Score']].head(5).round(3), use_container_width=True)
        st.balloons()
        st.success(f"**Top Priority**: {top_products.iloc[0]['Product_ID']}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🚚 Supply Chain Pro Analytics**")
with col2:
    st.markdown("*AI Recommendations • Real-time Analytics*")
with col3:
    st.markdown(f"**{len(df_scenario):,} records | {scenario} scenario**")
