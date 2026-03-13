import streamlit as st
import pandas as pd
import plotly.express as px
from utils import get_anomaly_data, get_association_rules

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

# Sidebar
st.sidebar.header("Global Filters")
category = st.sidebar.multiselect("Product Category", df['Product_Category'].unique())
mode = st.sidebar.multiselect("Transport Mode", df['Transportation_Mode'].unique())

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Anomaly & Risk", "Root Cause Analysis", "Forecasting"])

with tab1:
    st.subheader("Key Performance Indicators")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Fill Rate", f"{df['Fill_Rate'].mean():.2f}%")
    c2.metric("Total Lost Sales Cost", f"${df['Lost_Sales_Cost'].sum():,.0f}")
    c3.metric("Avg Delivery Delay", f"{df['Delivery_Delay'].mean():.1f} Days")
    
    st.markdown("---")
    fig = px.bar(df.groupby('Product_Category')['Actual_Demand'].sum().reset_index(), 
                 x='Product_Category', y='Actual_Demand', title="Demand by Category")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Anomaly Detection")
    anomalies = get_anomaly_data(df)
    st.write("High-risk disruption events detected by the model:")
    st.dataframe(anomalies[['Date', 'Product_ID', 'Delivery_Delay', 'Disruption_Type']])

with tab3:
    st.subheader("Association Rule Mining")
    rules = get_association_rules(df)
    st.write("Patterns: Which disruption types correlate with specific transport modes?")
    st.dataframe(rules)

with tab4:
    st.subheader("Scenario Simulation")
    delay_input = st.slider("Simulate Lead Time Increase (Days)", 0, 30, 0)
    impacted_df = df.copy()
    impacted_df['New_Lead_Time'] = impacted_df['Lead_Time'] + delay_input
    st.write(f"New Average Lead Time: {impacted_df['New_Lead_Time'].mean():.2f} days")
