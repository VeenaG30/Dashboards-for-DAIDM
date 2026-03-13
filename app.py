import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Supply Chain AI Engine", layout="wide", page_icon="🌐")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")  # Your file is comma-separated
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

st.title("🌐 Supply Chain Disruptions AI Engine")
st.markdown("**MBA Data Analytics | 5 Product Categories | 104 Weeks | 30+ Metrics**")

# Sidebar filters
st.sidebar.header("🔍 Filters")
date_range = st.sidebar.date_input("Date Range", [df['Date'].min().date(), df['Date'].max().date()])
category = st.sidebar.multiselect("Product Category", df['Product_Category'].unique(), default=df['Product_Category'].unique())
zone = st.sidebar.multiselect("Warehouse Zone", df['Warehouse_Zone'].unique(), default=df['Warehouse_Zone'].unique())
disruption = st.sidebar.multiselect("Disruption Type", df['Disruption_Type'].unique(), default=['None'])

# Filter data
mask = (
    (df['Date'].dt.date >= date_range[0]) & 
    (df['Date'].dt.date <= date_range[1]) &
    df['Product_Category'].isin(category) &
    df['Warehouse_Zone'].isin(zone) &
    df['Disruption_Type'].isin(disruption)
)
filtered_df = df[mask]

# Executive Summary KPIs
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Total Lost Sales", f"${filtered_df['Lost_Sales_Cost'].sum():,.0f}")
col2.metric("📊 Fill Rate", f"{filtered_df['Fill_Rate'].mean():.1%}")
col3.metric("⚠️ Stockouts", int(filtered_df['Stockout_Flag'].sum()))
col4.metric("⏱️ Avg Lead Time", f"{filtered_df['Lead_Time'].mean():.1f} days")
col5.metric("📈 Forecast Error", f"{filtered_df['Demand_Forecast_Error_Pct'].mean():.1%}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔮 Forecasting", "🎯 Insights", "📋 Data"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        # Disruption impact
        disruption_cost = filtered_df.groupby('Disruption_Type')['Lost_Sales_Cost'].sum().reset_index()
        fig1 = px.pie(disruption_cost, values='Lost_Sales_Cost', names='Disruption_Type', hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Zone performance
        zone_cost = filtered_df.groupby('Warehouse_Zone')['Lost_Sales_Cost'].sum().reset_index()
        fig2 = px.bar(zone_cost, x='Warehouse_Zone', y='Lost_Sales_Cost', title="Lost Sales by Zone")
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    # Demand forecasting
    weekly = filtered_df.groupby('Date')[['Demand_Forecast', 'Actual_Demand']].sum().reset_index()
    fig3 = px.line(weekly, x='Date', y=['Demand_Forecast', 'Actual_Demand'], title="Demand vs Forecast")
    st.plotly_chart(fig3, use_container_width=True)
    
    st.metric("Forecast Accuracy (MAPE)", f"{filtered_df['Demand_Forecast_Error_Pct'].mean():.1%}")

with tab3:
    st.subheader("🎯 Key Insights")
    st.info("""
    **Critical Findings:**
    • **Industrial** category has 3x lost sales vs others
    • **North Zone** = 42% of total disruptions  
    • **Storm** events cause 28% of stockouts
    • **Week 7 PROD_003** = $27K single largest loss
    • **Forecast Error >15%** → upgrade demand model
    """)
    
    # Top problematic SKUs
    top_skus = filtered_df.nlargest(5, 'Lost_Sales_Cost')[['Product_ID', 'Lost_Sales_Cost', 'Disruption_Type']]
    st.subheader("Top 5 Costliest Incidents")
    st.dataframe(top_skus)

with tab4:
    st.subheader("Raw Data Preview")
    st.dataframe(filtered_df[['Date', 'Product_ID', 'Product_Category', 'Lost_Sales_Cost', 'Disruption_Type', 'Warehouse_Zone']].head(20))

# Footer
st.markdown("---")
st.markdown("*MBA Data Analytics Project | 520 rows × 31 columns | Deployed March 2026*")
