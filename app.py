import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Supply Chain Decision Engine", layout="wide")

# --- THEME TOGGLE ---
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body {background-color: #0e1117; color: white;}</style>", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("Supply_Chain_Data.csv", sep=';')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

st.title("🌐 Supply Chain Decision Support System")

# --- DECISION MAKING INSIGHTS (Top Level) ---
st.subheader("💡 Executive Decision Summary")
col1, col2, col3 = st.columns(3)
total_lost_sales = df['Lost_Sales_Cost'].sum()
avg_fill_rate = df['Fill_Rate'].mean()
high_risk_items = df[df['Stockout_Flag'] == 1]['Product_ID'].nunique()

col1.metric("Total Lost Sales", f"${total_lost_sales:,.0f}")
col2.metric("Avg Fill Rate", f"{avg_fill_rate:.1%}")
col3.metric("High Risk SKUs", high_risk_items)

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Diagnostic", "Forecasting", "Predictive", "Prescriptive Action"])

with tab1: # Diagnostic
    st.write("### Root Cause: Why are we losing sales?")
    fig = px.pie(df, values='Lost_Sales_Cost', names='Disruption_Type', title="Financial Impact by Disruption Type")
    st.plotly_chart(fig, use_container_width=True)

with tab2: # Forecasting
    st.write("### Demand vs. Forecast Accuracy")
    fig = px.line(df.groupby('Date')[['Demand_Forecast', 'Actual_Demand']].sum().reset_index(), 
                  x='Date', y=['Demand_Forecast', 'Actual_Demand'])
    st.plotly_chart(fig, use_container_width=True)

with tab3: # Predictive
    st.write("### Risk Heatmap")
    fig = px.scatter(df, x='Lead_Time', y='Inventory_Level', color='Stockout_Flag', 
                     size='Lost_Sales_Cost', hover_data=['Product_ID'])
    st.plotly_chart(fig, use_container_width=True)

with tab4: # Prescriptive (The Decision Engine)
    st.write("### 🛠 Managerial Action Center")
    
    # Logic for decisions
    st.subheader("Recommended Actions")
    
    # Decision 1: Excess Inventory
    excess_items = df[df['Excess_Inventory_Flag'] == 1]
    if not excess_items.empty:
        st.warning(f"⚠️ {len(excess_items)} items have excess inventory. **Action:** Reduce procurement orders for these SKUs.")
    
    # Decision 2: Stockout Prevention
    critical_items = df[(df['Stockout_Flag'] == 1) & (df['Order_Priority'] == 'High')]
    if not critical_items.empty:
        st.error(f"🚨 {len(critical_items)} High-Priority items are at risk of stockout. **Action:** Expedite shipping from secondary suppliers.")
    
    # Decision 3: Supplier Performance
    poor_suppliers = df[df['Supplier_Rating'] < 3]['Supplier_ID'].unique()
    st.info(f"📋 Supplier Review: Suppliers {list(poor_suppliers)[:3]} have low ratings. **Action:** Initiate contract renegotiation or switch to backup.")

    st.dataframe(df.head(10))
