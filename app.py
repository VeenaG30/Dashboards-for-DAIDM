import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Supply Chain Decision Engine", layout="wide")

# --- THEME TOGGLE ---
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("<style>body {background-color: #0e1117; color: white;}</style>", unsafe_allow_html=True)

# --- LOAD AND CLEAN DATA ---
@st.cache_data
def load_data():
    try:
        # Attempt to read with semicolon, fallback to comma
        df = pd.read_csv("supply_chain_data.csv", sep=';')
        
        # If only one column was read, it means the separator was wrong
        if len(df.columns) == 1:
            df = pd.read_csv("supply_chain_data.csv", sep=',')
            
        # CRITICAL: Clean column names (remove leading/trailing spaces/newlines)
        df.columns = df.columns.str.strip()
        
        # Convert Date if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

df = load_data()

# --- APP LOGIC ---
if df is not None:
    st.title("🌐 Supply Chain Decision Support System")
    
    # Debugging: Show columns if something is missing
    # st.write("Columns detected:", df.columns.tolist()) 

    # --- EXECUTIVE SUMMARY ---
    st.subheader("💡 Executive Decision Summary")
    col1, col2, col3 = st.columns(3)
    
    # Use .get() or check existence to prevent KeyError
    lost_sales = df['Lost_Sales_Cost'].sum() if 'Lost_Sales_Cost' in df.columns else 0
    fill_rate = df['Fill_Rate'].mean() if 'Fill_Rate' in df.columns else 0
    high_risk = df[df['Stockout_Flag'] == 1]['Product_ID'].nunique() if 'Stockout_Flag' in df.columns else 0

    col1.metric("Total Lost Sales", f"${lost_sales:,.0f}")
    col2.metric("Avg Fill Rate", f"{fill_rate:.1%}")
    col3.metric("High Risk SKUs", high_risk)

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Diagnostic", "Forecasting", "Predictive", "Prescriptive Action"])

    with tab1:
        if 'Disruption_Type' in df.columns and 'Lost_Sales_Cost' in df.columns:
            fig = px.pie(df, values='Lost_Sales_Cost', names='Disruption_Type', title="Financial Impact by Disruption Type")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Required columns for Diagnostic missing.")

    with tab2:
        if 'Date' in df.columns and 'Demand_Forecast' in df.columns:
            fig = px.line(df.groupby('Date')[['Demand_Forecast', 'Actual_Demand']].sum().reset_index(), 
                          x='Date', y=['Demand_Forecast', 'Actual_Demand'])
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if 'Lead_Time' in df.columns and 'Inventory_Level' in df.columns:
            fig = px.scatter(df, x='Lead_Time', y='Inventory_Level', color='Stockout_Flag', 
                             size='Lost_Sales_Cost', hover_data=['Product_ID'])
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.write("### 🛠 Managerial Action Center")
        if 'Excess_Inventory_Flag' in df.columns:
            excess = df[df['Excess_Inventory_Flag'] == 1]
            if not excess.empty:
                st.warning(f"⚠️ {len(excess)} items have excess inventory. Reduce procurement.")
        
        if 'Supplier_Rating' in df.columns:
            poor = df[df['Supplier_Rating'] < 3]['Supplier_ID'].unique()
            st.info(f"📋 Supplier Review: {len(poor)} suppliers have low ratings.")
            
        st.dataframe(df.head(10))

else:
    st.warning("Data could not be loaded. Please check your CSV file name and format.")
