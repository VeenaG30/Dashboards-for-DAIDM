import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Supply Chain AI Engine", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    filename = "data.csv"
    try:
        # Try reading with semicolon, fallback to comma
        df = pd.read_csv(filename, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(filename, sep=',')
        
        df.columns = df.columns.str.strip()
        
        # Find the Date column (case-insensitive)
        date_col = next((col for col in df.columns if col.lower() == 'date'), None)
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

df = load_data()

if df is not None:
    st.title("🌐 Supply Chain Decision Support & AI Engine")

    # --- EXECUTIVE SUMMARY ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Lost Sales", f"${df['Lost_Sales_Cost'].sum():,.0f}")
    col2.metric("Avg Fill Rate", f"{df['Fill_Rate'].mean():.1%}")
    col3.metric("High Risk SKUs", int(df['Stockout_Flag'].sum()))
    col4.metric("Avg Lead Time", f"{df['Lead_Time'].mean():.1f} Days")

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["Diagnostic", "Forecasting", "Predictive AI", "Prescriptive Action"])

    with tab1:
        st.subheader("Financial Impact by Disruption")
        fig = px.pie(df, values='Lost_Sales_Cost', names='Disruption_Type', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Demand vs. Forecast")
        fig = px.line(df.groupby('Date')[['Demand_Forecast', 'Actual_Demand']].sum().reset_index(), x='Date', y=['Demand_Forecast', 'Actual_Demand'])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Inventory Segmentation (K-Means Clustering)")
        features = df[['Inventory_Level', 'Holding_Cost']].dropna()
        kmeans = KMeans(n_clusters=3, n_init=10).fit(features)
        df['Cluster'] = kmeans.labels_.astype(str)
        fig_cluster = px.scatter(df, x='Inventory_Level', y='Holding_Cost', color='Cluster', title="Inventory vs Holding Cost Clusters")
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.subheader("Lead Time Impact Analysis (Regression)")
        model = LinearRegression().fit(df[['Lead_Time']], df['Lost_Sales_Cost'])
        fig_reg = px.scatter(df, x='Lead_Time', y='Lost_Sales_Cost', trendline="ols", title="Correlation: Lead Time vs Lost Sales")
        st.plotly_chart(fig_reg, use_container_width=True)

    with tab4:
        st.subheader("🛠 Managerial Action Center")
        delay = st.sidebar.slider("Simulate Lead Time Increase (Days)", 0, 30, 5)
        impact = model.predict([[delay]])[0]
        st.warning(f"If lead time increases by {delay} days, projected lost sales increase by **${impact:,.2f}**")
        st.dataframe(df.head(10))

    # --- FOOTER ---
    theme = st.sidebar.radio("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("<style>body {background-color: #0e1117; color: white;}</style>", unsafe_allow_html=True)

else:
    st.error("Could not load 'data.csv'. Please ensure the file is in the root directory of your GitHub repo.")
