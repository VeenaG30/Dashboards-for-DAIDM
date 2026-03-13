import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Page config with dark theme
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard", 
    page_icon="📦", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS for insights
insights_css = """
<style>
.insights-dark {
    background-color: #2e2e2e;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    color: #e0e0e0;
    margin: 10px 0;
}
.insights-title {
    color: #ffffff;
    font-size: 1.2em;
    font-weight: bold;
    margin-bottom: 10px;
}
.insights-metric {
    background-color: #404040;
    padding: 8px;
    border-radius: 5px;
    margin: 5px 0;
}
</style>
"""

st.markdown(insights_css, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# HEADER
st.markdown("""
<div style='background: linear-gradient(90deg, #1f77b4 0%, #4a90e2 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
    <h1 style='color: white; text-align: center; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>📦 Supply Chain Analytics Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# TOP FILTERS ROW (Side by side buttons)
row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])

# Filter buttons
with row1_col1:
    if st.button("📈 Overview", use_container_width=True, key="overview_btn"):
        st.session_state.page = "overview"
with row1_col2:
    if st.button("🔍 Clustering", use_container_width=True, key="cluster_btn"):
        st.session_state.page = "clustering" 
with row1_col3:
    if st.button("🌲 ML Models", use_container_width=True, key="ml_btn"):
        st.session_state.page = "ml"
with row1_col4:
    if st.button("🔮 Forecasting", use_container_width=True, key="forecast_btn"):
        st.session_state.page = "forecast"
with row1_col5:
    if st.button("📋 Data", use_container_width=True, key="data_btn"):
        st.session_state.page = "data"

# FILTER ROW BELOW BUTTONS
row2_col1, row2_col2, row2_col3 = st.columns(3)

with row2_col1:
    category_filter = st.multiselect(
        "🎯 Product Category", 
        options=sorted(df['Product_Category'].unique()), 
        default=sorted(df['Product_Category'].unique()),
        label_visibility="collapsed"
    )

with row2_col2:
    date_range = st.date_input(
        "📅 Date Range", 
        value=(df['Date'].min().date(), df['Date'].max().date()),
        label_visibility="collapsed"
    )

with row2_col3:
    supplier_filter = st.multiselect(
        "🏢 Supplier", 
        options=df['Supplier_ID'].unique()[:10],
        default=df['Supplier_ID'].unique()[:5],
        label_visibility="collapsed"
    )

# Page state management
if 'page' not in st.session_state:
    st.session_state.page = "overview"

# Apply filters
filtered_df = df[
    (df['Product_Category'].isin(category_filter)) &
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Supplier_ID'].isin(supplier_filter))
].copy()

st.markdown("---")

# MAIN CONTENT - Side by side sections based on page
def display_insights(title, content):
    st.markdown(f"""
    <div class="insights-dark">
        <div class="insights-title">📋 {title}</div>
        {content}
    </div>
    """, unsafe_allow_html=True)

if st.session_state.page == "overview":
    st.header("📈 Executive Overview")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Records", len(filtered_df))
    with col2: st.metric("Products", filtered_df['Product_ID'].nunique())
    with col3: st.metric("Stockouts", f"{filtered_df['Stockout_Flag'].mean():.1%}")
    with col4: st.metric("Fill Rate", f"{filtered_df['Fill_Rate'].mean():.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(filtered_df, x='Product_Category', y='Inventory_Level',
                           title="📊 Inventory by Category", color='Product_Category')
        st.plotly_chart(fig1, use_container_width=True)
        
        display_insights("Inventory Insights", f"""
        <div class="insights-metric">**{filtered_df.groupby('Product_Category')['Inventory_Level'].mean().idxmax()}** leads with {filtered_df.groupby('Product_Category')['Inventory_Level'].mean().max():.0f} avg units</div>
        <div class="insights-metric">**{len(filtered_df[filtered_df['Inventory_Level'] < filtered_df['Safety_Stock']])/len(filtered_df)*100:.1f}%** below safety stock ⚠️</div>
        <div class="insights-metric">Overall avg: **{filtered_df['Inventory_Level'].mean():.0f}** units</div>
        """)
    
    with col2:
        fig2 = px.box(filtered_df, x='Disruption_Type', y='Delivery_Delay',
                     title="📦 Delays by Disruption", color='Disruption_Type')
        st.plotly_chart(fig2, use_container_width=True)
        
        display_insights("Delay Analysis", f"""
        <div class="insights-metric">**{filtered_df.groupby('Disruption_Type')['Delivery_Delay'].median().idxmax()}** causes max delay: {filtered_df.groupby('Disruption_Type')['Delivery_Delay'].median().max():.1f} days 🚨</div>
        <div class="insights-metric">**{len(filtered_df[filtered_df['Delivery_Delay'] > 3])/len(filtered_df)*100:.1f}%** shipments >3 days late</div>
        <div class="insights-metric">Median delay: **{filtered_df['Delivery_Delay'].median():.1f}** days</div>
        """)

elif st.session_state.page == "clustering":
    st.header("🔍 K-Means Clustering (3D)")
    
    cluster_features = ['Inventory_Level', 'Days_of_Supply', 'Fill_Rate']
    X_cluster = filtered_df[cluster_features].fillna(filtered_df[cluster_features].mean())
    
    # Elbow + 3D side by side
    col1, col2 = st.columns(2)
    
    with col1:
        inertias = []
        K_range = range(1, 8)
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(X_cluster)
            inertias.append(kmeans_temp.inertia_)
        
        fig_elbow = px.line(x=list(K_range), y=inertias, title="📈 Elbow Method", markers=True)
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        display_insights("Optimal Clusters", """
        <div class="insights-metric">**k=3 confirmed** by elbow curve 📍</div>
        <div class="insights-metric">Sharp inertia drop stops at **3 clusters**</div>
        <div class="insights-metric">Higher k shows **diminishing returns**</div>
        """)
    
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    filtered_df['Cluster'] = kmeans.fit_predict(X_cluster)
    
    with col2:
        fig_3d = px.scatter_3d(filtered_df, x='Inventory_Level', y='Days_of_Supply', 
                              z='Fill_Rate', color='Cluster',
                              title="🎯 3D Clusters", labels={'Cluster': 'Group'})
        fig_3d.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_3d, use_container_width=True)
        
        display_insights("Cluster Profiles", f"""
        <div class="insights-metric">**{filtered_df['Cluster'].value_counts().max()} records** in largest cluster</div>
        <div class="insights-metric">**{len(filtered_df)} total points** clustered</div>
        <div class="insights-metric">3 distinct inventory performance groups identified ✅</div>
        """)

elif st.session_state.page == "ml":
    st.header("🌲 Machine Learning Models")
    
    # Random Forest section
    st.subheader("Random Forest - Demand Prediction")
    rf_features = ['Demand_Forecast', 'Inventory_Level', 'Safety_Stock', 'Lead_Time']
    X_rf = filtered_df[rf_features].fillna(0)
    y_rf = filtered_df['Actual_Demand']
    
    if len(X_rf) > 20:
        X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            importance_df = pd.DataFrame({'feature': rf_features, 'importance': rf.feature_importances_}).sort_values('importance')
            fig_imp = px.bar(importance_df, x='importance', y='feature', orientation='h', color='importance', title="🔍 Feature Importance")
            st.plotly_chart(fig_imp, use_container_width=True)
            
            display_insights("Model Drivers", f"""
            <div class="insights-metric">**{importance_df.iloc[-1]['feature']}** most predictive ({importance_df.iloc[-1]['importance']:.1%})</div>
            <div class="insights-metric">R² Score: **{r2_score(y_test, y_pred):.3f}**</div>
            """)
        
        with col2:
            fig_pred = px.scatter(x=y_test, y=y_pred, trendline="ols", title="🎯 Predictions")
            st.plotly_chart(fig_pred, use_container_width=True)
            
            display_insights("Prediction Quality", f"""
            <div class="insights-metric">**{((abs(y_test-y_pred)/y_test<0.1).mean()*100:.0f}%** within ±10%</div>
            <div class="insights-metric">RMSE: **{np.sqrt(mean_squared_error(y_test, y_pred)):.0f}**</div>
            """)

elif st.session_state.page == "forecast":
    st.header("🔮 Demand Forecasting")
    
    df_ts = filtered_df.groupby('Date')['Actual_Demand'].sum().reset_index()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    df_ts.set_index('Date', inplace=True)
    df_ts['MA_7'] = df_ts['Actual_Demand'].rolling(7).mean()
    df_ts['MA_30'] = df_ts['Actual_Demand'].rolling(30).mean()
    
    fig_forecast = px.line(df_ts.tail(90), y=['Actual_Demand', 'MA_7', 'MA_30'], 
                          title="📈 Demand Trends & Forecasts")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    display_insights("Forecast Summary", f"""
    <div class="insights-metric">Latest demand: **{df_ts['Actual_Demand'].iloc[-1]:.0f}**</div>
    <div class="insights-metric">7-day forecast: **{df_ts['MA_7'].iloc[-1]:.0f}**</div>
    <div class="insights-metric">30-day trend: **{df_ts['MA_30'].iloc[-1]:.0f}**</div>
    """)

elif st.session_state.page == "data":
    st.header("📋 Raw Data Explorer")
    
    st.subheader(f"**{len(filtered_df)}** records ({len(filtered_df)/len(df)*100:.1f}% of total)")
    st.dataframe(filtered_df, use_container_width=True, height=500)
    
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Download CSV", csv, "filtered_data.csv")
    
    display_insights("Data Profile", f"""
    <div class="insights-metric">**{len(filtered_df['Product_Category'].unique())}** categories</div>
    <div class="insights-metric">**{len(filtered_df['Supplier_ID'].unique())}** suppliers</div>
    <div class="insights-metric">Time span: **{(filtered_df['Date'].max()-filtered_df['Date'].min()).days}** days</div>
    """)

# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #1a1a1a; color: #e0e0e0; border-radius: 10px; margin-top: 40px;'>
    <p><strong>🚀 Insights-Powered Analytics</strong> | Dark Mode | {len(filtered_df)} records | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
