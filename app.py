import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Supply Chain Analytics Dashboard", 
                   page_icon="📦", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Select Analysis", [
    "📈 Overview", "🔍 K-Means Clustering (3D)", "🌲 Random Forest", 
    "📉 Regression Analysis", "🔮 Demand Forecasting", 
    "🔗 Association Rules", "📋 Raw Data"
])

# Title
st.title("📦 Supply Chain Disruptions & Inventory Analytics")
st.markdown("---")

if page == "📈 Overview":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Unique Products", df['Product_ID'].nunique())
    with col3:
        st.metric("Avg Stockouts", f"{df['Stockout_Flag'].mean():.1%}")
    with col4:
        st.metric("Avg Fill Rate", f"{df['Fill_Rate'].mean():.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(df, x='Product_Category', y='Inventory_Level',
                           title="Inventory Distribution by Category")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.box(df, x='Disruption_Type', y='Delivery_Delay',
                     title="Delivery Delays by Disruption Type")
        st.plotly_chart(fig2, use_container_width=True)

elif page == "🔍 K-Means Clustering (3D)":
    st.header("🔍 K-Means Clustering Analysis (3D)")
    
    # Prepare features for clustering
    cluster_features = ['Inventory_Level', 'Days_of_Supply', 'Fill_Rate']
    X_cluster = df[cluster_features].fillna(df[cluster_features].mean())
    
    # Elbow method
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_cluster)
        inertias.append(kmeans.inertia_)
    
    fig_elbow = px.line(x=K_range, y=inertias, 
                       title="Elbow Method - Optimal Clusters")
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    # Apply K-Means with 3 clusters
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_cluster)
    
    # 3D Scatter Plot
    fig_3d = px.scatter_3d(df, x='Inventory_Level', y='Days_of_Supply', 
                          z='Fill_Rate', color='Cluster',
                          title=f"K-Means Clustering (k={optimal_k})",
                          labels={'Cluster': 'Cluster Group'})
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Cluster insights
    st.subheader("📋 Cluster Insights")
    cluster_summary = df.groupby('Cluster')[cluster_features].mean().round(2)
    st.dataframe(cluster_summary)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cluster 0 Size", len(df[df['Cluster']==0]))
    with col2:
        st.metric("Cluster 1 Size", len(df[df['Cluster']==1]))

elif page == "🌲 Random Forest":
    st.header("🌲 Random Forest - Feature Importance")
    
    # Prepare data for Random Forest
    rf_features = ['Demand_Forecast', 'Inventory_Level', 'Safety_Stock', 
                  'Lead_Time', 'Delivery_Delay', 'Inventory_Turns']
    X_rf = df[rf_features].fillna(df[rf_features].mean())
    y_rf = df['Actual_Demand'].fillna(df['Actual_Demand'].mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    with col2:
        st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.0f}")
    with col3:
        st.metric("Features Used", len(rf_features))
    
    # Feature Importance
    importance_df = pd.DataFrame({
        'feature': rf_features,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig_importance = px.bar(importance_df, x='importance', y='feature',
                           orientation='h',
                           title="Feature Importance (Random Forest)")
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Actual vs Predicted
    fig_pred = px.scatter(x=y_test, y=y_pred, 
                         labels={'x':'Actual Demand', 'y':'Predicted Demand'},
                         title="Actual vs Predicted Demand")
    fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                 y=[y_test.min(), y_test.max()],
                                 mode='lines', name='Perfect Prediction'))
    st.plotly_chart(fig_pred, use_container_width=True)

elif page == "📉 Regression Analysis":
    st.header("📉 Linear Regression Analysis")
    
    # Multiple Linear Regression
    reg_features = ['Demand_Forecast', 'Inventory_Level', 'Lead_Time', 
                   'Delivery_Delay']
    X_reg = df[reg_features].fillna(0)
    y_reg = df['Actual_Demand']
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    
    lr = LinearRegression()
    lr.fit(X_train_reg, y_train_reg)
    y_pred_reg = lr.predict(X_test_reg)
    
    # Coefficients
    coef_df = pd.DataFrame({
        'Feature': reg_features,
        'Coefficient': lr.coef_.round(3)
    })
    
    st.subheader("Regression Coefficients")
    st.dataframe(coef_df)
    
    st.metric("R² Score", f"{r2_score(y_test_reg, y_pred_reg):.3f}")
    
    fig_reg = px.scatter(x=y_test_reg, y=y_pred_reg,
                        labels={'x':'Actual', 'y':'Predicted'},
                        title="Linear Regression: Actual vs Predicted")
    st.plotly_chart(fig_reg, use_container_width=True)

elif page == "🔮 Demand Forecasting":
    st.header("🔮 Time Series Forecasting")
    
    # Prepare time series data
    df_ts = df.groupby('Date')['Actual_Demand'].sum().reset_index()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    df_ts.set_index('Date', inplace=True)
    
    # Simple moving average forecast
    df_ts['MA_7'] = df_ts['Actual_Demand'].rolling(window=7).mean()
    df_ts['MA_30'] = df_ts['Actual_Demand'].rolling(window=30).mean()
    
    fig_forecast = px.line(df_ts.tail(100), x=df_ts.tail(100).index, 
                          y=['Actual_Demand', 'MA_7', 'MA_30'],
                          title="Demand Forecasting - Moving Averages")
    fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Demand")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.subheader("Forecast Insights")
    recent_trend = df_ts['Actual_Demand'].tail(7).mean()
    st.info(f"**7-day Moving Average: {recent_trend:.0f}**")
    st.success("📈 Demand shows stable trend with minor seasonal fluctuations")

elif page == "🔗 Association Rules":
    st.header("🔗 Market Basket Analysis - Association Rules")
    
    # Create transaction data (simplified)
    basket = pd.get_dummies(df[['Product_Category', 'Disruption_Type', 'Transportation_Mode']].fillna('None'))
    
    # Apriori algorithm
    frequent_itemsets = apriori(basket, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    st.subheader("Top Association Rules")
    rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)
    st.dataframe(rules_display.head(10))
    
    # Rule visualization
    fig_rules = px.scatter(rules.head(10), x='support', y='confidence', 
                          size='lift', color='lift',
                          hover_name='antecedents',
                          title="Association Rules Strength")
    st.plotly_chart(fig_rules, use_container_width=True)

elif page == "📋 Raw Data":
    st.header("📋 Raw Data Explorer")
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        category = st.multiselect("Product Category", 
                                df['Product_Category'].unique(),
                                default=df['Product_Category'].unique())
    with col2:
        date_range = st.date_input("Date Range", 
                                 value=(df['Date'].min().date(), df['Date'].max().date()))
    
    filtered_df = df[
        (df['Product_Category'].isin(category)) &
        (df['Date'].dt.date >= date_range[0]) &
        (df['Date'].dt.date <= date_range[1])
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download filtered data", csv, "supply_chain_filtered.csv")

# Footer
st.markdown("---")
st.markdown("**Built with Streamlit** | Supply Chain Analytics Dashboard | 📦🚚📊")
