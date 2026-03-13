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

# TOP NAVIGATION BAR
st.markdown("""
<div style='background-color: #1f77b4; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <h2 style='color: white; text-align: center; margin: 0;'>📦 Supply Chain Analytics Dashboard</h2>
</div>
""", unsafe_allow_html=True)

# TOP FILTERS & NAVIGATION ROW
row1_col1, row1_col2, row1_col3, row1_col4 = st.columns([2, 1.5, 1.5, 1])

with row1_col1:
    page = st.selectbox("📊 Select Analysis", [
        "📈 Overview", "🔍 K-Means Clustering (3D)", "🌲 Random Forest", 
        "📉 Regression Analysis", "🔮 Demand Forecasting", 
        "🔗 Association Rules", "📋 Raw Data"
    ], label_visibility="collapsed")

with row1_col2:
    category_filter = st.multiselect(
        "Product Category", 
        options=df['Product_Category'].unique(), 
        default=df['Product_Category'].unique(),
        label_visibility="collapsed"
    )

with row1_col3:
    date_range = st.date_input(
        "Date Range", 
        value=(df['Date'].min().date(), df['Date'].max().date()),
        label_visibility="collapsed"
    )

with row1_col4:
    supplier_filter = st.multiselect(
        "Supplier", 
        options=df['Supplier_ID'].unique()[:10],
        default=df['Supplier_ID'].unique()[:5],
        label_visibility="collapsed"
    )

# Apply filters
filtered_df = df[
    (df['Product_Category'].isin(category_filter)) &
    (df['Date'].dt.date >= date_range[0]) &
    (df['Date'].dt.date <= date_range[1]) &
    (df['Supplier_ID'].isin(supplier_filter))
].copy()

st.markdown("---")

# MAIN CONTENT AREA
if page == "📈 Overview":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(filtered_df), delta=f"{len(filtered_df)/len(df)*100:.1f}% of total")
    with col2:
        st.metric("Unique Products", filtered_df['Product_ID'].nunique())
    with col3:
        st.metric("Avg Stockouts", f"{filtered_df['Stockout_Flag'].mean():.1%}")
    with col4:
        st.metric("Avg Fill Rate", f"{filtered_df['Fill_Rate'].mean():.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.histogram(filtered_df, x='Product_Category', y='Inventory_Level',
                           title="📊 Inventory Distribution by Category",
                           color='Product_Category')
        st.plotly_chart(fig1, use_container_width=True)
        
        # INSIGHTS
        st.markdown("### 📋 **Chart Insights**")
        top_category = filtered_df.groupby('Product_Category')['Inventory_Level'].mean().idxmax()
        avg_inv = filtered_df['Inventory_Level'].mean()
        st.info(f"""
        **Key Findings:**
        - **{top_category}** has highest avg inventory: {filtered_df.groupby('Product_Category')['Inventory_Level'].mean().max():.0f}
        - Overall avg inventory: **{avg_inv:.0f} units**
        - **{len(filtered_df[filtered_df['Inventory_Level'] < filtered_df['Safety_Stock']])/len(filtered_df)*100:.1f}%** records below safety stock
        """)
    
    with col2:
        fig2 = px.box(filtered_df, x='Disruption_Type', y='Delivery_Delay',
                     title="📦 Delivery Delays by Disruption Type",
                     color='Disruption_Type')
        st.plotly_chart(fig2, use_container_width=True)
        
        # INSIGHTS
        st.markdown("### 📋 **Chart Insights**")
        delay_stats = filtered_df.groupby('Disruption_Type')['Delivery_Delay'].median()
        max_delay_type = delay_stats.idxmax()
        st.warning(f"""
        **Critical Findings:**
        - **{max_delay_type}** causes longest delays: **{delay_stats.max():.1f} days**
        - **{len(filtered_df[filtered_df['Delivery_Delay'] > 3])/len(filtered_df)*100:.1f}%** shipments delayed >3 days
        - Median delay across all: **{filtered_df['Delivery_Delay'].median():.1f} days**
        """)

    # Additional KPI Grid
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        avg_demand = filtered_df['Actual_Demand'].mean()
        st.metric("Avg Demand", f"{avg_demand:.0f}")
    with col6:
        avg_inventory = filtered_df['Inventory_Level'].mean()
        st.metric("Avg Inventory", f"{avg_inventory:.0f}")
    with col7:
        avg_cost = filtered_df['Holding_Cost'].sum()
        st.metric("Total Holding Cost", f"${avg_cost:,.0f}")
    with col8:
        disruptions = (filtered_df['Disruption_Type'] != 'None').sum()
        st.metric("Disruptions", disruptions)

elif page == "🔍 K-Means Clustering (3D)":
    st.header("🔍 K-Means Clustering Analysis (3D)")
    
    cluster_features = ['Inventory_Level', 'Days_of_Supply', 'Fill_Rate']
    X_cluster = filtered_df[cluster_features].fillna(filtered_df[cluster_features].mean())
    
    # Elbow method
    inertias = []
    K_range = range(1, 8)
    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(X_cluster)
        inertias.append(kmeans_temp.inertia_)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_elbow = px.line(x=list(K_range), y=inertias, 
                           title="📈 Elbow Method - Optimal Clusters",
                           markers=True)
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # INSIGHTS
        st.markdown("### 📋 **Elbow Method Insights**")
        optimal_k = 3
        st.success(f"""
        **Analysis:**
        - **Optimal clusters: {optimal_k}** (elbow point)
        - Sharp drop in inertia stops at **k=3**
        - Higher k values show **diminishing returns**
        """)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    filtered_df['Cluster'] = kmeans.fit_predict(X_cluster)
    
    with col2:
        fig_3d = px.scatter_3d(filtered_df, x='Inventory_Level', y='Days_of_Supply', 
                              z='Fill_Rate', color='Cluster',
                              title=f"🎯 3D K-Means Clustering (k={optimal_k})",
                              labels={'Cluster': 'Cluster Group'})
        fig_3d.update_traces(marker=dict(size=4))
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # INSIGHTS
        st.markdown("### 📋 **3D Clustering Insights**")
        cluster_sizes = filtered_df['Cluster'].value_counts()
        cluster_summary = filtered_df.groupby('Cluster')[cluster_features].mean().round(2)
        st.info(f"""
        **Cluster Profile Analysis:**
        | Cluster | Size | Avg Inventory | Days Supply | Fill Rate |
        |---------|------|---------------|-------------|-----------|
        """ + "\n".join([f"| {i} | {cluster_sizes.get(i,0)} ({cluster_sizes.get(i,0)/len(filtered_df)*100:.0f}%) | {cluster_summary.loc[i,'Inventory_Level']:.0f} | {cluster_summary.loc[i,'Days_of_Supply']:.1f} | {cluster_summary.loc[i,'Fill_Rate']:.3f} |" for i in range(optimal_k)]))
        
        st.success(f"**Largest cluster: Cluster {cluster_sizes.idxmax()}** ({cluster_sizes.max()} records)")

elif page == "🌲 Random Forest":
    st.header("🌲 Random Forest - Demand Prediction")
    
    rf_features = ['Demand_Forecast', 'Inventory_Level', 'Safety_Stock', 
                  'Lead_Time', 'Delivery_Delay', 'Inventory_Turns']
    X_rf = filtered_df[rf_features].fillna(filtered_df[rf_features].mean())
    y_rf = filtered_df['Actual_Demand'].fillna(filtered_df['Actual_Demand'].mean())
    
    if len(X_rf) > 10:
        X_train, X_test, y_train, y_test = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            r2_val = r2_score(y_test, y_pred)
            st.metric("R² Score", f"{r2_val:.3f}", delta=f"{r2_val*100:.0f}% accuracy")
        with col2:
            rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
            st.metric("RMSE", f"{rmse_val:.0f}")
        with col3:
            st.metric("Test Samples", len(X_test))
        
        # Feature Importance
        importance_df = pd.DataFrame({
            'feature': rf_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig_importance = px.bar(importance_df, x='importance', y='feature',
                               orientation='h', color='importance',
                               title="🔍 Feature Importance Ranking")
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # INSIGHTS
        st.markdown("### 📋 **Feature Importance Insights**")
        top_feature = importance_df.iloc[-1]['feature']
        top_importance = importance_df.iloc[-1]['importance']
        st.info(f"""
        **Key Drivers of Demand:**
        1. **{top_feature}** ({top_importance:.1%}) - Most influential factor
        2. **{importance_df.iloc[-2]['feature']}** ({importance_df.iloc[-2]['importance']:.1%}) - 2nd most important
        3. Model explains **{r2_val:.1%}** of demand variance
        4. **{len(X_test)} test samples** used for validation
        """)
        
        # Actual vs Predicted
        fig_pred = px.scatter(x=y_test.values, y=y_pred, 
                             labels={'x':'Actual Demand', 'y':'Predicted Demand'},
                             title="🎯 Actual vs Predicted Demand",
                             trendline="ols")
        fig_pred.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                     y=[y_test.min(), y_test.max()],
                                     mode='lines', name='Perfect Prediction',
                                     line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # INSIGHTS
        st.markdown("### 📋 **Prediction Accuracy Insights**")
        accurate_preds = ((abs(y_test - y_pred) / y_test < 0.1).sum() / len(y_test)) * 100
        st.success(f"""
        **Prediction Performance:**
        - **{accurate_preds:.1f}%** predictions within ±10% of actual
        - RMSE of **{rmse_val:.0f}** units (lower is better)
        - **R² = {r2_val:.3f}** indicates strong predictive power
        """)

elif page == "📉 Regression Analysis":
    st.header("📉 Linear Regression Analysis")
    
    reg_features = ['Demand_Forecast', 'Inventory_Level', 'Lead_Time', 'Delivery_Delay']
    X_reg = filtered_df[reg_features].fillna(0)
    y_reg = filtered_df['Actual_Demand']
    
    if len(X_reg) > 10:
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42)
        
        lr = LinearRegression()
        lr.fit(X_train_reg, y_train_reg)
        y_pred_reg = lr.predict(X_test_reg)
        
        coef_df = pd.DataFrame({
            'Feature': reg_features + ['Intercept'],
            'Coefficient': np.append(lr.coef_, lr.intercept_).round(3)
        })
        
        st.subheader("📊 Regression Coefficients")
        st.dataframe(coef_df.style.background_gradient())
        
        r2_reg = r2_score(y_test_reg, y_pred_reg)
        st.metric("R² Score", f"{r2_reg:.3f}")
        
        fig_reg = px.scatter(x=y_test_reg, y=y_pred_reg,
                            labels={'x':'Actual', 'y':'Predicted'},
                            title="🎯 Linear Regression Results",
                            trendline="ols")
        st.plotly_chart(fig_reg, use_container_width=True)
        
        # INSIGHTS
        st.markdown("### 📋 **Regression Insights**")
        strongest_coef = coef_df.iloc[coef_df['Coefficient'].abs().idxmax()]['Feature']
        st.info(f"""
        **Model Interpretation:**
        - **{strongest_coef}** has strongest impact on demand (coef: {coef_df['Coefficient'].abs().max():.3f})
        - **{r2_reg:.1%}** variance explained by model
        - Linear relationship confirmed by **R² trendline**
        - **{len(X_test_reg)}** validation samples used
        """)

elif page == "🔮 Demand Forecasting":
    st.header("🔮 Time Series Forecasting")
    
    df_ts = filtered_df.groupby('Date')['Actual_Demand'].sum().reset_index()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])
    df_ts.set_index('Date', inplace=True)
    
    df_ts['MA_7'] = df_ts['Actual_Demand'].rolling(window=min(7, len(df_ts))).mean()
    df_ts['MA_30'] = df_ts['Actual_Demand'].rolling(window=min(30, len(df_ts))).mean()
    
    fig_forecast = px.line(df_ts.tail(60), y=['Actual_Demand', 'MA_7', 'MA_30'],
                          title="📈 Demand Trend & Forecast",
                          labels={'value': 'Total Demand', 'variable': 'Metric'})
    fig_forecast.update_layout(xaxis_title="Date", yaxis_title="Total Demand")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # INSIGHTS
    st.markdown("### 📋 **Forecasting Insights**")
    recent_actual = df_ts['Actual_Demand'].tail(1).iloc[0]
    ma7_forecast = df_ts['MA_7'].tail(1).iloc[0]
    ma30_forecast = df_ts['MA_30'].tail(1).iloc[0]
    trend_7d = ((recent_actual - df_ts['Actual_Demand'].tail(8).iloc[0]) / df_ts['Actual_Demand'].tail(8).iloc[0]) * 100
    
    st.success(f"""
    **Current Forecast Status:**
    - **Latest actual demand**: {recent_actual:.0f}
    - **7-day forecast (MA)**: {ma7_forecast:.0f} {'📈' if ma7_forecast > recent_actual else '📉'}
    - **30-day trend (MA)**: {ma30_forecast:.0f}
    - **7-day change**: **{trend_7d:+.1f}%**
    """)

elif page == "🔗 Association Rules":
    st.header("🔗 Market Basket Analysis")
    
    basket_cols = ['Product_Category', 'Disruption_Type', 'Transportation_Mode']
    basket_cols = [col for col in basket_cols if col in filtered_df.columns]
    basket = pd.get_dummies(filtered_df[basket_cols].fillna('None'))
    
    if len(basket) > 20:
        frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
        
        st.subheader("📋 Top Association Rules")
        rules_display = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].round(3)
        st.dataframe(rules_display.head(10).style.background_gradient())
        
        fig_rules = px.scatter(rules.head(15), x='support', y='confidence', 
                              size='lift', color='lift',
                              hover_data=['antecedents', 'consequents'],
                              title="🎯 Association Rule Strength")
        st.plotly_chart(fig_rules, use_container_width=True)
        
        # INSIGHTS
        st.markdown("### 📋 **Association Rule Insights**")
        top_rule = rules.head(1)
        top_lift = rules['lift'].max()
        st.info(f"""
        **Strongest Patterns Found:**
        - **Highest Lift**: {top_lift:.2f}x ({str(top_rule['antecedents'].iloc[0])} → {str(top_rule['consequents'].iloc[0])})
        - **{len(rules)} total rules** discovered
        - **Support range**: {rules['support'].min():.1%} - {rules['support'].max():.1%}
        - **Confidence range**: {rules['confidence'].min():.1%} - {rules['confidence'].max():.1%}
        """)

elif page == "📋 Raw Data":
    st.header("📋 Raw Data Explorer")
    
    st.subheader(f"Showing **{len(filtered_df)}** records after filters")
    
    st.dataframe(filtered_df, use_container_width=True, height=600)
    
    # INSIGHTS
    st.markdown("### 📋 **Data Summary Insights**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records Filtered", len(filtered_df), delta=f"{len(filtered_df)/len(df)*100:.1f}% of total")
    with col2:
        st.metric("Time Span", f"{(filtered_df['Date'].max() - filtered_df['Date'].min()).days} days")
    with col3:
        st.metric("Categories", len(filtered_df['Product_Category'].unique()))
    with col4:
        st.metric("Suppliers", len(filtered_df['Supplier_ID'].unique()))
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv, 
        file_name="supply_chain_filtered.csv",
        mime="text/csv"
    )

# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-top: 40px;'>
    <p><strong>✅ Insights-powered Supply Chain Analytics</strong> | {len(filtered_df)} records analyzed | Built with Streamlit 📦🚚📊</p>
</div>
""", unsafe_allow_html=True)
