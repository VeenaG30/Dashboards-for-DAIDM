"""
🚀 SUPPLY CHAIN DASHBOARD - FINAL VERSION
✅ 5th Tab: Raw Data 
✅ 3D Cluster Graph (BIGGER)
✅ All previous fixes included
✅ Production Ready
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Supply Chain Analytics", page_icon="📦", layout="wide")

st.markdown("""
<style>
.metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 0.5rem;}
.insight-box {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
              color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_supply_chain_data():
    np.random.seed(42)
    n_records = 5000
    
    products = ['SKU-'+str(i).zfill(4) for i in range(1, 301)]
    customers = ['CUST-'+str(i).zfill(4) for i in range(1, 51)]
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'Product_ID': np.random.choice(products, n_records),
        'Customer_ID': np.random.choice(customers, n_records),
        'Region': np.random.choice(regions, n_records),
        'Date': pd.date_range('2023-01-01', periods=n_records//10, freq='D').repeat(10)[:n_records],
        'Actual_Demand': np.random.poisson(50, n_records),
        'Lead_Time_Days': np.random.randint(2, 30, n_records),
        'Delivery_Status': np.random.choice(['On Time', 'Delayed', 'Early'], n_records, p=[0.7, 0.25, 0.05]),
        'Order_Value': np.random.uniform(100, 5000, n_records),
        'Supplier_Rating': np.round(np.random.uniform(1, 5, n_records), 2),
        'Inventory_Level': np.random.randint(0, 1000, n_records)
    }
    
    df = pd.DataFrame(data)
    df['Actual_Demand'] = df['Actual_Demand'].clip(0)
    return df

@st.cache_data
def load_data():
    df = generate_supply_chain_data()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def abc_analysis(df):
    demand_summary = df.groupby('Product_ID', as_index=False)['Actual_Demand'].sum()
    demand_summary = demand_summary.sort_values('Actual_Demand', ascending=False)
    total_demand = demand_summary['Actual_Demand'].sum()
    demand_summary['Cumulative_Percent'] = demand_summary['Actual_Demand'].cumsum() / total_demand
    demand_summary['ABC'] = pd.cut(demand_summary['Cumulative_Percent'], 
                                   bins=[0, 0.7, 0.9, 1], labels=['A', 'B', 'C'])
    return demand_summary

def get_kpis(df):
    return {
        'total_orders': len(df),
        'on_time_rate': (df['Delivery_Status'] == 'On Time').mean() * 100,
        'avg_lead_time': df['Lead_Time_Days'].mean(),
        'total_demand': df['Actual_Demand'].sum(),
        'avg_inventory': df['Inventory_Level'].mean()
    }

def create_3d_cluster(df):
    """Create BIG 3D cluster visualization"""
    features = ['Actual_Demand', 'Lead_Time_Days', 'Order_Value']
    X = df[features].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster to dataframe
    df_cluster = df.copy()
    df_cluster['Cluster'] = clusters
    
    return df_cluster, kmeans, scaler

# MAIN APP
st.title("📦 Supply Chain Analytics Dashboard")

df = load_data()

# === TABS FIRST (5 TABS NOW) ===
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔄 Demand", "🚚 Operations", "💰 Finance", "📋 Raw Data"])

# === FILTERS BELOW TABS ===
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    date_range = st.date_input("📅 Date Range", value=(df['Date'].min().date(), df['Date'].max().date()))
with col2:
    region_filter = st.multiselect("🌍 Region", options=sorted(df['Region'].unique()), default=sorted(df['Region'].unique()))
with col3:
    status_filter = st.multiselect("✅ Status", options=sorted(df['Delivery_Status'].unique()), default=sorted(df['Delivery_Status'].unique()))

# Apply filters
mask = (
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Region'].isin(region_filter)) &
    (df['Delivery_Status'].isin(status_filter))
)
df_filtered = df[mask].copy()
df_filtered['Month_Year'] = df_filtered['Date'].dt.strftime('%Y-%m')

# === TAB 1: OVERVIEW ===
with tab1:
    st.header("📊 Executive Overview")
    
    kpis = get_kpis(df_filtered)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown(f'<div class="metric-card"><h3>📋 Orders</h3><h2>{int(kpis["total_orders"]):,}</h2></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h3>✅ On-Time</h3><h2>{kpis["on_time_rate"]:.1f}%</h2></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><h3>⏱️ Lead Time</h3><h2>{kpis["avg_lead_time"]:.1f}d</h2></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><h3>📈 Demand</h3><h2>{int(kpis["total_demand"]):,}</h2></div>', unsafe_allow_html=True)
    with col5: st.markdown(f'<div class="metric-card"><h3>📦 Inventory</h3><h2>{int(kpis["avg_inventory"]):,}</h2></div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        abc = abc_analysis(df_filtered)
        fig_abc = px.bar(abc.head(15), x='Product_ID', y='Actual_Demand', 
                        color='ABC', title="🔝 Top 15 SKUs",
                        color_discrete_map={'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'})
        fig_abc.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_abc, use_container_width=True)
    
    with col_b:
        fig_lead = px.histogram(df_filtered, x='Lead_Time_Days', nbins=20, 
                              color='Delivery_Status', title="⏰ Lead Times")
        fig_lead.update_layout(height=350)
        st.plotly_chart(fig_lead, use_container_width=True)
    
    a_class_count = len(abc[abc['ABC'] == 'A'])
    st.markdown(f"""
    <div class="insight-box">
        <h3>💡 Actionable Insights</h3>
        <ul>
            <li><strong>{a_class_count}</strong> A-Class SKUs (70%+ demand)</li>
            <li>On-time: <strong>{kpis["on_time_rate"]:.1f}%</strong></li>
            <li>Lead time: <strong>{kpis["avg_lead_time"]:.1f}</strong> days</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === TAB 2: DEMAND ===
with tab2:
    st.header("🔄 Demand Planning")
    
    col1, col2 = st.columns(2)
    with col1:
        demand_daily = df_filtered.groupby(df_filtered['Date'].dt.date)['Actual_Demand'].sum().reset_index()
        demand_daily['Date'] = pd.to_datetime(demand_daily['Date'])
        fig_trend = px.line(demand_daily, x='Date', y='Actual_Demand', title="📈 Daily Demand")
        fig_trend.update_layout(height=350)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        demand_monthly = df_filtered.groupby('Month_Year')['Actual_Demand'].sum().reset_index()
        demand_monthly['Month_Year'] = pd.to_datetime(demand_monthly['Month_Year'] + '-01')
        fig_monthly = px.bar(demand_monthly, x='Month_Year', y='Actual_Demand', title="📊 Monthly Demand")
        fig_monthly.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    recent_avg = df_filtered['Actual_Demand'].tail(30).mean()
    st.markdown(f"""
    <div class="insight-box">
        <h3>💡 Demand Insights</h3>
        <ul>
            <li>Recent avg: <strong>{recent_avg:.0f}</strong> units/day</li>
            <li>Weekly forecast: <strong>{recent_avg*7:.0f}</strong> units</li>
            <li>Monitor monthly seasonality</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === TAB 3: OPERATIONS ===
with tab3:
    st.header("🚚 Operations")
    
    col1, col2 = st.columns(2)
    with col1:
        region_stats = df_filtered.groupby('Region', as_index=False).agg({
            'Actual_Demand': 'sum',
            'Delivery_Status': lambda x: (x == 'On Time').mean() * 100
        }).round(2)
        fig_region = px.bar(region_stats, x='Region', y=['Actual_Demand', 'Delivery_Status'],
                          title="🌍 Regional Performance", barmode='group')
        fig_region.update_layout(height=350)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(df_filtered, names='Delivery_Status', title="📋 Status Breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    if len(region_stats) > 0:
        top_region = region_stats.loc[region_stats['Actual_Demand'].idxmax(), 'Region']
        delayed_pct = (df_filtered['Delivery_Status'] == 'Delayed').mean() * 100
        st.markdown(f"""
        <div class="insight-box">
            <h3>💡 Operations Insights</h3>
            <ul>
                <li><strong>{top_region}</strong>: Top demand region</li>
                <li><strong>{delayed_pct:.1f}%</strong> delayed orders</li>
                <li>Optimize capacity in peak regions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# === TAB 4: FINANCE + 3D CLUSTER ===
with tab4:
    st.header("💰 Finance & Clustering")
    
    col1, col2 = st.columns(2)
    with col1:
        sample_data = df_filtered.sample(min(1000, len(df_filtered)))
        fig_scatter = px.scatter(sample_data, x='Actual_Demand', y='Order_Value', 
                               color='Region', size='Lead_Time_Days', 
                               title="💵 Demand vs Order Value")
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # ✅ BIG 3D CLUSTER GRAPH
        df_cluster, kmeans, scaler = create_3d_cluster(df_filtered.sample(2000))
        
        fig_3d = px.scatter_3d(df_cluster, x='Actual_Demand', y='Order_Value', z='Lead_Time_Days',
                             color='Cluster',
                             title="🌐 3D Customer Clustering",
                             color_discrete_sequence=px.colors.qualitative.Set1,
                             size_max=10)
        fig_3d.update_layout(
            height=500,  # BIGGER
            scene=dict(
                xaxis_title="Demand",
                yaxis_title="Order Value",
                zaxis_title="Lead Time",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))  # 3D perspective
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    
    avg_value = df_filtered['Order_Value'].mean()
    high_value_pct = (df_filtered['Order_Value'] > avg_value * 1.5).mean() * 100
    cluster_counts = df_cluster['Cluster'].value_counts()
    st.markdown(f"""
    <div class="insight-box">
        <h3>💡 Financial & Clustering Insights</h3>
        <ul>
            <li>Avg order: <strong>${avg_value:,.0f}</strong></li>
            <li><strong>{high_value_pct:.0f}%</strong> high-value orders</li>
            <li>5 customer clusters identified for targeted strategies</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === TAB 5: RAW DATA ===
with tab5:
    st.header("📋 Raw Data Explorer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Filtered Dataset")
        st.dataframe(
            df_filtered[['Product_ID', 'Customer_ID', 'Region', 'Date', 'Actual_Demand', 
                        'Lead_Time_Days', 'Delivery_Status', 'Order_Value']].head(1000),
            use_container_width=True,
            height=600
        )
    
    with col2:
        st.subheader("📈 Dataset Summary")
        st.metric("Total Records", len(df_filtered))
        st.metric("Date Range", f"{df_filtered['Date'].min().date()} to {df_filtered['Date'].max().date()}")
        st.metric("Unique Products", df_filtered['Product_ID'].nunique())
        st.metric("Unique Customers", df_filtered['Customer_ID'].nunique())
        
        # Download button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="💾 Download CSV",
            data=csv,
            file_name="supply_chain_data_filtered.csv",
            mime="text/csv"
        )
    
    st.markdown("""
    <div class="insight-box">
        <h3>💡 Data Tips</h3>
        <ul>
            <li>Adjust filters above to explore different data slices</li>
            <li>Download filtered data for further analysis</li>
            <li>Clustering data available in Finance tab</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown("*✅ Production Ready | 5 Tabs | 3D Clustering | Raw Data Export*")
