"""
🚀 SUPPLY CHAIN DASHBOARD - DUPLICATE ID FIXED
✅ Unique keys for ALL widgets
✅ Summary at END of each tab  
✅ Filters ABOVE graphs in each tab
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
.summary-box {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
              color: white; padding: 1.5rem; border-radius: 15px; margin: 2rem 0;}
.reco-card {background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); 
            color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;}
.filter-card {background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 5px solid #007bff;}
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
    features = ['Actual_Demand', 'Lead_Time_Days', 'Order_Value']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_cluster = df.copy()
    df_cluster['Cluster'] = clusters
    return df_cluster

# MAIN APP
st.title("📦 Supply Chain Analytics Dashboard")
df = load_data()

# === 6 TABS ===
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview", "🔄 Demand", "🚚 Operations", 
    "💰 Finance", "📋 Raw Data", "🤖 Recommendations"
])

# === TAB 1: OVERVIEW ===
with tab1:
    st.header("📊 Executive Overview")
    
    # ✅ FIXED: Unique keys for widgets
    st.markdown('<div class="filter-card"><strong>🔧 Overview Filters</strong></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: 
        date_range = st.date_input("Date", value=(df['Date'].min().date(), df['Date'].max().date()), key="overview_date")
    with col2: 
        region_filter = st.multiselect("Region", options=sorted(df['Region'].unique()), default=sorted(df['Region'].unique()), key="overview_region")
    with col3: 
        status_filter = st.multiselect("Status", options=sorted(df['Delivery_Status'].unique()), default=sorted(df['Delivery_Status'].unique()), key="overview_status")
    
    mask = (
        (df['Date'] >= pd.to_datetime(date_range[0])) &
        (df['Date'] <= pd.to_datetime(date_range[1])) &
        (df['Region'].isin(region_filter)) &
        (df['Delivery_Status'].isin(status_filter))
    )
    df_tab = df[mask].copy()
    
    # KPIs
    kpis = get_kpis(df_tab)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown(f'<div class="metric-card"><h3>📋 Orders</h3><h2>{int(kpis["total_orders"]):,}</h2></div>', unsafe_allow_html=True)
    with col2: st.markdown(f'<div class="metric-card"><h3>✅ On-Time</h3><h2>{kpis["on_time_rate"]:.1f}%</h2></div>', unsafe_allow_html=True)
    with col3: st.markdown(f'<div class="metric-card"><h3>⏱️ Lead Time</h3><h2>{kpis["avg_lead_time"]:.1f}d</h2></div>', unsafe_allow_html=True)
    with col4: st.markdown(f'<div class="metric-card"><h3>📈 Demand</h3><h2>{int(kpis["total_demand"]):,}</h2></div>', unsafe_allow_html=True)
    with col5: st.markdown(f'<div class="metric-card"><h3>📦 Inventory</h3><h2>{int(kpis["avg_inventory"]):,}</h2></div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        abc = abc_analysis(df_tab)
        fig_abc = px.bar(abc.head(15), x='Product_ID', y='Actual_Demand', color='ABC', 
                        title="Top 15 SKUs", color_discrete_map={'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'})
        st.plotly_chart(fig_abc, use_container_width=True)
    
    with col_b:
        fig_lead = px.histogram(df_tab, x='Lead_Time_Days', nbins=20, color='Delivery_Status', title="Lead Times")
        st.plotly_chart(fig_lead, use_container_width=True)
    
    # SUMMARY
    a_class_count = len(abc[abc['ABC'] == 'A'])
    st.markdown(f"""
    <div class="summary-box">
        <h3>📋 Overview Summary</h3>
        <ul>
            <li><strong>{len(df_tab):,}</strong> orders | <strong>{a_class_count}</strong> A-Class SKUs</li>
            <li>On-time: <strong>{kpis["on_time_rate"]:.1f}%</strong> | Lead time: <strong>{kpis["avg_lead_time"]:.1f}d</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === TAB 2: DEMAND ===
with tab2:
    st.header("🔄 Demand Planning")
    
    st.markdown('<div class="filter-card"><strong>🔧 Demand Filters</strong></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: 
        date_range = st.date_input("Date", value=(df['Date'].min().date(), df['Date'].max().date()), key="demand_date")
    with col2: 
        region_filter = st.multiselect("Region", options=sorted(df['Region'].unique()), default=sorted(df['Region'].unique()), key="demand_region")
    
    mask = (
        (df['Date'] >= pd.to_datetime(date_range[0])) &
        (df['Date'] <= pd.to_datetime(date_range[1])) &
        (df['Region'].isin(region_filter))
    )
    df_tab = df[mask].copy()
    df_tab['Month_Year'] = df_tab['Date'].dt.strftime('%Y-%m')
    
    col1, col2 = st.columns(2)
    with col1:
        demand_daily = df_tab.groupby(df_tab['Date'].dt.date)['Actual_Demand'].sum().reset_index()
        demand_daily['Date'] = pd.to_datetime(demand_daily['Date'])
        fig_trend = px.line(demand_daily, x='Date', y='Actual_Demand', title="Daily Demand")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        demand_monthly = df_tab.groupby('Month_Year')['Actual_Demand'].sum().reset_index()
        demand_monthly['Month_Year'] = pd.to_datetime(demand_monthly['Month_Year'] + '-01')
        fig_monthly = px.bar(demand_monthly, x='Month_Year', y='Actual_Demand', title="Monthly Demand")
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # SUMMARY
    recent_avg = df_tab['Actual_Demand'].tail(30).mean()
    st.markdown(f"""
    <div class="summary-box">
        <h3>📋 Demand Summary</h3>
        <ul>
            <li>Total: <strong>{df_tab["Actual_Demand"].sum():,.0f}</strong> | Recent avg: <strong>{recent_avg:.0f}/day</strong></li>
            <li>Weekly forecast: <strong>{recent_avg*7:.0f}</strong> units</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === TAB 3: OPERATIONS ===
with tab3:
    st.header("🚚 Operations")
    
    st.markdown('<div class="filter-card"><strong>🔧 Operations Filters</strong></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: 
        region_filter = st.multiselect("Region", options=sorted(df['Region'].unique()), default=sorted(df['Region'].unique()), key="ops_region")
    with col2: 
        status_filter = st.multiselect("Status", options=sorted(df['Delivery_Status'].unique()), default=sorted(df['Delivery_Status'].unique()), key="ops_status")
    
    mask = (df['Region'].isin(region_filter)) & (df['Delivery_Status'].isin(status_filter))
    df_tab = df[mask].copy()
    
    col1, col2 = st.columns(2)
    with col1:
        region_stats = df_tab.groupby('Region', as_index=False).agg({
            'Actual_Demand': 'sum',
            'Delivery_Status': lambda x: (x == 'On Time').mean() * 100
        }).round(2)
        fig_region = px.bar(region_stats, x='Region', y=['Actual_Demand', 'Delivery_Status'], barmode='group', title="Regional Performance")
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col2:
        fig_pie = px.pie(df_tab, names='Delivery_Status', title="Status Breakdown")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # SUMMARY
    top_region = region_stats.loc[region_stats['Actual_Demand'].idxmax(), 'Region'] if len(region_stats) > 0 else 'N/A'
    st.markdown(f"""
    <div class="summary-box">
        <h3>📋 Operations Summary</h3>
        <ul>
            <li>Top region: <strong>{top_region}</strong> | Orders: <strong>{len(df_tab):,}</strong></li>
            <li>Regions analyzed: <strong>{df_tab['Region'].nunique()}</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === TAB 4: FINANCE ===
with tab4:
    st.header("💰 Finance & Clustering")
    
    st.markdown('<div class="filter-card"><strong>🔧 Finance Filters</strong></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: 
        min_value = st.slider("Min Order Value", 0, 5000, 0, key="finance_value")
    with col2: 
        region_filter = st.multiselect("Region", options=sorted(df['Region'].unique()), default=sorted(df['Region'].unique()), key="finance_region")
    
    mask = (df['Order_Value'] >= min_value) & (df['Region'].isin(region_filter))
    df_tab = df[mask].copy()
    
    col1, col2 = st.columns(2)
    with col1:
        sample_data = df_tab.sample(min(1000, len(df_tab)))
        fig_scatter = px.scatter(sample_data, x='Actual_Demand', y='Order_Value', color='Region', size='Lead_Time_Days', title="Demand vs Value")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        df_cluster = create_3d_cluster(df_tab.sample(2000))
        fig_3d = px.scatter_3d(df_cluster, x='Actual_Demand', y='Order_Value', z='Lead_Time_Days',
                             color='Cluster', title="3D Clustering", size_max=12,
                             color_discrete_sequence=px.colors.qualitative.Set1)
        fig_3d.update_layout(height=550, scene=dict(camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))))
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # SUMMARY
    avg_value = df_tab['Order_Value'].mean()
    st.markdown(f"""
    <div class="summary-box">
        <h3>📋 Finance Summary</h3>
        <ul>
            <li>Avg order: <strong>${avg_value:,.0f}</strong> | Total: <strong>${df_tab["Order_Value"].sum():,.0f}</strong></li>
            <li>Customer clusters: <strong>5</strong> segments</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === TAB 5: RAW DATA ===
with tab5:
    st.header("📋 Raw Data")
    
    st.markdown('<div class="filter-card"><strong>🔧 Data Filters</strong></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: 
        rows_limit = st.slider("Rows", 100, 2000, 1000, key="data_rows")
    with col2: 
        sort_col = st.selectbox("Sort by", ['Date', 'Actual_Demand', 'Order_Value'], key="data_sort")
    
    df_tab = df.sort_values(sort_col).head(rows_limit)
    st.dataframe(df_tab, use_container_width=True, height=600)
    
    # SUMMARY
    st.markdown(f"""
    <div class="summary-box">
        <h3>📋 Data Summary</h3>
        <ul>
            <li>Showing: <strong>{len(df_tab):,}</strong> of <strong>{len(df):,}</strong> records</li>
            <li>Sorted by: <strong>{sort_col}</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === TAB 6: RECOMMENDATIONS ===
with tab6:
    st.header("🤖 AI Recommendations")
    
    kpis = get_kpis(df)
    recommendations = []
    if kpis['on_time_rate'] < 90:
        recommendations.append({'priority': '🔴 CRITICAL', 'action': 'Improve Delivery', 'details': f"Need {95-kpis['on_time_rate']:.1f}% improvement"})
    
    for reco in recommendations:
        st.markdown(f"""
        <div class="reco-card">
            <h3>{reco['priority']}</h3>
            <h4>{reco['action']}</h4>
            <p>{reco['details']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # SUMMARY
    st.markdown(f"""
    <div class="summary-box">
        <h3>📋 Recommendations Summary</h3>
        <ul>
            <li>Critical actions: <strong>{len(recommendations)}</strong></li>
            <li>On-time delivery: <strong>{kpis["on_time_rate"]:.1f}%</strong></li>
            <li>Review daily for updates</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("*✅ FIXED: Unique Keys | Summary Every Tab | Filters Above Graphs*")
