"""
🚀 SUPPLY CHAIN ANALYTICS DASHBOARD - TAB LAYOUT
✅ Tabs ABOVE graphs, Filters BELOW tabs
✅ Insights at end of EACH tab
✅ Production-ready for Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 0.5rem;}
    .insight-box {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                  color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;}
    </style>
""", unsafe_allow_html=True)

# SAMPLE DATA GENERATION
@st.cache_data
def generate_supply_chain_data():
    np.random.seed(42)
    n_records = 10000
    
    products = ['SKU-'+str(i).zfill(4) for i in range(1, 501)]
    customers = ['CUST-'+str(i).zfill(4) for i in range(1, 101)]
    regions = ['North', 'South', 'East', 'West']
    
    data = {
        'Product_ID': np.random.choice(products, n_records),
        'Customer_ID': np.random.choice(customers, n_records),
        'Region': np.random.choice(regions, n_records),
        'Date': pd.date_range('2023-01-01', periods=n_records, freq='D').repeat(10)[:n_records],
        'Actual_Demand': np.random.poisson(50, n_records) + np.random.normal(0, 10, n_records),
        'Lead_Time_Days': np.random.randint(2, 30, n_records),
        'Delivery_Status': np.random.choice(['On Time', 'Delayed', 'Early'], n_records, p=[0.7, 0.25, 0.05]),
        'Order_Value': np.random.uniform(100, 5000, n_records),
        'Supplier_Rating': np.random.uniform(1, 5, n_records),
        'Inventory_Level': np.random.randint(0, 1000, n_records)
    }
    
    df = pd.DataFrame(data)
    df['Actual_Demand'] = df['Actual_Demand'].clip(0)
    df['Month'] = df['Date'].dt.to_period('M')
    df['Week'] = df['Date'].dt.isocalendar().week
    return df

@st.cache_data
def load_data():
    df = generate_supply_chain_data()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Analysis functions
def abc_analysis(df):
    demand_summary = df.groupby('Product_ID')['Actual_Demand'].sum().reset_index()
    demand_summary = demand_summary.sort_values('Actual_Demand', ascending=False)
    demand_summary['Cumulative_Percent'] = demand_summary['Actual_Demand'].cumsum() / demand_summary['Actual_Demand'].sum()
    demand_summary['ABC'] = pd.cut(demand_summary['Cumulative_Percent'], 
                                   bins=[0, 0.7, 0.9, 1], labels=['A', 'B', 'C'])
    return demand_summary

def calculate_kpis(df):
    total_orders = len(df)
    on_time_rate = (df['Delivery_Status'] == 'On Time').mean() * 100
    avg_lead_time = df['Lead_Time_Days'].mean()
    total_demand = df['Actual_Demand'].sum()
    avg_inventory = df['Inventory_Level'].mean()
    
    return {
        'Total Orders': total_orders,
        'On-Time Delivery': f"{on_time_rate:.1f}%",
        'Avg Lead Time': f"{avg_lead_time:.1f} days",
        'Total Demand': f"{total_demand:,.0f}",
        'Avg Inventory': f"{avg_inventory:.0f}"
    }

# MAIN APP
def main():
    st.title("📦 Supply Chain Analytics Dashboard")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # === TABS FIRST (ABOVE FILTERS) ===
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔄 Demand Planning", "🚚 Operations", "💰 Finance"])
    
    # === GLOBAL FILTERS BELOW TABS ===
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        date_range = st.date_input(
            "📅 Date Range",
            value=(df['Date'].min(), df['Date'].max()),
            min_value=df['Date'].min(),
            max_value=df['Date'].max()
        )
    
    with col2:
        region_filter = st.multiselect(
            "🌍 Region",
            options=sorted(df['Region'].unique()),
            default=sorted(df['Region'].unique())
        )
    
    with col3:
        status_filter = st.multiselect(
            "✅ Delivery Status",
            options=df['Delivery_Status'].unique(),
            default=df['Delivery_Status'].unique()
        )
    
    # Apply filters
    mask = (
        (df['Date'] >= pd.to_datetime(date_range[0])) &
        (df['Date'] <= pd.to_datetime(date_range[1])) &
        (df['Region'].isin(region_filter)) &
        (df['Delivery_Status'].isin(status_filter))
    )
    df_filtered = df[mask].copy()
    
    # === TAB 1: OVERVIEW ===
    with tab1:
        st.header("📊 Executive Overview")
        
        # KPIs
        col1, col2, col3, col4, col5 = st.columns(5)
        kpis = calculate_kpis(df_filtered)
        
        with col1: st.markdown(f'<div class="metric-card"><h3>📋 Orders</h3><h2>{kpis["Total Orders"]:,.0f}</h2></div>', unsafe_allow_html=True)
        with col2: st.markdown(f'<div class="metric-card"><h3>✅ On-Time</h3><h2>{kpis["On-Time Delivery"]}</h2></div>', unsafe_allow_html=True)
        with col3: st.markdown(f'<div class="metric-card"><h3>⏱️ Lead Time</h3><h2>{kpis["Avg Lead Time"]}</h2></div>', unsafe_allow_html=True)
        with col4: st.markdown(f'<div class="metric-card"><h3>📈 Demand</h3><h2>{kpis["Total Demand"]}</h2></div>', unsafe_allow_html=True)
        with col5: st.markdown(f'<div class="metric-card"><h3>📦 Inventory</h3><h2>{kpis["Avg Inventory"]}</h2></div>', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            abc = abc_analysis(df_filtered)
            fig_abc = px.bar(abc.head(15), x='Product_ID', y='Actual_Demand', 
                           color='ABC', title="Top 15 SKUs - ABC Analysis",
                           color_discrete_map={'A': '#FF6B6B', 'B': '#4ECDC4', 'C': '#45B7D1'})
            fig_abc.update_layout(height=350, xaxis_tickangle=45)
            st.plotly_chart(fig_abc, use_container_width=True)
        
        with col_b:
            fig_lead = px.histogram(df_filtered, x='Lead_Time_Days', nbins=30,
                                  color='Delivery_Status', title="Lead Time Distribution")
            fig_lead.update_layout(height=350)
            st.plotly_chart(fig_lead, use_container_width=True)
        
        # INSIGHTS
        st.markdown("""
        <div class="insight-box">
            <h3>💡 Key Insights</h3>
            <ul>
                <li><strong>A-Class SKUs</strong> represent 70% of total demand - focus inventory here</li>
                <li><strong>On-time delivery</strong> at {on_time}% - target improvement if <95%</li>
                <li><strong>Avg lead time</strong> {lead_time} days - optimize suppliers with high variance</li>
            </ul>
        </div>
        """.format(on_time=kpis['On-Time Delivery'], lead_time=kpis['Avg Lead Time']), unsafe_allow_html=True)
    
    # === TAB 2: DEMAND PLANNING ===
    with tab2:
        st.header("🔄 Demand Planning & Forecasting")
        
        col1, col2 = st.columns(2)
        with col1:
            demand_trend = df_filtered.groupby('Date')['Actual_Demand'].sum().reset_index()
            fig_trend = px.line(demand_trend, x='Date', y='Actual_Demand', 
                              title="Daily Demand Trend")
            fig_trend.update_layout(height=350)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            monthly_demand = df_filtered.groupby('Month')['Actual_Demand'].sum().reset_index()
            fig_monthly = px.bar(monthly_demand, x='Month', y='Actual_Demand',
                               title="Monthly Demand Pattern")
            fig_monthly.update_layout(height=350)
            st.plotly_chart(fig_monthly, use_container_width=True)
        
        # INSIGHTS
        st.markdown("""
        <div class="insight-box">
            <h3>💡 Demand Planning Insights</h3>
            <ul>
                <li><strong>Seasonality:</strong> Check monthly patterns for planning cycles</li>
                <li><strong>Trend:</strong> {trend_status} demand trajectory</li>
                <li><strong>Forecasting:</strong> Use 3-month moving average for short-term predictions</li>
            </ul>
        </div>
        """.format(trend_status="Increasing" if demand_trend['Actual_Demand'].iloc[-1] > demand_trend['Actual_Demand'].iloc[0] else "Stable"), unsafe_allow_html=True)
    
    # === TAB 3: OPERATIONS ===
    with tab3:
        st.header("🚚 Operations & Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            region_perf = df_filtered.groupby('Region').agg({
                'Actual_Demand': 'sum',
                'Delivery_Status': lambda x: (x == 'On Time').mean() * 100
            }).round(2)
            fig_region = px.bar(region_perf.reset_index(), x='Region',
                              y=['Actual_Demand', 'Delivery_Status'],
                              title="Regional Performance", barmode='group')
            fig_region.update_layout(height=350)
            st.plotly_chart(fig_region, use_container_width=True)
        
        with col2:
            fig_status = px.pie(df_filtered, names='Delivery_Status', 
                              title="Delivery Status Breakdown")
            st.plotly_chart(fig_status, use_container_width=True)
        
        # INSIGHTS
        st.markdown("""
        <div class="insight-box">
            <h3>💡 Operations Insights</h3>
            <ul>
                <li><strong>Regional Focus:</strong> Prioritize {top_region} operations</li>
                <li><strong>Delivery Bottleneck:</strong> {delayed_pct}% delayed orders need attention</li>
                <li><strong>Capacity Planning:</strong> Balance workload across regions</li>
            </ul>
        </div>
        """.format(top_region=region_perf.index[region_perf['Actual_Demand'].idxmax()],
                   delayed_pct=(df_filtered['Delivery_Status'] == 'Delayed').mean()*100), unsafe_allow_html=True)
    
    # === TAB 4: FINANCE ===
    with tab4:
        st.header("💰 Financial Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_value = px.scatter(df_filtered, x='Actual_Demand', y='Order_Value',
                                 color='Region', size='Lead_Time_Days',
                                 title="Demand vs Order Value")
            fig_value.update_layout(height=350)
            st.plotly_chart(fig_value, use_container_width=True)
        
        with col2:
            numeric_cols = ['Actual_Demand', 'Lead_Time_Days', 'Order_Value', 'Supplier_Rating']
            corr_matrix = df_filtered[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, title="Correlation Matrix", 
                               color_continuous_scale='RdBu_r', aspect="auto")
            fig_corr.update_layout(height=350)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # INSIGHTS
        st.markdown("""
        <div class="insight-box">
            <h3>💡 Financial Insights</h3>
            <ul>
                <li><strong>High-Value Items:</strong> Focus on top 20% SKUs by order value</li>
                <li><strong>Supplier Impact:</strong> Rating correlates with {corr_val:.2f} to lead time</li>
                <li><strong>Cost Optimization:</strong> Reduce lead time = higher order values</li>
            </ul>
        </div>
        """.format(corr_val=corr_matrix.loc['Supplier_Rating', 'Lead_Time_Days']), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
