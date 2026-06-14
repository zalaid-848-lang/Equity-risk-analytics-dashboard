# app_enhanced.py - ENHANCED EQUITY RISK ANALYTICS DASHBOARD

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Enhanced Equity Risk Analytics", layout="wide")

st.title("🚀 Enhanced Equity Portfolio Risk Clustering & Anomaly Detection")
st.markdown("*Version 2.0 - Expanded Stock Universe for Robust Clustering*")

@st.cache_data
def load_enhanced_data():
    """Load enhanced dataset (60+ stocks)"""
    try:
        umap_df = pd.read_csv('data/enhanced/umap_clusters_enhanced.csv', index_col=0)
        feature_matrix = pd.read_csv('data/enhanced/risk_features_enhanced.csv', index_col=0)
        return umap_df, feature_matrix
    except FileNotFoundError:
        st.warning("⚠️ Enhanced data not found. Trying original data...")
        try:
            umap_df = pd.read_csv('data/processed/umap_clusters.csv', index_col=0)
            feature_matrix = pd.read_csv('data/processed/risk_features.csv', index_col=0)
            st.info("📊 Showing original 10-stock analysis. Run enhanced notebook for 60+ stocks.")
            return umap_df, feature_matrix
        except FileNotFoundError:
            st.error("""
            ❌ No data files found!
            
            Please run either:
            1. Enhanced notebook to generate data/enhanced/ files
            2. Original notebook to generate data/processed/ files
            """)
            return None, None

# Load data
umap_df, feature_matrix = load_enhanced_data()

if umap_df is not None and feature_matrix is not None:
    # Sidebar
    st.sidebar.header("🎛️ Dashboard Controls")
    
    # Filters
    show_outliers = st.sidebar.checkbox("Show Outliers (Cluster -1)", value=True)
    anomaly_threshold = st.sidebar.slider("Anomaly Threshold (z-score)", 1.0, 3.0, 2.0, 0.1)
    
    # Filter data
    filtered_umap = umap_df.copy()
    if not show_outliers:
        filtered_umap = filtered_umap[filtered_umap['Cluster'] != -1]
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Stocks", len(umap_df))
    with col2:
        n_clusters = len(set(umap_df['Cluster'])) - (1 if -1 in umap_df['Cluster'] else 0)
        st.metric("Risk Clusters Found", n_clusters)
    with col3:
        n_outliers = sum(umap_df['Cluster'] == -1)
        st.metric("Outlier Stocks", n_outliers)
    with col4:
        high_anomaly = sum(umap_df['Anomaly_Score'] > anomaly_threshold)
        st.metric(f"Anomalies (>{anomaly_threshold})", high_anomaly)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔬 UMAP Explorer", "⚠️ Anomaly Detector", "📊 Cluster Profiles"])
    
    with tab1:
        st.subheader("2D UMAP Projection - Risk-Based Clustering")
        
        color_by = st.radio("Color by:", ["Cluster", "Anomaly Score"], horizontal=True)
        
        if color_by == "Cluster":
            fig = px.scatter(
                filtered_umap,
                x='UMAP1',
                y='UMAP2',
                color='Cluster',
                hover_name=filtered_umap.index,
                title="Stocks Colored by Risk Profile Cluster",
                color_continuous_scale='Viridis'
            )
        else:
            fig = px.scatter(
                filtered_umap,
                x='UMAP1',
                y='UMAP2',
                color='Anomaly_Score',
                hover_name=filtered_umap.index,
                title="Stocks Colored by Anomaly Score",
                color_continuous_scale='RdYlGn_r'
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Anomaly Detection Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            anomalies = umap_df[umap_df['Anomaly_Score'] > anomaly_threshold]
            st.metric("Anomalies Found", len(anomalies))
        
        with col2:
            st.metric("Avg Anomaly Score", f"{umap_df['Anomaly_Score'].mean():.3f}")
        
        # Anomaly distribution
        fig = px.histogram(
            umap_df,
            x='Anomaly_Score',
            nbins=20,
            title="Distribution of Anomaly Scores",
            labels={'Anomaly_Score': 'Anomaly Score (z-score)', 'count': 'Number of Stocks'}
        )
        fig.add_vline(x=anomaly_threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly table
        if len(anomalies) > 0:
            st.dataframe(anomalies[['Cluster', 'Anomaly_Score']].sort_values('Anomaly_Score', ascending=False))
    
    with tab3:
        st.subheader("Risk Metric Comparison by Cluster")
        
        cluster_avg = feature_matrix.groupby(umap_df['Cluster']).mean()
        
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            ['Sharpe_Ratio', 'Volatility', 'Sortino_Ratio', 'Max_Drawdown', 'VaR_95'],
            default=['Sharpe_Ratio', 'Volatility', 'Sortino_Ratio']
        )
        
        if selected_metrics:
            fig = px.bar(
                cluster_avg.reset_index(),
                x='Cluster',
                y=selected_metrics,
                barmode='group',
                title="Average Risk Metrics by Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Complete Risk Metrics")
        combined_df = feature_matrix.copy()
        combined_df['Cluster'] = umap_df['Cluster']
        combined_df['Anomaly_Score'] = umap_df['Anomaly_Score']
        st.dataframe(combined_df.sort_values('Cluster').round(4))

st.sidebar.markdown("---")
st.sidebar.info("Enhanced dashboard with expanded stock universe")
