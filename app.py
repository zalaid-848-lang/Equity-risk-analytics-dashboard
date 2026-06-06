import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Equity Risk Analytics Dashboard", layout="wide")

st.title("Equity Portfolio Risk Clustering & Anomaly Detection")

@st.cache_data
def load_data():
    umap_df = pd.read_csv('data/processed/umap_clusters.csv', index_col=0)
    feature_matrix = pd.read_csv('data/processed/risk_features.csv', index_col=0)
    return umap_df, feature_matrix

umap_df, feature_matrix = load_data()

# Sidebar filters
st.sidebar.header("Filters")
show_outliers = st.sidebar.checkbox("Show Outliers (Cluster -1)", value=True)

# Filter data
filtered_umap = umap_df.copy()
if not show_outliers:
    filtered_umap = filtered_umap[filtered_umap['Cluster'] != -1]

# Tabs
tab1, tab2, tab3 = st.tabs(["UMAP Cluster Explorer", "Anomaly Detector", "Cluster Profiles"])

with tab1:
    st.subheader("2D UMAP Projection - Risk-Based Clustering")
    fig = px.scatter(
        filtered_umap,
        x='UMAP1',
        y='UMAP2',
        color='Cluster',
        hover_name=filtered_umap.index,
        title="Stocks Colored by Risk Profile Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Anomaly Detection Explorer")
    threshold = st.slider("Anomaly Threshold (z-score)", 1.0, 3.0, 2.0)
    anomalies = umap_df[umap_df['Anomaly_Score'] > threshold]
    st.write(f"Found **{len(anomalies)}** anomalies above threshold")
    st.dataframe(anomalies[['Cluster', 'Anomaly_Score']])

with tab3:
    st.subheader("Risk Metric Comparison by Cluster")
    cluster_avg = feature_matrix.groupby(umap_df['Cluster']).mean()
    fig = px.bar(
        cluster_avg.reset_index(),
        x='Cluster',
        y=['Sharpe_Ratio', 'Volatility', 'Sortino_Ratio'],
        barmode='group',
        title="Average Risk Metrics per Cluster"
    )
    st.plotly_chart(fig, use_container_width=True)
