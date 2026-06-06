# Equity Risk Analytics Dashboard

## Overview
An interactive dashboard for equity portfolio risk analysis using unsupervised machine learning. This tool clusters stocks based on risk profiles and detects anomalies using UMAP dimensionality reduction and HDBSCAN clustering.

## Author
zalaid-848-lang

## Features
- **Risk-Based Clustering**: Groups stocks with similar risk characteristics
- **Anomaly Detection**: Identifies outlier stocks with unusual risk patterns
- **Interactive Visualizations**: Explore clusters and adjust anomaly thresholds in real-time
- **Risk Metrics Analysis**: Compare Sharpe ratio, volatility, and Sortino ratio across clusters

## Tech Stack
- **Data Collection**: yfinance, pandas
- **Machine Learning**: UMAP, HDBSCAN, scikit-learn
- **Visualization**: Plotly, Streamlit
- **Language**: Python 3.8+

## Installation

### 1. Clone the repository
```
git clone https://github.com/zalaid-848-lang/equity-risk-analytics-dashboard.git
cd equity-risk-analytics-dashboard
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the dashboard
```
streamlit run app.py
```

## Project Structure
```
equity-risk-analytics-dashboard/
├── app.py                 # Main Streamlit dashboard
├── data/
│   └── processed/         # Processed risk metrics and cluster data
│       ├── risk_features.csv
│       └── umap_clusters.csv
├── requirements.txt       # Python dependencies
├── .gitignore            # Files to exclude from Git
└── README.md             # Project documentation
```

## Methodology

### 1. Feature Engineering
Calculated 8 risk metrics for each stock:
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Skewness and kurtosis
- Sortino ratio
- Value at Risk (VaR)

### 2. Dimensionality Reduction
Applied UMAP to reduce 8-dimensional risk space to 2D for visualization while preserving local and global structure.

### 3. Clustering
Used HDBSCAN for density-based clustering that automatically identifies outliers as noise points.

### 4. Anomaly Detection
Calculated anomaly scores based on Sharpe ratio deviation from cluster means.

## Results
- Identified distinct risk clusters across stocks
- Successfully flagged outlier stocks as anomalies
- Revealed sector-specific risk patterns

## License
MIT