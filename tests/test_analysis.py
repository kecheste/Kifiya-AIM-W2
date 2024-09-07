import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pytest

# Sample data for testing
sample_data = {
    'MSISDN': [1, 1, 2, 2, 3, 3, 3],
    'bearer id': [1, 2, 1, 2, 1, 2, 3],
    'Dur. (ms)': [100, 200, 150, 250, 300, 350, 400],
    'Total DL (Bytes)': [500, 600, 700, 800, 900, 1000, 1100],
    'Total UL (Bytes)': [100, 200, 300, 400, 500, 600, 700],
    'application': ['App1', 'App1', 'App2', 'App2', 'App3', 'App3', 'App3']
}

df = pd.DataFrame(sample_data)

def test_engagement_metrics():
    # Calculate engagement metrics
    engagement_metrics = df.groupby('MSISDN').agg(
        session_frequency=('bearer id', 'count'),
        total_duration=('Dur. (ms)', 'sum'),
        total_traffic=('Total DL (Bytes)', 'sum') + ('Total UL (Bytes)', 'sum')
    ).reset_index()

    assert engagement_metrics.shape[0] == 3
    assert 'total_traffic' in engagement_metrics.columns

def test_cluster_assignment():
    # Normalize and run KMeans
    scaler = StandardScaler()
    normalized_metrics = scaler.fit_transform(df[['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(normalized_metrics)

    assert len(clusters) == df.shape[0]

def test_top_users_per_app():
    top_users_per_app = df.groupby(['application', 'MSISDN']).agg(
        total_traffic=('Total DL (Bytes)', 'sum') + ('Total UL (Bytes)', 'sum')
    ).reset_index()

    assert top_users_per_app.shape[0] == 6

if __name__ == "__main__":
    pytest.main()