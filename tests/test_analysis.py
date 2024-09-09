import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import unittest

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

class TestAnalysis(unittest.TestCase):

    def test_engagement_metrics(self):
        engagement_metrics = df.groupby('MSISDN').agg(
            session_frequency=('bearer id', 'count'),
            total_duration=('Dur. (ms)', 'sum'),
            total_dl=('Total DL (Bytes)', 'sum'),
            total_ul=('Total UL (Bytes)', 'sum')
        )
        
        engagement_metrics['total_traffic'] = engagement_metrics['total_dl'] + engagement_metrics['total_ul']
        
        engagement_metrics = engagement_metrics.reset_index()

        self.assertEqual(engagement_metrics.shape[0], 3)
        self.assertIn('total_traffic', engagement_metrics.columns)

    def test_cluster_assignment(self):
        scaler = StandardScaler()
        normalized_metrics = scaler.fit_transform(df[['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)']])
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(normalized_metrics)

        self.assertEqual(len(clusters), df.shape[0])

    def test_top_users_per_app(self):
        top_users_per_app = df.groupby(['application', 'MSISDN']).agg(
            total_dl=('Total DL (Bytes)', 'sum'),
            total_ul=('Total UL (Bytes)', 'sum')
        )

        top_users_per_app['total_traffic'] = top_users_per_app['total_dl'] + top_users_per_app['total_ul']

        top_users_per_app = top_users_per_app.reset_index()

        self.assertEqual(top_users_per_app.shape[0], 3)

if __name__ == "__main__":
    unittest.main()
