import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class ExperienceAnalytics:
    def __init__(self, df):
        self.df = df
        
    def aggregate_data(df):
        df['TCP DL Retrans. Vol (Bytes)'].fillna(df['TCP DL Retrans. Vol (Bytes)'].mean(), inplace=True)
        df['TCP UL Retrans. Vol (Bytes)'].fillna(df['TCP UL Retrans. Vol (Bytes)'].mean(), inplace=True)
        
        df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
        df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean(), inplace=True)

        aggregated = df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'TCP UL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Handset Type': 'first'
        }).reset_index()

        return aggregated

    def compute_values(df):
        tcp_dl_top = df['TCP DL Retrans. Vol (Bytes)'].nlargest(10).reset_index(drop=True)
        tcp_dl_bottom = df['TCP DL Retrans. Vol (Bytes)'].nsmallest(10).reset_index(drop=True)
        tcp_dl_most_frequent = df['TCP DL Retrans. Vol (Bytes)'].mode().reset_index(drop=True)

        if len(tcp_dl_most_frequent) < 10:
            tcp_dl_most_frequent = tcp_dl_most_frequent.tolist() + [tcp_dl_most_frequent[0]] * (10 - len(tcp_dl_most_frequent))

        tcp_ul_top = df['TCP UL Retrans. Vol (Bytes)'].nlargest(10).reset_index(drop=True)
        tcp_ul_bottom = df['TCP UL Retrans. Vol (Bytes)'].nsmallest(10).reset_index(drop=True)
        tcp_ul_most_frequent = df['TCP UL Retrans. Vol (Bytes)'].mode().reset_index(drop=True)

        if len(tcp_ul_most_frequent) < 10:
            tcp_ul_most_frequent = tcp_ul_most_frequent.tolist() + [tcp_ul_most_frequent[0]] * (10 - len(tcp_ul_most_frequent))

        tcp_dl_summary = pd.DataFrame({
            'Top 10': tcp_dl_top,
            'Bottom 10': tcp_dl_bottom,
            'Most Frequent': tcp_dl_most_frequent
        })

        tcp_ul_summary = pd.DataFrame({
            'Top 10': tcp_ul_top,
            'Bottom 10': tcp_ul_bottom,
            'Most Frequent': tcp_ul_most_frequent
        })


        return {
            'TCP DL Summary': tcp_dl_summary,
            'TCP UL Summary': tcp_ul_summary
        }
        
    def distribution_per_handset(df, top_n=10):
        tcp_dl_distribution = df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean().reset_index()
        tcp_ul_distribution = df.groupby('Handset Type')['TCP UL Retrans. Vol (Bytes)'].mean().reset_index()

        tcp_dl_distribution = tcp_dl_distribution.nlargest(top_n, 'TCP DL Retrans. Vol (Bytes)')
        tcp_ul_distribution = tcp_ul_distribution.nlargest(top_n, 'TCP UL Retrans. Vol (Bytes)')

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.barplot(x='Handset Type', y='TCP DL Retrans. Vol (Bytes)', data=tcp_dl_distribution)
        plt.title('Average TCP DL Retransmission per Handset Type')
        plt.xticks(rotation=90)

        plt.subplot(1, 2, 2)
        sns.barplot(x='Handset Type', y='TCP UL Retrans. Vol (Bytes)', data=tcp_ul_distribution)
        plt.title('Average TCP UL Retransmission per Handset Type')
        plt.xticks(rotation=90)

        plt.tight_layout()
        plt.show()

    def perform_clustering(df):
        features = df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg RTT UL (ms)']]
        df = df.drop(columns=['Handset Type'])    
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(features)

        cluster_counts = df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        
        return df,cluster_counts

    def create_dashboard():
        pass
