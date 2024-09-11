import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import psycopg2
from dotenv import load_dotenv

load_dotenv()
def calculate_scores(df, engagement_clusters, experience_clusters):
    
    engagement_centroid = engagement_clusters[['total_DL', 'total_UL']].mean().values.reshape(1, -1)
    experience_centroid = experience_clusters[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']].mean().values.reshape(1, -1)

    df['Engagement Score'] = pairwise_distances(df[['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']], engagement_centroid).flatten()
    
    df['Experience Score'] = pairwise_distances(df[['Avg RTT DL (ms)', 'Avg RTT UL (ms)']], experience_centroid).flatten()
    result_df = df[['MSISDN/Number','Engagement Score', 'Experience Score']]
    return df,result_df

def calculate_satisfaction(df):
    """
    Calculate satisfaction scores and return the top 10 customers
    based on their satisfaction scores.
    """
    df['Satisfaction Score'] = (df['Engagement Score'] + df['Experience Score']) / 2
    top_customers = df.nlargest(10, 'Satisfaction Score')
    return top_customers[['MSISDN/Number', 'Satisfaction Score']]

def regression_model(df):
    """
    Build a linear regression model to predict satisfaction scores
    based on engagement and experience scores. Return the model and its MSE.
    """
    X = df[['Engagement Score', 'Experience Score']]
    y = df['Satisfaction Score']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train) 
    
    y_pred = model.predict(X_test)  # Make predictions
    mse = mean_squared_error(y_test, y_pred)  # Calculate mean squared error
    
    return model, mse,X_test,X_train

def kmeans_clustering(df):
    """
    Perform K-Means clustering on engagement and experience scores
    and assign the cluster labels to the DataFrame.
    """
    kmeans = KMeans(n_clusters=2, random_state=42)
    df['Engagement & Experience Cluster'] = kmeans.fit_predict(df[['Engagement Score', 'Experience Score']])
    return df

def aggregate_scores(df):
    """
    Aggregate average satisfaction and experience scores per cluster.
    """
    cluster_summary = df.groupby('Engagement & Experience Cluster').agg({
        'Satisfaction Score': 'mean',
        'Experience Score': 'mean'
    }).reset_index()
    return cluster_summary

def export_to_postgresql(df):
    """
    Export engagement, experience, and satisfaction scores to a PostgreSQL database.
    """
    connection = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    
    cursor = connection.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_scores (
            IMSI FLOAT,
            EngagementScore FLOAT,
            ExperienceScore FLOAT,
            SatisfactionScore FLOAT
        )
    ''')

    for _, row in df.iterrows():
        cursor.execute('''
            INSERT INTO user_scores (IMSI, EngagementScore, ExperienceScore, SatisfactionScore) 
            VALUES (%s, %s, %s, %s)
        ''', (row['IMSI'], row['Engagement Score'], row['Experience Score'], row['Satisfaction Score']))

    connection.commit()  # Commit the transaction
    cursor.close()  # Close the cursor
    connection.close()  # Close the connection


def track_model_deployment(model, X_train, X_test, mse):
    """
    Track model parameters and metrics using MLflow.
    """
    mlflow.start_run()  # Start a new MLflow run
    mlflow.log_param("model_type", "Linear Regression")  # Log model type
    mlflow.log_param("train_size", len(X_train))  # Log training size
    mlflow.log_param("test_size", len(X_test))  # Log testing size
    mlflow.log_metric("mse", mse)  # Log mean squared error

    # Create an input example
    input_example = np.array([X_train[0]])  # Example input from training data

    # Save the trained model with input example
    mlflow.sklearn.log_model(model, "model", input_example=input_example)
    mlflow.end_run()  # End the MLflow run