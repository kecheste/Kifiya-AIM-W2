import logging
import psycopg2
import pandas as pd
import os

logging.basicConfig(level=logging.INFO)

def load_data_from_postgres(query):
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD")
        )

        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        logging.info(f"An Error Occurred: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()
