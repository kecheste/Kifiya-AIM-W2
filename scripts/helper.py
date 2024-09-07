import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

def load_data_from_postgres(query):
   
    try:
        load_dotenv()

        # Retrieve database connection parameters from environment variables
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_USER = os.getenv("DB_USER")

        # Establish a connection to the PostgreSQL database
        connection = psycopg2.connect(
            host=DB_HOST,       # Database host
            port=DB_PORT,       # Database port
            database=DB_NAME,   # Database name
            password=DB_PASSWORD,# Database password
            user=DB_USER        # Database user
        )
        
        # Execute the query and load the result into a DataFrame
        df = pd.read_sql_query(query, connection)
        
        # Close the database connection
        connection.close()
        
        return df  # Return the DataFrame containing the query results

    except Exception as e:
        # Print an error message if an exception occurs
        print(f"An Error Occurred: {e}")