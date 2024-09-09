import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from scripts.helper import load_data_from_postgres
import os

class TestLoadDataFromPostgres(unittest.TestCase):
    @patch('scripts.helper.pd.read_sql_query')
    @patch('scripts.helper.psycopg2.connect')
    def test_load_data_from_postgres_success(self, mock_connect, mock_read_sql_query):
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mock_df = pd.DataFrame({'col1': [1, 2], 'col2': ['A', 'B']})
        mock_read_sql_query.return_value = mock_df

        query = "SELECT * FROM xdr_data"
        result_df = load_data_from_postgres(query)

        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT")
        DB_NAME = os.getenv("DB_NAME")
        DB_PASSWORD = os.getenv("DB_PASSWORD")
        DB_USER = os.getenv("DB_USER")

        mock_connect.assert_called_once_with(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            password=DB_PASSWORD,
            user=DB_USER
        )

        mock_read_sql_query.assert_called_once_with(query, mock_conn)

        pd.testing.assert_frame_equal(result_df, mock_df)

        mock_conn.close.assert_called_once()

    @patch('scripts.helper.psycopg2.connect', side_effect=Exception('Connection failed'))
    def test_load_data_from_postgres_exception(self, mock_connect):
        query = "SELECT * FROM xdr_data"
        
        with self.assertLogs(level='INFO') as log:
            result = load_data_from_postgres(query)

            self.assertIsNone(result)

            log_output = log.output
            self.assertTrue(any('An Error Occurred: Connection failed' in message for message in log_output))

if __name__ == '__main__':
    unittest.main()
