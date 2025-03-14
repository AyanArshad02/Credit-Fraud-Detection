# data ingestion
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # Adjust if needed
from src.logger import logging
from sklearn.utils import resample
from src.connections import s3_connection
from dotenv import load_dotenv
load_dotenv()


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data by handling fraud detection."""
    try:
        logging.info("Pre-processing fraud detection data...")
        
        if 'Class' not in df.columns:
            raise KeyError("Missing required column: 'Class'")
        
        normal = df[df['Class'] == 0]  # Get all normal transactions
        fraud = df[df['Class'] == 1]   # Get all fraud transactions
        
        logging.info(f"Original normal transactions: {normal.shape}, fraud transactions: {fraud.shape}")
        
        normal_under_sample = resample(normal, replace=False, n_samples=len(fraud), random_state=27)
        df_new = pd.concat([normal_under_sample, fraud])
        
        logging.info(f"New dataset shape after undersampling: {df_new.shape}")
        
        return df_new
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.info('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # params = load_params(params_path='params.yaml')
        # test_size = params['data_ingestion']['test_size']
        test_size = 0.2

        bucket_name = os.getenv("BUCKET_NAME")
        aws_access_key = os.getenv("AWS_ACCESS_KEY")
        aws_secret_key = os.getenv("AWS_SECRET_KEY")
        
        # df = load_data(data_url='notebooks/creditcard.csv') # to fetch data locally
        s3 = s3_connection.s3_operations(bucket_name, aws_access_key, aws_secret_key)
        df = s3.fetch_file_from_s3("creditcard.csv")



        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()