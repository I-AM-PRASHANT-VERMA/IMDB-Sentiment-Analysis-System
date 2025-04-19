from datasets import load_dataset
import pandas as pd
import os
from src.logger import logging
from src.exception import CustomException
import sys

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join('artifacts', 'data.csv')
        self.train_data_path = os.path.join('artifacts', 'train.csv')
        self.test_data_path = os.path.join('artifacts', 'test.csv')

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Load IMDB dataset
            dataset = load_dataset('imdb')
            df_train = pd.DataFrame(dataset['train'])
            df_test = pd.DataFrame(dataset['test'])
            
            # Combine for raw data
            df_raw = pd.concat([df_train, df_test])
            
            # Save data
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df_raw.to_csv(self.raw_data_path, index=False)
            df_train.to_csv(self.train_data_path, index=False)
            df_test.to_csv(self.test_data_path, index=False)
            
            logging.info("Data ingestion completed")
            
            return (
                self.train_data_path,
                self.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)