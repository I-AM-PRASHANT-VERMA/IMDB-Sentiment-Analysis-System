import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import sys

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def get_data_transformer(self):
        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            return vectorizer
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer()

            # Fit and transform training data
            input_feature_train_arr = preprocessing_obj.fit_transform(train_df['text']).toarray()
            target_feature_train_arr = train_df['label'].values

            # Transform test data
            input_feature_test_arr = preprocessing_obj.transform(test_df['text']).toarray()
            target_feature_test_arr = test_df['label'].values

            logging.info("Applying preprocessing object on training and testing datasets")

            save_object(
                file_path=self.preprocessor_path,
                obj=preprocessing_obj
            )

            return (
                input_feature_train_arr,
                target_feature_train_arr,
                input_feature_test_arr,
                target_feature_test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)