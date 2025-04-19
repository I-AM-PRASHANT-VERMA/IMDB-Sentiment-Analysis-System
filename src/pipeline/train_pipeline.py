from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging

def train_pipeline():
    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        (X_train, y_train, X_test, y_test) = data_transformation.initiate_data_transformation(train_path, test_path)

        # Model Training
        model_trainer = ModelTrainer()
        test_accuracy = model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        print(f"Model trained with test accuracy: {test_accuracy:.4f}")
        return test_accuracy

    except Exception as e:
        raise e

if __name__ == "__main__":
    train_pipeline()