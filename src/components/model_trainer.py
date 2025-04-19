import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.metrics_path = os.path.join('artifacts', 'training_metrics.txt')

    def evaluate_model(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Training Logistic Regression model")
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate
            train_accuracy, train_precision, train_recall, train_f1 = self.evaluate_model(y_train, y_train_pred)
            test_accuracy, test_precision, test_recall, test_f1 = self.evaluate_model(y_test, y_test_pred)

            # Save metrics
            metrics = {
                'Training Accuracy': train_accuracy,
                'Training Precision': train_precision,
                'Training Recall': train_recall,
                'Training F1 Score': train_f1,
                'Test Accuracy': test_accuracy,
                'Test Precision': test_precision,
                'Test Recall': test_recall,
                'Test F1 Score': test_f1
            }

            with open(self.metrics_path, 'w') as f:
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")

            # Save model
            save_object(file_path=self.model_path, obj=model)

            logging.info("Model training completed")
            return test_accuracy

        except Exception as e:
            raise CustomException(e, sys)