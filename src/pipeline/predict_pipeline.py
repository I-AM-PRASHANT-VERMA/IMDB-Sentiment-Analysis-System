import os
import sys
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

    def predict(self, text):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            # Transform input text
            text_features = preprocessor.transform([text]).toarray()

            # Make prediction
            prediction = model.predict(text_features)
            probability = model.predict_proba(text_features)

            # Return sentiment and confidence
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            confidence = max(probability[0])

            return sentiment, confidence

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    text = "This movie was fantastic! The acting was great and the story was compelling."
    pipeline = PredictPipeline()
    sentiment, confidence = pipeline.predict(text)
    print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")