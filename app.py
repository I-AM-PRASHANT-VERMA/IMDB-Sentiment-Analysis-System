from flask import Flask, render_template, request, redirect, url_for
from src.pipeline.predict_pipeline import PredictPipeline
from src.logger import logging

app = Flask(__name__)

@app.route('/')
def index():  # Renamed from home() to avoid confusion
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            text = request.form['text']
            if not text.strip():
                return render_template('home.html', error="Please enter a review")

            pipeline = PredictPipeline()
            sentiment, confidence = pipeline.predict(text)

            return render_template(
                'home.html',
                prediction_text=f"Sentiment: {sentiment}",
                confidence=f"Confidence: {confidence:.2%}",
                user_review=text
            )
        except Exception as e:
            logging.error(str(e))
            return render_template('home.html', error="An error occurred during prediction")
    # Changed from redirect to render_template for GET requests
    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)