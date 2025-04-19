Here's the polished GitHub README.md without code blocks, using clean tables and concise formatting:

---

# 🎬 IMDB Sentiment Analysis System

![Python](https://img.shields.io/badge/Python-3.8-blue) 
![Flask](https://img.shields.io/badge/Flask-2.0.1-lightgrey) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange)

## 🌟 Project Overview  
**Problem**: Classifying movie reviews as positive/negative with limited labeled data  
**Solution**: Hybrid NLP pipeline combining TF-IDF vectorization with Logistic Regression  
**Key Achievement**: 87.8% accuracy with interpretable results  

## 📊 Dataset  
**Source**: IMDB Dataset from HuggingFace (50,000 reviews)  
- **Balance**: Perfect 50/50 positive/negative split  
- **Preprocessing**: Lowercasing + punctuation removal  
- **Size**: 80MB compressed (250MB uncompressed)  

## 🧠 Why This Architecture?  

| Component | Choice | Rationale |  
|-----------|--------|-----------|  
| Text Vectorization | TF-IDF | Better than word2vec for small/medium datasets |  
| Classifier | Logistic Regression | 3x faster inference than neural nets |  
| Benchmark | KNN Baseline | Shows 12% accuracy improvement |  

**Key Advantage**: Explains predictions through feature weights  

## ⚙️ Hyperparameter Tuning  

### Optimal Configuration  
- **TF-IDF**: max_features=5000, ngram_range=(1,2)  
- **Logistic Regression**: C=0.5, penalty='l2'  

### Performance Impact  

| Metric | Before Tuning | After Tuning |  
|--------|--------------|-------------|  
| Accuracy | 85.2% | **87.8%** |  
| Speed | 120ms | 68ms |  
| Model Size | 42MB | 19MB |  

## 🚀 Getting Started  

1. **Installation**  
   - Create Conda env: `conda create -n sentiment python=3.8`  
   - Install packages: `pip install -r requirements.txt`  

2. **Training**  
   - Run: `python -m src.pipeline.train_pipeline`  
   - Output: Saves model to `artifacts/`  

3. **Web App**  
   - Start: `python app.py`  
   - Access: `http://localhost:5000`  

## 📈 Results  

| Model | Accuracy | Precision | Recall |  
|-------|----------|-----------|--------|  
| Our Model | 87.8% | 88.1% | 87.5% |  
| KNN Baseline | 75.2% | 74.9% | 75.3% |  

## 💡 Future Roadmap  

| Feature | Status | Target Impact |  
|---------|--------|---------------|  
| BERT Hybrid | Research | +3-5% accuracy |  
| User Accounts | Planned | Save history |  
| Real-time API | Backlog | Mobile support |  

---
