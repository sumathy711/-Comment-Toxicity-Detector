# -Comment-Toxicity-Detector
A deep learning-based web application that classifies online comments into six toxicity categories: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate.

## 🚀 Features
- **Live Detection**: Analyze single sentences in real-time.
- **Bulk Analysis**: Process multiple comments from a CSV file (Head/Tail/Random).
- **Data Insights**: Visual exploration of the training dataset (EDA).

## 🛠️ Tech Stack
- **Deep Learning**: TensorFlow/Keras (Bidirectional LSTM)
- **NLP**: NLTK (Lemmatization, Tokenization)
- **Web App**: Streamlit
- **Data**: Pandas, NumPy

  # Run the app:
  streamlit run app.py
   

## 📊 Model Performance
- **Algorithm**: Bidirectional LSTM with class weighting to handle imbalanced data.
- **Loss Function**: Binary Crossentropy.
- **Preprocessing**: Custom cleaning pipeline including contraction expansion and lemmatization.
