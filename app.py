import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Reduces warnings on some systems
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Get the directory where app.py is currently located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- INITIALIZE NLTK ---
@st.cache_resource
def init_nltk():
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)

init_nltk()
lemmatizer = WordNetLemmatizer()

# --- 1. CLEANING FUNCTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"[^a-zA-Z!?\s]", " ", text)

    # Lemmatization
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --- 2. LOAD MODEL & TOKENIZER ---
@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'toxicity_model.h5')
    tokenizer_path = os.path.join(BASE_DIR, 'tokenizer.json') 
    
    # 1. Load the Model
    model = tf.keras.models.load_model(model_path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # 2. >>> FIX: Read the file as a raw string using f.read() <<<
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_string = f.read()  # Read as raw text string
        tokenizer = tokenizer_from_json(tokenizer_string) # Pass string to Keras
        
    return model, tokenizer
model, tokenizer = load_assets()
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_LEN = 150 

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Toxicity Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Comment Toxicity Detector")

tab1, tab2, tab3 = st.tabs(["🔍 Live Detection", "📂 Bulk Analysis", "📊 Data Insights"])

with tab1:
    st.header("Real-time Analysis")
    user_input = st.text_area("Enter a comment to analyze:", placeholder="Type something here...", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            prediction = model.predict(padded)[0]
            
            # >>> FIX: Check if ANY category crosses 50%, not just base toxic <<<
            any_toxic = np.any(prediction > 0.5)
            max_score = np.max(prediction)
            
            if any_toxic:
                st.error(f"⚠️ High Toxicity Warning (Max Flag: {max_score:.2%})")
            else:
                st.success("✅ Comment appears to be safe and clean.")

            st.write("---")
            # Show individual category breakdown
            for i, cat in enumerate(categories):
                col1, col2 = st.columns([2, 8])
                with col1:
                    st.write(f"**{cat.replace('_', ' ').title()}**")
                with col2:
                    st.progress(float(prediction[i]))
                    st.caption(f"{prediction[i]:.2%}")

with tab2:
    st.header("Bulk Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox("Select Sample Type:", ["First Rows (Head)", "Last Rows (Tail)", "Random Sample"])
    with col2:
        num_rows = st.number_input("Number of rows:", min_value=1, max_value=500, value=20)

    if st.button("Run Bulk Analysis"):
        test_csv_path = os.path.join(BASE_DIR, 'test.csv')
        try:
            with st.spinner(f"Processing data from test.csv..."):
                # >>> FIX: Use chunking/tail operators to survive multiline text breaks <<<
                if mode == "First Rows (Head)":
                    df = pd.read_csv(test_csv_path, nrows=num_rows)
                elif mode == "Last Rows (Tail)":
                    df = pd.read_csv(test_csv_path).tail(num_rows)
                else: 
                    df = pd.read_csv(test_csv_path).sample(n=num_rows)

                # Process and Predict
                df['cleaned'] = df['comment_text'].apply(clean_text)
                seqs = tokenizer.texts_to_sequences(df['cleaned'])
                padded = pad_sequences(seqs, maxlen=MAX_LEN)
                preds = model.predict(padded)

                # Format Results
                for i, cat in enumerate(categories):
                    df[cat] = preds[:, i]
                    df[cat] = df[cat].apply(lambda x: f"{x:.2%}")

                st.subheader(f"Results: {mode}")
                st.dataframe(df.drop(columns=['cleaned'])) # Hide cleaning step from end users

                # Download Options
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download This Sample", csv, "sample_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error loading file: {e}. Please check that 'test.csv' is placed inside: {BASE_DIR}")

with tab3:
    st.header("Data Insights (EDA)")
    train_csv_path = os.path.join(BASE_DIR, 'train.csv')
    try:
        @st.cache_data
        def load_eda_data(path):
            return pd.read_csv(path, usecols=categories)

        df_train = load_eda_data(train_csv_path)

        total_samples = len(df_train)
        toxic_count = df_train['toxic'].sum()
        toxic_pct = (toxic_count / total_samples) * 100
        st.metric("Overall Toxicity Rate in Dataset", f"{toxic_pct:.2f}%")

        st.subheader("Category Distribution Breakdown")
        st.bar_chart(df_train.sum())

        st.subheader("Text Preprocessing Visual Sample")
        df_sample = pd.read_csv(train_csv_path, usecols=['comment_text'], nrows=5)
        df_sample['processed_result'] = df_sample['comment_text'].apply(clean_text)
        st.table(df_sample)

    except FileNotFoundError:
        st.error(f"Missing 'train.csv'. Please place it inside: {BASE_DIR}")