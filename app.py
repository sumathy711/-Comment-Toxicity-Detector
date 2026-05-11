import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Reduces warnings on some systems
import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    # Load the model and re-compile to ensure metrics/optimizer are active
    model = tf.keras.models.load_model('toxicity_model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    return model, tokenizer

model, tokenizer = load_assets()
categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_LEN = 150 # Matches your successful training and test results

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Toxicity Detector", page_icon="🛡️")
st.title("🛡️ Comment Toxicity Detector")

tab1, tab2, tab3 = st.tabs(["🔍 Live Detection", "📂 Bulk Analysis", "📊 Data Insights"])

with tab1:
    st.header("Real-time Analysis")
    user_input = st.text_area("Enter a comment to analyze:", placeholder="Type something here...")
    
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text first.")
        else:
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            prediction = model.predict(padded)[0]
            
            # Highlight Overall Status
            is_toxic = prediction[0] > 0.5
            if is_toxic:
                st.error(f"⚠️ High Toxicity Warning ({prediction[0]:.2%})")
            else:
                st.success("✅ Comment appears to be safe.")

            st.write("---")
            # Show individual category breakdown
            for i, cat in enumerate(categories):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**{cat.replace('_', ' ').title()}**")
                with col2:
                    st.progress(float(prediction[i]))
                    st.caption(f"{prediction[i]:.2%}")

with tab2:
    st.header("Bulk Analysis")
    
    # 1. Setup Selection Options
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox("Select Sample Type:", ["First Rows (Head)", "Last Rows (Tail)", "Random Sample"])
    with col2:
        num_rows = st.number_input("Number of rows:", min_value=1, max_value=500, value=20)

    if st.button("Run Bulk Analysis"):
        try:
            with st.spinner(f"Fetching {num_rows} rows from test.csv..."):
                # 2. Smart Loading based on selection
                if mode == "First Rows (Head)":
                    df = pd.read_csv('test.csv', nrows=num_rows)
                
                elif mode == "Last Rows (Tail)":
                    # To get the tail without loading everything, we skip rows
                    total_rows = sum(1 for line in open('test.csv', encoding='utf-8')) - 1
                    df = pd.read_csv('test.csv', skiprows=range(1, total_rows - num_rows + 1))
                
                else: # Random Sample
                    # Load a larger chunk and shuffle to keep it memory efficient
                    df = pd.read_csv('test.csv', nrows=2000).sample(n=num_rows)

                # 3. Process and Predict
                df['cleaned'] = df['comment_text'].apply(clean_text)
                seqs = tokenizer.texts_to_sequences(df['cleaned'])
                padded = pad_sequences(seqs, maxlen=150)
                preds = model.predict(padded)

                # 4. Format Results
                for i, cat in enumerate(categories):
                    df[cat] = preds[:, i]
                    # Optional: Format as percentage for the dataframe view
                    df[cat] = df[cat].apply(lambda x: f"{x:.2%}")

                st.subheader(f"Results: {mode}")
                st.dataframe(df)

                # 5. Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download This Sample", csv, "sample_results.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}. Ensure test.csv is in the project folder.")
with tab3:
    st.header("Data Insights (EDA)")
    try:
        # 1. Use our cached function to get the data
        @st.cache_data
        def load_eda_data():
            # Load only the columns we need for the chart and metrics
            return pd.read_csv('train.csv', usecols=categories)

        df_train = load_eda_data() # Now df_train is defined!

        # 2. Show the Metric at the top
        total_samples = len(df_train)
        toxic_count = df_train['toxic'].sum()
        toxic_pct = (toxic_count / total_samples) * 100
        st.metric("Overall Toxicity Rate in Dataset", f"{toxic_pct:.2f}%")

        # 3. Category Distribution Chart
        st.subheader("Category Distribution")
        st.bar_chart(df_train.sum())

        # 4. Cleaning Example (loads only 5 rows)
        st.subheader("Cleaning Example (Raw vs Preprocessed)")
        df_sample = pd.read_csv('train.csv', usecols=['comment_text'], nrows=5)
        df_sample['cleaned'] = df_sample['comment_text'].apply(clean_text)
        st.table(df_sample)

    except FileNotFoundError:
        st.error("Missing `train.csv`. Please ensure it is in the project folder to view insights.")