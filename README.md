# Patient's Condition Classification Using Drug Reviews
This project leverages Natural Language Processing (NLP) to classify patient conditions based on their drug reviews. It aims to recommend suitable drugs by analyzing user feedback and understanding drug effectiveness and side effects.

## 📌 Business Objective
Classify conditions such as Depression, High Blood Pressure, and Diabetes Type 2 using patient-written reviews. The ultimate goal is to recommend drugs based on real-world feedback.

## 📊 Dataset Overview
Total Entries: 161,297 drug reviews

Features:

DrugName (categorical)

Condition (target)

Review (text)

Rating (1–10)

Date (review timestamp)

UsefulCount (helpfulness votes)

## 🔧 Setup

git clone <repository_url>
cd <repository_name>
pip install -r requirements.txt
Download NLTK resources:

python
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')


## ▶️ Run the App

streamlit run app.py


## 🧠 How It Works
Preprocess Review

Lowercasing, punctuation removal, tokenization

Stopword removal, POS tagging, lemmatization, stemming

Model Prediction Pipeline

Loads TF-IDF Vectorizer, Label Encoder, and trained classification model

Transforms review → TF-IDF → Prediction → Label Decoding

User Interface

Users input a drug review

App predicts the most likely medical condition

## 📁 File Structure
app.py: Streamlit app

best_drug_condition_model.pkl: Trained classifier

tfidf_vectorizer.pkl: TF-IDF model

label_encoder.pkl: Condition label encoder

requirements.txt: Python dependencies

Project.pptx: Project slides

Patient's Condition Classification.docx: Project brief

## 🔮 Future Enhancements
Expand condition categories

Integrate sentiment analysis

Improve accuracy using advanced NLP models (e.g., BERT)

Add user feedback loop for model retraining

