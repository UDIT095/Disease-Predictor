import streamlit as st
import pandas as pd
import joblib
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources (only needed once)
for resource in ['wordnet', 'averaged_perceptron_tagger', 'stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else 'tokenizers/punkt')
    except LookupError:
        nltk.download(resource)

# Load stopwords, lemmatizer, and stemmer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Helper functions
def lemmatize_word(tag):
    tag = tag[0].lower()
    tag_dict = {"j": wordnet.ADJ, "n": wordnet.NOUN, "v": wordnet.VERB, "r": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tagged_tokens = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(word, lemmatize_word(tag)) for word, tag in tagged_tokens]
    stemmed_tokens = [stemmer.stem(word) for word in lemmatized_tokens]
    return " ".join(stemmed_tokens)

# Load the model, vectorizer, and encoder
try:
    model = joblib.load('best_drug_condition_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Streamlit UI
st.title("Drug Condition Prediction")
review_text = st.text_area("Enter the patient review:", "")

if st.button("Predict Condition"):
    if review_text:
        processed_input = preprocess_text(review_text)
        input_tfidf = tfidf_vectorizer.transform([processed_input])

        # Predict
        prediction_encoded = model.predict(input_tfidf)
        predicted_condition = label_encoder.inverse_transform(prediction_encoded)[0]

        st.subheader("Predicted Condition:")
        st.write(predicted_condition)
    else:
        st.warning("Please enter the patient review.")