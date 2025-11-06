import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords, wordnet
import string
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download NLTK resources (only needed once)
for resource in ['wordnet', 'averaged_perceptron_tagger', 'stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource != 'punkt' else 'tokenizers/punkt')
    except LookupError:
        nltk.download(resource)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

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

# Load the trained model, vectorizer, encoder, and expected column list
try:
    model = joblib.load('best_drug_condition_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    training_columns = joblib.load('training_features.pkl')

    # Remove 'rating' if it's in the training columns (important for alignment later)
    if 'rating' in training_columns:
        training_columns.remove('rating')

except FileNotFoundError as e:
    st.error(f"Error loading files: {e}. Please ensure all necessary files are in the same directory.")
    st.stop()

st.title("Drug Condition Prediction")

review_text = st.text_area("Enter the patient review:", "")
useful_count = st.number_input("Enter the useful count:", value=0, step=1)

if st.button("Predict Condition"):
    if review_text and useful_count is not None:
        processed_input = preprocess_text(review_text)

        # TF-IDF transformation
        input_features_text = tfidf_vectorizer.transform([processed_input]).toarray()
        input_df_text = pd.DataFrame(input_features_text, columns=[str(i) for i in range(input_features_text.shape[1])])

        # Other numerical features (only usefulCount now)
        input_features_other = pd.DataFrame({
            'usefulCount': [useful_count]
        })

        # Combine all features
        input_df = pd.concat([input_df_text, input_features_other.reset_index(drop=True)], axis=1)

        # Align with training columns
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[training_columns]

        # Predict
        prediction_encoded = model.predict(input_df)
        predicted_condition = label_encoder.inverse_transform(prediction_encoded)[0]
        st.subheader("Predicted Condition:")
        st.write(predicted_condition)
    else:
        st.warning("Please enter all the required information.")