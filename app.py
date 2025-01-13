import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import pyperclip  # For clipboard operations

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Initialize necessary tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def transform_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)  # Remove punctuations
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and token.strip()]
    return ' '.join(tokens)

# Load the models
try:
    tfidf = pickle.load(open('tfidf.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Ensure 'tfidf.pkl' and 'model.pkl' are in the same directory as this script.")
    st.stop()

# App UI
st.sidebar.image("filtering.png", use_container_width=True)  # Add your custom logo here
st.title('📧 Email Spam Classifier')
st.markdown("### Classify whether an email is **Spam** or **Not Spam** 🔍")

# Example Emails Section
st.sidebar.header("Test with Examples")
examples = {
    "Spam Example": "Congratulations! You've won a $1000 gift card. Click here to claim your prize.",
    "Not Spam Example": "Hello John, just checking in to see if you’re available for a meeting tomorrow."
}

for example_name, example_text in examples.items():
    st.sidebar.markdown(f"**{example_name}**")
    st.sidebar.text_area("", value=example_text, height=70, key=example_name, disabled=True)
    if st.sidebar.button(f"Copy {example_name}", key=f"copy_{example_name}"):
        pyperclip.copy(example_text)  # Copy the example email text to the clipboard
        st.sidebar.success(f"Copied {example_name} to clipboard! You can paste it into the text area above.")

# Main Text Area
input_mail = st.text_area('Enter the message', placeholder='Type or paste your email content here...', help="Copy an example email from the sidebar or type your own.")

# Prediction Button Logic
if st.button('Predict', help="Click to analyze the email content"):
    if not input_mail.strip():
        st.warning('Please enter a message to classify.')
    else:
        with st.spinner('Classifying...'):
            try:
                # Transform the input
                transformed_mail = transform_text(input_mail)

                # Create vector
                vector_input = tfidf.transform([transformed_mail]).toarray()

                # Make prediction
                result = model.predict(vector_input)[0]

                # Show result with feedback
                if result == 1:
                    st.error('🛑 This is Spam.')
                else:
                    st.success('✅ This is Not Spam.')

                # Feedback Section
                st.markdown("### 📋 Was this prediction correct?")
                feedback = st.radio("", ["Yes", "No"])
                if feedback == "No":
                    st.text_input("What was wrong with the prediction?", help="Provide additional feedback for improvement.")

            except Exception as e:
                st.error(f'An error occurred: {str(e)}')

# Additional Info Section
st.divider()
st.markdown("""
### Learn More
This app uses a machine learning model to classify emails based on their content. It's trained on a dataset of labeled spam and non-spam emails. 
To improve accuracy, consider using more advanced preprocessing techniques or retraining the model on updated datasets.
""")
st.info("For more details or to contribute, visit [GitHub](https://github.com/Akshat-Sharma-110011).")

# Metrics Section (Optional)
if st.checkbox("Show Model Performance Metrics"):
    st.metric(label="Accuracy", value="98.5%")
    st.metric(label="Precision", value="99.2%")
    st.metric(label="Recall", value="97.8%")
