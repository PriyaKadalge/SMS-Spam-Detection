import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load saved files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Sidebar
st.sidebar.title("Project Information")
st.sidebar.write("Spam Classifier using Machine Learning")
st.sidebar.write("Model: Multinomial Naive Bayes")
st.sidebar.write("Vectorizer: TF-IDF")

# Main title
st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check whether it is spam or not.")

# User input
input_sms = st.text_area("Enter the message")

# Message statistics
if input_sms:
    st.write(f"Character Count: {len(input_sms)}")
    st.write(f"Word Count: {len(input_sms.split())}")

# Prediction
if st.button('Predict'):

    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Show processed text
    st.subheader("Processed Text")
    st.write(transformed_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Probability
    prob = model.predict_proba(vector_input)

    # Result
    if result == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Not Spam Message")

    # Prediction confidence
    st.subheader("Prediction Confidence")
    st.write(prob)

# Example section
st.subheader("Example Messages")
st.write("Spam: Congratulations! You have won a free mobile recharge.")
st.write("Not Spam: Please attend the meeting at 10 AM tomorrow.")

# Footer
st.markdown("---")
st.write("Developed using Streamlit and Machine Learning")
