import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Set NLTK data path to a writable directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download required NLTK data with better error handling
@st.cache_resource
def download_nltk_data():
    try:
        # Set NLTK data path
        nltk.data.path.append(nltk_data_dir)
        
        # Download punkt tokenizer
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        # Download stopwords
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        return True
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")
        return False

# Download NLTK data
download_nltk_data()

ps=PorterStemmer()

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

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classification Model')
st.markdown('Made with &hearts; by Shazim Javed')
input_sms=st.text_area('Enter Your Message:')

if st.button("🚀 Enter to Predict", use_container_width=True,type="primary"):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.success("This Message is Spam")
    else:
        st.success("This Message is Not Spam")
