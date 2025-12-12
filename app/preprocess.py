import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import streamlit as st

ps = PorterStemmer()

@st.cache_resource
def ensure_nltk_data():
    
    for res in ["punkt_tab", "punkt"]:
        try:
            nltk.data.find(f"tokenizers/{res}")
        except LookupError:
            nltk.download(res, quiet=True)

    
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

ensure_nltk_data()


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
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return y
