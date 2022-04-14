import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for word in text:
        if (word.isalnum()) and (word not in stopwords.words('english')) and (word not in string.punctuation):
            y.append(word)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)


cv = pickle.load(open('cv.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email Spam Classifier')
input_email = st.text_area(label='Email text')

# preprocess input
# Vectorise
# Predict
# Display

# 1. preprocess text
transformed_email = transform_text(input_email)

# 2. Vectorise
vector_input = cv.transform([transformed_email])

# 3. Predict
result = model.predict(vector_input)[0]

# 4. Display
if st.button('PREDICT'):
    input_email = st.empty()
    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT A SPAM')