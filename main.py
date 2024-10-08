import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

model = load_model('simple_rnn_imdb.h5')

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}



def preprocess_text(text):
  words=text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

import streamlit as st

st.title('IMDB Movie Review Sentence Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    sentimwnt = 'positive' if prediction[0][0] > 0.5 else 'negative'

    st.write(f'Sentiment: {sentimwnt}')
    st.write(f'prediction Score: {prediction[0][0]}')

else:
   st.write('please enter a movie review')