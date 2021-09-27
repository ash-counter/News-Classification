#-*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:20:03 2021

@author: pauly
"""

import numpy as np
#import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
#from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
#from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
#from tensorflow.keras.models import Sequential
#from keras.utils.np_utils import to_categorical
import streamlit as st

def welcome():
    print("Welcome all")
    
model =load_model("C:/Users/pauly/Project/Notebooks/News Classification/Model/my_model.h5")

def predict(txt):
    txt = ["A soccer was eaten by Elvis Presley."]

    #cleaning and preprocessing the text
    cleaned = []
    for i in range(0,len(txt)):
        msg = re.sub('[^a-zA-Z]',' ',txt[i])
        msg = msg.lower()
        msg = msg.split()
        ps = PorterStemmer()
        msg = [ps.stem(words) for words in msg if not words in set(stopwords.words('english'))]
        msg = ' '.join(msg)
        cleaned.append(msg)
        
    #one hot encoding and embedding layer
    dict_size = 5000
    one_hot_mat = [one_hot(words,dict_size) for words in cleaned]
    embedded_layer = pad_sequences(one_hot_mat,padding = 'pre',maxlen = 150)
    embedded_layer

    #prediction
    pred = model.predict(embedded_layer)
    cat = ['Business','Science','Entertainment','Health']
    print(pred, cat[np.argmax(pred)])
    return pred

def main():
    st.title("News Classification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit News Classification ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    txt = st.text_input("News Headline","Type Here")
    
    result=""
    if st.button("Predict"):
        result=predict(txt)
    cat = ['Business','Science','Entertainment','Health']
   # print(result, cat[np.argmax(result)])    
    st.success('This is {} news'.format(cat[np.argmax(result)]))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        
if __name__=="__main__":
    main()
