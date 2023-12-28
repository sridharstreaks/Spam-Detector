import streamlit as st
import pickle
import string
import nltk


# Text Transformer
def transform(string):
    #lowering case 
    string=string.lower()
    
    # splitting by words
    string=nltk.word_tokenize(string)
    
    #remove special characters
    import re
    string=[re.sub('[^a-zA-Z0-9]+', '', _) for _ in string]
    while("" in string):
        string.remove("")
    
    #remove stopwords and stemming
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    temp=[]
    for i in string:
        if i not in stop_words:
            # Stemming
            stemmed=ps.stem(i)
            temp.append(stemmed)
            
    string = temp[:]
    temp.clear()
            
    #getting s single string
    string=' '.join(string)
    
    return string

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model_bnb.pkl','rb'))

st.title("Spam Detector by Streaks V.1.0")

input_text=st.text_area("Enter the Message")

if st.button('Predict'):

    # 1. preprocess
    transformed_input = transform(input_text)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_input])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")