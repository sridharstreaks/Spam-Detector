import streamlit as st
import pickle
import string
import nltk
import pandas as pd

# Text Transformer
def transform(string_input):
    #lowering case 
    string_input=string_input.lower()
    
    # splitting by words
    string_input=nltk.word_tokenize(string_input)
    
    #remove special characters
    import re
    string_input=[re.sub('[^a-zA-Z0-9]+', '', _) for _ in string_input]
    while("" in string_input):
        string_input.remove("")
    
    #remove stopwords and stemming
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    temp=[]
    for i in string_input:
        if i not in stop_words:
            # Stemming
            stemmed=ps.stem(i)
            temp.append(stemmed)
            
    string_input = temp[:]
    temp.clear()
            
    #getting s single string_input
    string_input=' '.join(string_input)
    
    return string_input

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model_bnb.pkl','rb'))

def main():
    # Initialize a session state variable that tracks the sidebar state (either 'expanded' or 'collapsed').
    if 'sidebar_state' not in st.session_state:
        st.session_state.sidebar_state = 'collapsed'

    # Streamlit set_page_config method has a 'initial_sidebar_state' argument that controls sidebar state.
    st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state)
    
    st.title("Spam Detector by Streaks V.2")

    input_text=st.text_area("Enter the Message")
    input_text=input_text.strip()

    if st.button('Predict'):

        if input_text is None or input_text=='':
            st.error("Please Enter a Text")
        else:
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

    st.subheader('Not Correct?')
    st.markdown('Help the model by contributing your data anonymously')

    # Toggle sidebar state between 'expanded' and 'expanded'.
    if st.button('Contribute!'):
        st.session_state.sidebar_state = 'expanded' if st.session_state.sidebar_state == 'collapsed' else 'collapsed'
        # Force an app rerun after switching the sidebar state.
        st.experimental_rerun()
    
    st.sidebar.subheader('Make Contribution Here')
    contribute_text = st.sidebar.text_area("Enter the Message you want to contribute")
    contribute_text=contribute_text.strip()
    contribute_label = st.sidebar.radio("Select the Label", ["0", "1"], index=None,captions=['Not Spam','Spam'])

    if st.sidebar.button('Finish Contribution'):
        if contribute_text is None or contribute_text=='':
            st.sidebar.error("Please Enter a Text")
        else:
            # Save the contribution to a CSV file
            contribution_data = {'text': [contribute_text], 'target': [contribute_label]}

            try:
                contribution_df = pd.read_csv('contributions.csv')
                contribution_df = pd.concat([contribution_df, pd.DataFrame(contribution_data)], ignore_index=True)
            except FileNotFoundError:
                contribution_df = pd.DataFrame(contribution_data)
                
            contribution_df.to_csv('contributions.csv', index=False)
            st.success("Thanks for your contribution!")

    # Read the contributions
    contribution_df = pd.read_csv('contributions.csv')
    if len(contribution_df)>=2:
            try:

                # Transform and vectorize the contributions
                contribution_df['text'] = contribution_df['text'].apply(transform)
                X_contribution = tfidf.fit_transformtransform(contribution_df['text'])
                y_contribution = contribution_df['target']

                # Update the model with new data
                model.partial_fit(X_contribution, y_contribution, classes=[0, 1])
                contribution_df.drop(contribution_df.index, inplace=True)
                contribution_df.to_csv('contributions.csv', index=False)

                st.success("Model Updated")

            except FileNotFoundError:
                st.error("No contributions found for retraining!")

    

    st.warning('The App still fails to identiy tranctional messages correctly. So, use the app with caution. Also more contributions would be really helpful in training the model on those type of messages', icon="⚠️")
if __name__=="__main__":
    main()