import streamlit as st
import pickle
from preprocess import transform_text


tfi=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

st.title("Spam Email Detection")

email=st.text_area("Enter the Email")

if st.button("Check the email"):
    

# 1. preprocess

    pretxt_list=transform_text(email)
    pretxt=" ".join(pretxt_list)

#Vectorize

    vectxt=tfi.transform([pretxt])

#predict

    result=model.predict(vectxt)[0]

#display

    if result==1:
        st.subheader("The Email is Spam")
    else:
        st.subheader("The Email is not spam")
