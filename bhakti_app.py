import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
clf = pickle.load(open("mymodel.pkl","rb"))

def predict(data):
    clf = pickle.load(open("mymodel.pkl","rb"))
    return clf.predict(data)

#use to give tital in webpage
st.title("Salary prediction dataset using Machine Learning")
#showing a small note in webpage
st.markdown("This Model Identify salary")

#give header
st.header("Salary based on no of Experience")
col1,col2 = st.columns(2)

with col1:
	st.text("YearsExperience")
	YOE = st.slider("Salary...", 1.0, 10000.0, 0.5)#min/max/step mark

st.text('')
if st.button("Seles Prediction "):
    result = clf.predict(np.array([[YearsExperience]]))
    st.text(result[0])

st.markdown("Developed By Bhakti Soni at Polytechnice Daman")
