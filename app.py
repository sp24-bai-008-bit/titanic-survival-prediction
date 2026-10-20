import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open("svc_trained_model.pkl", "rb"))

# same label lists as the training notebook, so the codes match
pclass_le = LabelEncoder().fit(["First", "Second", "Third"])
gender_le = LabelEncoder().fit(["Male", "Female"])
sibling_le = LabelEncoder().fit(["Zero", "One", "Two", "Three"])
embarked_le = LabelEncoder().fit(["Southampton", "Cherbourg", "Queenstown"])

st.title("Titanic Passenger Survival Prediction")

pclass = st.selectbox("Passenger Class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Male", "Female"])
sibling = st.selectbox("Siblings", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox("Embarked", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict"):
    row = pd.DataFrame({
        "PClass": [pclass_le.transform([pclass])[0]],
        "Gender": [gender_le.transform([gender])[0]],
        "Sibling": [sibling_le.transform([sibling])[0]],
        "Embarked": [embarked_le.transform([embarked])[0]],
    })
    if model.predict(row)[0] == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: NOT SURVIVED")
