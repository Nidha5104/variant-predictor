import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("final_model.pkl")

st.title("🧬 Variant Predictor")

allele_freq = st.number_input("Allele Frequency", value=0.001)
cadd = st.number_input("CADD", value=10.0)
revel = st.number_input("REVEL", value=0.5)
sift = st.number_input("SIFT", value=0.05)
polyphen = st.number_input("PolyPhen", value=0.5)
ppi = st.number_input("PPI Score", value=100.0)

if st.button("Predict"):
    input_data = np.array([[allele_freq, cadd, revel, sift, polyphen, ppi]])
    pred = model.predict(input_data)[0]

    if pred == 1:
        st.error("Pathogenic")
    else:
        st.success("Benign")
