import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("earnings_model.pkl")

st.title("Earnings Manipulation Detection")

st.write("Enter financial ratios to predict manipulation risk")

DSRI = st.number_input("DSRI", value=1.0)
GMI = st.number_input("GMI", value=1.0)
AQI = st.number_input("AQI", value=1.0)
SGI = st.number_input("SGI", value=1.0)
DEPI = st.number_input("DEPI", value=1.0)
SGAI = st.number_input("SGAI", value=1.0)
ACCR = st.number_input("ACCR", value=0.0)
LEVI = st.number_input("LEVI", value=1.0)

if st.button("Predict"):
    input_data = np.array([[DSRI, GMI, AQI, SGI, DEPI, SGAI, ACCR, LEVI]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ High Risk of Earnings Manipulation (Prob: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Earnings Manipulation (Prob: {prob:.2f})")
