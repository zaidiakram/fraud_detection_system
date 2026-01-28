import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

model = joblib.load("fraud_detection_model.jb")
encoders = joblib.load("label_encoders.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

st.title("Fraud Detection System")
st.write("Enter transaction details below")

category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0)
lat = st.number_input("Transaction Latitude",format="%.6f")
lon = st.number_input("Transaction Longitude",format="%.6f")
merch_lat = st.number_input("Merchant Latitude",format="%.6f")
merch_lon = st.number_input("Merchant Longitude",format="%.6f")

hour = st.slider("Transaction Hour", 0, 23, 12)
day = st.slider("Transaction Day", 1, 31, 15)
month = st.slider("Transaction Month", 1, 12, 6)

gender_ui = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number")

gender = "M" if gender_ui == "Male" else "F"
distance = haversine(lat, lon, merch_lat, merch_lon)

if st.button("Predict Fraudulence"):
    if category and cc_num:

        input_data = pd.DataFrame([[
            category,
            amt,
            hash(cc_num) % (10**6),
            hour,
            day,
            month,
            gender,
            distance
        ]], columns=[
            "category",
            "amt",
            "cc_num",
            "hour",
            "day",
            "month",
            "gender",
            "distance"
        ])

        for col in ["category", "gender"]:
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except:
                input_data[col] = -1

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.subheader("Prediction: Fraudulent Transaction")
        else:
            st.subheader("Prediction: Legitimate Transaction")

        st.write(f"Risk Score: {probability:.2f}")

    else:
        st.warning("Category and Credit Card Number are required.")
