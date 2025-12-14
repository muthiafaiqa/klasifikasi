import streamlit as st
import joblib
import numpy as np

st.title("Klasifikasi Transaksi")

model, scaler = joblib.load("model_rf_classification.pkl")

total = st.number_input("Total Harga (Rp)", min_value=0)
qty = st.number_input("Kuantitas", min_value=1)

if st.button("Prediksi"):
    data = scaler.transform([[total, qty]])
    pred = model.predict(data)[0]

    label = {
        0: "Transaksi Kecil",
        1: "Transaksi Sedang",
        2: "Transaksi Besar"
    }

    st.success(f"Hasil Klasifikasi: {label[pred]}")
