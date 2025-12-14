import streamlit as st
import joblib
import numpy as np

st.title("Klasifikasi Transaksi")

# Load model dan scaler
model, scaler = joblib.load("model_rf_classification.pkl")

# Input user
total = st.number_input("Total Harga (Rp)", min_value=0)
qty = st.number_input("Kuantitas", min_value=1)

if st.button("Prediksi"):
    # Transform input
    data = scaler.transform([[total, qty]])
    
    # Prediksi cluster
    cluster = model.predict(data)[0]

    # Mapping cluster ke label
    cluster_label = {
        0: "Transaksi Kecil",
        1: "Transaksi Besar",
        2: "Transaksi Sedang"
    }

    # Ambil label
    hasil = cluster_label[cluster]

    st.success(f"Hasil Prediksi: {hasil}")
