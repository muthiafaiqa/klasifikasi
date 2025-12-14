import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Klasifikasi Transaksi Otomatis")

# Load data cluster dan model
df_cluster = pd.read_csv("TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")  # Data asli untuk membangun centroid
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[['Total_Harga', 'Kuantitas']])

# Buat KMeans untuk mendapatkan centroid
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Urutkan cluster berdasarkan total_harga
cluster_order = np.argsort(centroids[:, 0])  # 0=Total_Harga
cluster_label = {}
labels = ["Transaksi Kecil", "Transaksi Sedang", "Transaksi Besar"]
for i, c in enumerate(cluster_order):
    cluster_label[c] = labels[i]

# Input user
total = st.number_input("Total Harga (Rp)", min_value=0)
qty = st.number_input("Kuantitas", min_value=1)

if st.button("Prediksi"):
    # Transform input
    data_scaled = scaler.transform([[total, qty]])
    
    # Prediksi cluster
    cluster = kmeans.predict(data_scaled)[0]
    
    # Ambil label sesuai urutan centroid
    hasil = cluster_label[cluster]
    
    st.success(f"Hasil Prediksi: {hasil}")

    # Optional: tampilkan centroid untuk referensi
    st.subheader("Centroid Cluster:")
    centroids_df = pd.DataFrame(centroids, columns=['Total_Harga', 'Kuantitas'])
    centroids_df['Label'] = [cluster_label[i] for i in range(len(centroids))]
    st.dataframe(centroids_df)
