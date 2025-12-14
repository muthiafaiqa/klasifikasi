import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Klasifikasi Transaksi Otomatis")

# Load dataset
df_cluster = pd.read_csv("TRANSAKSI_PENJUALAN_PRODUK_TOKO_BANGUNAN_SYNTHETIC.csv")

# Bersihkan nama kolom: hapus spasi, ubah ke huruf kecil, ganti spasi dengan underscore
df_cluster.columns = df_cluster.columns.str.strip().str.replace(" ", "_").str.lower()
# Sekarang kolom penting: 'id_transaksi', 'kuantitas', 'total_harga'

# Buat KMeans untuk mendapatkan centroid
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster[['total_harga', 'kuantitas']])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
centroids = scaler.inverse_transform(kmeans.cluster_centers_)

# Urutkan cluster berdasarkan total_harga
cluster_order = np.argsort(centroids[:, 0])
labels = ["Transaksi Kecil", "Transaksi Sedang", "Transaksi Besar"]
cluster_label = {c: labels[i] for i, c in enumerate(cluster_order)}

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

    # Tampilkan centroid untuk referensi
    centroids_df = pd.DataFrame(centroids, columns=['total_harga', 'kuantitas'])
    centroids_df['label'] = [cluster_label[i] for i in range(len(centroids))]
    st.subheader("Centroid Cluster:")
    st.dataframe(centroids_df)
