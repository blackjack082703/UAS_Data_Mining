import pandas as pd
import numpy as np
import os

# 1. Menentukan 'seed' agar hasil data acak selalu sama
np.random.seed(42)
n_rows = 2000

# 2. Membuat data dummy dengan Nama Kolom Bahasa Indonesia
data = {
    'ID_Pelanggan': range(50001, 50001 + n_rows),
    
    # Lama_Berlangganan (Tenure): Sengaja ada data kosong (NaN) untuk Soal 2.1
    'Lama_Berlangganan': np.random.choice([np.nan, 5.0, 10.0, 20.0, 30.0], n_rows, p=[0.1, 0.2, 0.3, 0.2, 0.2]),
    
    'Perangkat_Login': np.random.choice(['HP', 'Smartphone', 'Komputer'], n_rows),
    
    'Tingkat_Kota': np.random.choice([1, 2, 3], n_rows),
    
    # Jarak_Gudang_ke_Rumah: Sengaja ada Outliers (nilai > 100) untuk Soal 2.1
    'Jarak_Gudang_ke_Rumah': np.random.uniform(5, 120, n_rows), 
    
    'Metode_Pembayaran': np.random.choice(['Debit Card', 'UPI', 'CC', 'Credit Card', 'COD'], n_rows),
    
    'Jenis_Kelamin': np.random.choice(['Wanita', 'Pria'], n_rows),
    
    'Jam_di_Aplikasi': np.random.randint(1, 6, n_rows),
    
    'Skor_Kepuasan': np.random.randint(1, 6, n_rows),
    
    'Status_Perkawinan': np.random.choice(['Single', 'Menikah', 'Cerai'], n_rows),
    
    'Komplain': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    
    'Hari_Sejak_Pesanan_Terakhir': np.random.uniform(0, 25, n_rows),
    
    'Jumlah_Cashback': np.random.uniform(100, 500, n_rows),
    
    # Target tetap menggunakan nama 'Churn' agar sesuai dengan istilah di file soal
    'Churn': np.random.choice([0, 1], n_rows, p=[0.7, 0.3])
}

# 3. Mengubah dictionary menjadi DataFrame
df = pd.DataFrame(data)

# 4. Membuat folder tujuan
path_folder = 'data/raw'
if not os.path.exists(path_folder):
    os.makedirs(path_folder)

# 5. Menyimpan ke CSV
nama_file = f"{path_folder}/ecommerce_churn_dummy.csv"
df.to_csv(nama_file, index=False)

print("-" * 50)
print(f"BERHASIL: Dataset dengan kolom Bahasa Indonesia telah dibuat!")
print(f"Lokasi: {nama_file}")
print("-" * 50)
print("Daftar Kolom Baru:")
print(df.columns.tolist())