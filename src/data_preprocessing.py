import pandas as pd
import numpy as np
import os

def bersihkan_data(df):
    # 1. Menangani Data Hilang (Soal 2.1)
    df['Lama_Berlangganan'] = df['Lama_Berlangganan'].fillna(df['Lama_Berlangganan'].median())
    
    # 2. Menangani Inkonsistensi (Soal 2.1)
    # Menyesuaikan mapping untuk kolom yang tersedia di dataset Anda
    mapping_bayar = {'CC': 'Credit Card', 'Smartphone': 'HP'}
    df['Metode_Pembayaran'] = df['Metode_Pembayaran'].replace(mapping_bayar)
    
    # 3. Menangani Duplikat
    df = df.drop_duplicates()
    
    return df

def jalankan_preprosess():
    # Pastikan folder tujuan ada
    path_output = 'data/preprocess'
    if not os.path.exists(path_output):
        os.makedirs(path_output)
    
    # Load data dari path yang Anda tentukan
    path_raw = 'data/raw/ecommerce_churn_dummy.csv'
    if os.path.exists(path_raw):
        df = pd.read_csv(path_raw)
        df_bersih = bersihkan_data(df)
        
        # Simpan ke folder preprocess
        file_hasil = os.path.join(path_output, 'dataset_cleaned.csv')
        df_bersih.to_csv(file_hasil, index=False)
        print(f"✅ Sukses: Data bersih disimpan ke {file_hasil}")
    else:
        print(f"❌ Error: File {path_raw} tidak ditemukan!")

if __name__ == "__main__":
    jalankan_preprosess()