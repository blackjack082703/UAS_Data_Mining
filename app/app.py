import streamlit as st
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# KONFIGURASI PATH
# ==========================================
# Tambahkan root directory ke sys.path agar 'src' bisa diimport
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # Naik satu level ke root project
sys.path.append(root_dir)

st.set_page_config(page_title="Churn AI Ensemble Predictor", layout="wide")

# ==========================================
# 1. FUNGSI LOADER YANG ROBUST (SELF-HEALING)
# ==========================================
@st.cache_resource
def load_assets():
    """
    Memuat model dan data. 
    Jika model rusak/beda versi/tidak ada, otomatis jalankan training ulang.
    """
    model_path = os.path.join(root_dir, 'models/ensemble_model.pkl')
    data_path = os.path.join(root_dir, 'data/preprocess/dataset_cleaned.csv')
    
    model_ens = None
    
    # Fungsi internal untuk training ulang
    def retrain_model():
        with st.spinner("🔄 Mendeteksi perubahan versi atau model hilang. Melatih ulang model..."):
            try:
                from src.modeling import run_modeling_pipeline
                # Pastikan kita di root dir saat menjalankan script modeling agar path relative aman
                original_cwd = os.getcwd()
                os.chdir(root_dir) 
                run_modeling_pipeline()
                os.chdir(original_cwd) # Kembalikan ke dir asal
                st.toast("✅ Model berhasil diperbarui!", icon="🎉")
            except Exception as e:
                st.error(f"FATAL: Gagal melatih ulang model. Cek src/modeling.py. Error: {e}")
                st.stop()

    # LOGIKA UTAMA: Try Load -> Except -> Retrain -> Load Again
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError("File model belum ada.")
        
        # Coba load model
        model_ens = joblib.load(model_path)
        
    except (FileNotFoundError, AttributeError, ImportError, Exception) as e:
        # GANTI st.warning DENGAN INI:
        print(f"⚠️ [LOG SYSTEM] Model usang terdeteksi ({e}). Memulai retraining...") 
        st.toast("⚙️ Memperbarui kompatibilitas sistem...", icon="🔄") # Notifikasi halus
        
        # Hapus file lama jika ada
        if os.path.exists(model_path):
            os.remove(model_path)
            
        # Latih ulang
        retrain_model()
        
        # Load kembali
        model_ens = joblib.load(model_path)

    # Load dataset untuk EDA
    df_raw = pd.read_csv(data_path) if os.path.exists(data_path) else pd.DataFrame()
    
    return model_ens, df_raw

# Memuat Resource (Hanya memanggil fungsi ini, logika try-except sudah di dalam)
model, df_raw = load_assets()

# ==========================================
# 2. NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("📊 Navigasi Sistem")
menu = st.sidebar.radio("Pilih Halaman:", ["Prediksi Real-Time", "Dashboard Analisis (EDA)", "Metodologi & Evaluasi"])

# ==========================================
# 3. HALAMAN 1: PREDIKSI REAL-TIME
# ==========================================
if menu == "Prediksi Real-Time":
    st.title("🛡️ Prediksi Risiko Churn (Ensemble System)")
    st.write("Sistem ini memproses data melalui penggabungan model **XGBoost** dan **Logistic Regression**.")

    # Input Panel
    st.sidebar.header("📝 Input Profil Pelanggan")
    
    with st.sidebar.expander("Informasi Demografi & Akun", expanded=True):
        gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
        status = st.selectbox('Status Perkawinan', ['Single', 'Menikah', 'Cerai'])
        perangkat = st.selectbox('Perangkat Login', ['HP', 'Smartphone', 'Komputer'])
        kota = st.radio('Tingkat Kota', [1, 2, 3])

    with st.sidebar.expander("Aktivitas Belanja", expanded=True):
        lama = st.slider('Lama Berlangganan (Bulan)', 0, 72, 12)
        jam_app = st.slider('Jam di Aplikasi', 1, 15, 3)
        hari_pesan = st.number_input('Hari Sejak Pesanan Terakhir', 0, 60, 5)
        cashback = st.number_input('Jumlah Cashback', 0, 5000, 150)
        metode = st.selectbox('Metode Pembayaran', ['Debit Card', 'Credit Card', 'COD', 'UPI'])

    with st.sidebar.expander("Logistik & Kepuasan", expanded=True):
        jarak = st.number_input('Jarak Gudang ke Rumah (km)', 0, 200, 20)
        kepuasan = st.select_slider('Skor Kepuasan', options=[1, 2, 3, 4, 5], value=3)
        komplain = st.radio('Pernah Komplain?', [0, 1], format_func=lambda x: 'Ya' if x==1 else 'Tidak')

    if st.button('Mulai Analisis Sistem'):
        input_data = pd.DataFrame({
            'Lama_Berlangganan': [lama],
            'Jarak_Gudang_ke_Rumah': [jarak],
            'Jam_di_Aplikasi': [jam_app],
            'Skor_Kepuasan': [kepuasan],
            'Hari_Sejak_Pesanan_Terakhir': [hari_pesan],
            'Jumlah_Cashback': [cashback],
            'Komplain': [komplain],
            'Tingkat_Kota': [kota],
            'Perangkat_Login': [perangkat],
            'Metode_Pembayaran': [metode],
            'Jenis_Kelamin': [gender],
            'Status_Perkawinan': [status]
        })

        # Prediksi
        prob = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("🏁 Hasil Analisis Risiko")
        c1, c2 = st.columns(2)
        
        with c1:
            st.metric(label="Skor Probabilitas Churn", value=f"{prob:.2%}")
        with c2:
            if prediction == 1:
                st.error("🚨 HASIL: PELANGGAN BERISIKO BERHENTI (CHURN)")
            else:
                st.success("✅ HASIL: PELANGGAN DIPREDIKSI TETAP LOYAL")

        st.subheader("💡 Rekomendasi Tindakan")
        if prob > 0.5:
            st.warning("Pelanggan menunjukkan indikasi ketidakpuasan tinggi. Tim CRM disarankan memberikan voucher retensi segera.")
        else:
            st.info("Kondisi pelanggan stabil. Pertahankan program loyalitas yang sedang berjalan.")

# ==========================================
# 4. HALAMAN 2: DASHBOARD EDA
# ==========================================
elif menu == "Dashboard Analisis (EDA)":
    st.title("📈 Eksplorasi Data Faktor Churn")
    
    if not df_raw.empty:
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.write("### Pengaruh Komplain terhadap Churn")
            fig, ax = plt.subplots()
            sns.barplot(data=df_raw, x='Komplain', y='Churn', ax=ax, palette='RdYlGn_r')
            st.pyplot(fig)
        with col_eda2:
            st.write("### Hubungan Skor Kepuasan & Churn")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_raw, x='Churn', y='Skor_Kepuasan', ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Dataset tidak ditemukan. Pastikan file dataset_cleaned.csv tersedia.")

# ==========================================
# 5. HALAMAN 3: METODOLOGI
# ==========================================
elif menu == "Metodologi & Evaluasi":
    st.title("📑 Metodologi Model Ensemble")
    st.markdown("""
    Sistem ini menggunakan teknik **Ensemble Soft Voting**:
    1. **Preprocessing**: Otomatisasi pembersihan data (Median Imputer).
    2. **Algoritma**: Kombinasi XGBoost dan Logistic Regression.
    """)
    st.info("Evaluasi difokuskan pada F1-Score untuk menangani data tidak seimbang.")