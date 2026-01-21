import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. INISIALISASI & AUTO-TRAIN LOGIC
# ==========================================
st.set_page_config(page_title="Churn AI Ensemble Predictor", layout="wide")

def check_and_initialize():
    """Memastikan folder dan model tersedia sebelum aplikasi berjalan."""
    for folder in ['models', 'reports']:
        if not os.path.exists(folder):
            os.makedirs(folder)
            
    # Jika model utama belum ada, jalankan pelatihan otomatis
    if not os.path.exists('models/ensemble_model.pkl'):
        with st.status("🚀 Menginisialisasi Sistem: Melatih Model Ensemble..."):
            try:
                # Mengimpor dan menjalankan pipeline dari modeling.py
                from src.modeling import run_modeling_pipeline
                run_modeling_pipeline()
                st.success("✅ Model Berhasil Dilatih Otomatis!")
            except Exception as e:
                st.error(f"Gagal inisialisasi model: {e}")
                st.stop()

# Jalankan pengecekan saat aplikasi pertama kali dibuka (sangat berguna saat deployment)
check_and_initialize()

@st.cache_resource
def load_assets():
    """Memuat artefak model dan data ke dalam cache."""
    model_ens = joblib.load('models/ensemble_model.pkl')
    # Load dataset asli untuk kebutuhan Dashboard EDA (Soal 4.1)
    path_raw = 'data/preprocess/dataset_cleaned.csv' # Menggunakan data yang sudah bersih
    df_raw = pd.read_csv(path_raw) if os.path.exists(path_raw) else pd.DataFrame()
    return model_ens, df_raw

# Memuat Resource
try:
    model, df_raw = load_assets()
except Exception as e:
    st.error(f"Gagal memuat artefak. Jalankan 'src/modeling.py' terlebih dahulu. Error: {e}")

# ==========================================
# 2. NAVIGASI SIDEBAR
# ==========================================
st.sidebar.title("📊 Navigasi Sistem")
menu = st.sidebar.radio("Pilih Halaman:", ["Prediksi Real-Time", "Dashboard Analisis (EDA)", "Metodologi & Evaluasi"])

# ==========================================
# 3. HALAMAN 1: PREDIKSI REAL-TIME (Soal 4.2)
# ==========================================
if menu == "Prediksi Real-Time":
    st.title("🛡️ Prediksi Risiko Churn (Ensemble System)")
    st.write("Sistem ini memproses data melalui penggabungan model **XGBoost** dan **Logistic Regression**.")

    # Input Panel (12-13 Fitur sesuai dataset Anda)
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
        # Membuat DataFrame RAW (Sesuai kolom numeric_features & categorical_features di modeling.py)
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

        # Prediksi menggunakan Pipeline (Preprocessing otomatis)
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

        # Insight Bisnis (Soal 4.4)
        st.subheader("💡 Rekomendasi Tindakan")
        if prob > 0.5:
            st.warning("Pelanggan menunjukkan indikasi ketidakpuasan tinggi. Tim CRM disarankan memberikan voucher retensi segera.")
        else:
            st.info("Kondisi pelanggan stabil. Pertahankan program loyalitas yang sedang berjalan.")

# ==========================================
# 4. HALAMAN 2: DASHBOARD EDA (Soal 4.1)
# ==========================================
elif menu == "Dashboard Analisis (EDA)":
    st.title("📈 Eksplorasi Data Faktor Churn")
    st.write("Visualisasi ini menunjukkan hubungan antara perilaku pelanggan dan status churn.")

    if not df_raw.empty:
        col_eda1, col_eda2 = st.columns(2)
        
        with col_eda1:
            st.write("### Pengaruh Komplain terhadap Churn")
            fig, ax = plt.subplots()
            sns.barplot(data=df_raw, x='Komplain', y='Churn', ax=ax, palette='RdYlGn_r')
            st.pyplot(fig)
            st.caption("Insight: Riwayat komplain berkontribusi besar terhadap keputusan churn pelanggan.")

        with col_eda2:
            st.write("### Hubungan Skor Kepuasan & Churn")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_raw, x='Churn', y='Skor_Kepuasan', ax=ax)
            st.pyplot(fig)
            st.caption("Insight: Pelanggan yang churn cenderung memberikan skor kepuasan yang lebih rendah.")
    else:
        st.warning("Dataset tidak ditemukan di data/preprocess/dataset_cleaned.csv.")

# ==========================================
# 5. HALAMAN 3: METODOLOGI & EVALUASI (Soal 4.3 & 4.5)
# ==========================================
elif menu == "Metodologi & Evaluasi":
    st.title("📑 Metodologi Model Ensemble")
    
    st.markdown("""
    Sistem ini menggunakan teknik **Ensemble Soft Voting** untuk menjamin stabilitas prediksi.
    
    ### Alur Kerja:
    1. **Preprocessing**: Otomatisasi pembersihan data (Median Imputer) dan Encoding melalui *Scikit-learn Pipelines*.
    2. **XGBoost**: Menangani pola non-linear dan interaksi fitur yang kompleks (Tuned via Grid Search).
    3. **Logistic Regression**: Menjaga stabilitas prediksi melalui pola linear.
    4. **Soft Voting**: Menggabungkan rata-rata probabilitas dari kedua algoritma untuk hasil yang moderat.
    """)

    st.divider()
    st.subheader("Metrik Evaluasi Utama")
    st.latex(r"F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}")
    st.info("F1-Score dipilih karena dataset memiliki ketidakseimbangan kelas (*Imbalanced Class*).")
    
    # Menampilkan plot evaluasi dari folder reports jika ada
    if os.path.exists('reports/confusion_matrix.png'):
        st.write("### Confusion Matrix (Hasil Pelatihan)")
        st.image('reports/confusion_matrix.png')