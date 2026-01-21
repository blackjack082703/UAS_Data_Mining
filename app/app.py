import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Konfigurasi Halaman & Estetika
st.set_page_config(page_title="Churn Ensemble Predictor", layout="wide")

@st.cache_resource
def load_resources():
    # Memuat Pipeline Ensemble (Preprocessor + Voting Classifier)
    ensemble = joblib.load('models/ensemble_model.pkl')
    # Memuat model individu untuk keperluan transparansi/pembanding
    xgb = joblib.load('models/xgb_model.pkl')
    logreg = joblib.load('models/logreg_model.pkl')
    # Memuat data asli untuk halaman EDA
    path_data = 'data/raw/ecommerce_churn_dummy.csv'
    df_raw = pd.read_csv(path_data) if os.path.exists(path_data) else pd.DataFrame()
    return ensemble, xgb, logreg, df_raw

try:
    model_ens, model_xgb, model_log, df_raw = load_resources()
except Exception as e:
    st.error(f"Gagal memuat artefak model. Pastikan 'modeling.py' sudah dijalankan. Error: {e}")

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("📊 Navigasi Sistem")
menu = st.sidebar.radio("Pilih Halaman:", ["Prediksi Ensemble", "Dashboard Analisis (EDA)", "Metodologi & Perbandingan"])

# --- HALAMAN 1: PREDIKSI ENSEMBLE ---
if menu == "Prediksi Ensemble":
    st.title("🛡️ Sistem Prediksi Churn (Ensemble Learning)")
    st.write("Keputusan akhir dihitung berdasarkan rata-rata probabilitas dari model **XGBoost** dan **Logistic Regression**.")

    # Input Panel (13 Fitur Lengkap)
    st.sidebar.header("📝 Data Input Pelanggan")
    
    with st.sidebar.expander("Demografi & Akun", expanded=True):
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

    if st.button('Mulai Analisis Ensemble'):
        # Membuat DataFrame input mentah
        input_raw = pd.DataFrame({
            'Lama_Berlangganan': [lama],
            'Perangkat_Login': [perangkat],
            'Tingkat_Kota': [kota],
            'Jarak_Gudang_ke_Rumah': [jarak],
            'Metode_Pembayaran': [metode],
            'Jenis_Kelamin': [gender],
            'Jam_di_Aplikasi': [jam_app],
            'Skor_Kepuasan': [kepuasan],
            'Status_Perkawinan': [status],
            'Komplain': [komplain],
            'Hari_Sejak_Pesanan_Terakhir': [hari_pesan],
            'Jumlah_Cashback': [cashback]
        })

        # Prediksi Ensemble & Individual
        prob_ens = model_ens.predict_proba(input_raw)[0][1]
        pred_ens = model_ens.predict(input_raw)[0]
        
        prob_xgb = model_xgb.predict_proba(input_raw)[0][1]
        prob_log = model_log.predict_proba(input_raw)[0][1]

        # Tampilan Hasil
        st.divider()
        st.subheader("🏁 Hasil Analisis Akhir")
        
        c_main1, c_main2 = st.columns(2)
        with c_main1:
            st.metric(label="Skor Risiko Gabungan", value=f"{prob_ens:.2%}")
        with c_main2:
            if pred_ens == 1:
                st.error("KEPUTUSAN: BERISIKO CHURN")
            else:
                st.success("KEPUTUSAN: TETAP BERLANGGANAN")

        # Rincian Kontribusi Model
        st.write("### Rincian Kontribusi Algoritma")
        col_sub1, col_sub2 = st.columns(2)
        col_sub1.info(f"**XGBoost**: {prob_xgb:.2%}")
        col_sub2.info(f"**Logistic Regression**: {prob_log:.2%}")
        st.caption("Hasil akhir adalah nilai rata-rata tertimbang dari kedua algoritma di atas.")

# --- HALAMAN 2: DASHBOARD EDA ---
elif menu == "Dashboard Analisis (EDA)":
    st.title("📈 Analisis Deskriptif Pelanggan")
    if not df_raw.empty:
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Korelasi Komplain & Churn")
            fig, ax = plt.subplots()
            sns.barplot(data=df_raw, x='Komplain', y='Churn', ax=ax, palette='RdYlGn_r')
            st.pyplot(fig)
        with c2:
            st.write("### Pengaruh Cashback terhadap Churn")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_raw, x='Churn', y='Jumlah_Cashback', ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Data mentah tidak ditemukan.")

# --- HALAMAN 3: METODOLOGI ---
elif menu == "Metodologi & Perbandingan":
    st.title("📑 Metodologi Ensemble Learning")
    
    st.write("Sistem ini menggunakan teknik **Soft Voting Ensemble** untuk meminimalkan bias pada prediksi.")
    
    # [Image of ensemble model architecture]
    
    st.markdown("""
    ### Alur Kerja:
    1. **Preprocessing**: Data numerik dan kategorikal diproses melalui *Scikit-learn Pipeline* secara otomatis.
    2. **XGBoost**: Menangani pola non-linear dan interaksi fitur yang kompleks.
    3. **Logistic Regression**: Menjaga stabilitas prediksi dengan pola linear.
    4. **Soft Voting**: Menggabungkan probabilitas dari kedua model untuk hasil yang lebih moderat.
    """)
    
    st.subheader("Metrik Evaluasi Utama")
    st.latex(r"F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}")
    st.info("F1-Score dipilih sebagai acuan utama karena mampu memberikan penilaian adil pada data yang tidak seimbang (Imbalanced Class).")