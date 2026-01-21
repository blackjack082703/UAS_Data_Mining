Selamat! Dengan selesainya tahap Ensemble Learning (Soft Voting), proyek UAS Penambangan Data Anda kini berada pada level profesional. README.md adalah dokumen paling penting karena ini adalah "wajah" proyek Anda yang akan dilihat pertama kali oleh dosen penguji.Berikut adalah draf README.md yang lengkap, terstruktur, dan disesuaikan dengan metodologi gabungan model yang baru saja kita selesaikan:🛡️ E-Commerce Customer Churn Ensemble PredictorProyek Akhir Penambangan Data 2025/2026📝 Deskripsi ProyekProyek ini bertujuan untuk mendeteksi risiko churn (berhenti berlangganan) pelanggan e-commerce secara dini. Inovasi utama dalam sistem ini adalah penggunaan teknik Ensemble Learning (Soft Voting) yang menggabungkan dua algoritma dengan karakteristik berbeda untuk menghasilkan prediksi yang lebih stabil dan akurat.🚀 Fitur UtamaEnsemble Prediction: Menggabungkan probabilitas dari XGBoost (pola kompleks) dan Logistic Regression (stabilitas linear).Professional Pipeline: Prapemrosesan data (imputasi dan encoding) dilakukan secara otomatis dan terintegrasi.Interactive Dashboard: Visualisasi EDA (Exploratory Data Analysis) untuk memahami faktor pendorong churn seperti komplain dan skor kepuasan.Model Explainability: Interpretasi fitur menggunakan SHAP Values untuk transparansi hasil prediksi.📑 Metodologi KerjaSistem ini bekerja melalui empat tahap utama yang selaras dengan siklus penambangan data:Preprocessing: Menggunakan Scikit-learn Pipeline untuk imputasi median pada data numerik dan One-Hot Encoding pada data kategorikal.Modeling: Melatih model XGBoost dengan optimasi Grid Search CV dan Logistic Regression sebagai baseline.Ensemble: Menerapkan Soft Voting untuk mengambil rata-rata probabilitas dari kedua model.Evaluation: Menggunakan metrik F1-Score sebagai acuan utama karena ketidakseimbangan kelas (Imbalanced Class) pada data churn.📁 Struktur DirektoriPlaintextUAS_DATA_MINING/
├── app/
│   └── app.py              # Aplikasi Dashboard Streamlit
├── data/
│   ├── raw/                # Dataset mentah (dataset.csv)
│   └── preprocess/         # Dataset hasil pembersihan
├── models/
│   ├── ensemble_model.pkl  # Model gabungan utama
│   ├── xgb_model.pkl       # Model XGBoost individu
│   ├── logreg_model.pkl    # Model Logistic Regression individu
│   └── column_names.pkl    # Metadata nama fitur
├── notebooks/
│   └── 02_modeling.ipynb   # Eksperimen pemodelan
├── reports/
│   └── shap_summary.png    # Visualisasi interpretasi model
├── src/
│   ├── data_preprocessing.py
│   └── modeling.py         # Skrip pelatihan model ensemble
├── requirements.txt
└── README.md
🛠️ Instalasi & PenggunaanClone Repositori:Bashgit clone https://github.com/username/UAS_DATA_MINING.git
cd UAS_DATA_MINING
Instal Dependensi:Bashpip install -r requirements.txt
Pelatihan Model:Bashpython src/modeling.py
Jalankan Aplikasi:Bashstreamlit run app/app.py
📊 Evaluasi ModelModel dievaluasi menggunakan metrik keseimbangan antara Precision dan Recall:$$F1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$👤 Identitas PengembangNama: [Isi Nama Anda]Mata Kuliah: Penambangan DataSemester: Ganjil 2025/2026Tips Terakhir:Jangan lupa untuk menyertakan link video YouTube Anda di bagian bawah README tersebut sebagai bagian dari persyaratan dokumentasi UAS.
