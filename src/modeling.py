import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Sklearn & XGBoost Modules
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)

def run_modeling_pipeline():
    """
    Menjalankan alur kerja Modeling & Evaluation secara end-to-end.
    Mencakup Preprocessing, Tuning, Ensemble, dan Interpretasi.
    """
    print("1. Menyiapkan Lingkungan & Memuat Data...")
    
    # Membuat folder output jika belum ada (Soal 5)
    for folder in ['models', 'reports']:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Memuat dataset hasil preprocessing (Soal 2)
    data_path = 'data/preprocess/dataset_cleaned.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}. Jalankan preprocessing terlebih dahulu.")

    df = pd.read_csv(data_path)
    X = df.drop(['ID_Pelanggan', 'Churn'], axis=1)
    y = df['Churn']

    # Definisi 13 Fitur (Soal 1 & 4)
    numeric_features = ['Lama_Berlangganan', 'Jarak_Gudang_ke_Rumah', 'Jam_di_Aplikasi', 
                        'Skor_Kepuasan', 'Hari_Sejak_Pesanan_Terakhir', 'Jumlah_Cashback', 'Tingkat_Kota']
    categorical_features = ['Perangkat_Login', 'Metode_Pembayaran', 'Jenis_Kelamin', 'Status_Perkawinan', 'Komplain']

    # ==========================================
    # 2. PREPROCESSING PIPELINE (Soal 2.4)
    # ==========================================
    # Pipeline untuk data numerik (Imputasi Median + Scaling)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk data kategorikal (Imputasi Modus + One-Hot Encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Menggabungkan transformer ke dalam ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Pembagian data (80% Train, 20% Test) - Soal 2.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # 3. MODELING & TUNING (Soal 3.1 & 3.2)
    # ==========================================
    print("\n2. Melatih & Optimasi Model (LogReg & XGBoost)...")

    # --- Model 1: Logistic Regression (Baseline) ---
    logreg_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    logreg_pipe.fit(X_train, y_train)

    # --- Model 2: XGBoost dengan GridSearchCV (Optimasi) ---
    xgb_pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])

    # Parameter grid untuk tuning (Soal 3.2)
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1]
    }

    grid_xgb = GridSearchCV(xgb_pipe, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    best_xgb_clf = grid_xgb.best_estimator_.named_steps['classifier']

    # ==========================================
    # 4. ENSEMBLE LEARNING (SOFT VOTING)
    # ==========================================
    print("\n3. Membangun Ensemble Model (Soft Voting)...")

    # Menggabungkan kedua model (Soal 3.5)
    ensemble_clf = VotingClassifier(
        estimators=[
            ('xgb', best_xgb_clf),
            ('logreg', logreg_pipe.named_steps['classifier'])
        ],
        voting='soft'
    )

    # Pipeline Akhir untuk Ensemble
    ensemble_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ensemble', ensemble_clf)
    ])

    ensemble_pipeline.fit(X_train, y_train)

    # ==========================================
    # 5. EVALUASI (Soal 3.3)
    # ==========================================
    print("\n4. Mengevaluasi Performa Model...")

    def get_report(model, name):
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred))
        return f1

    get_report(logreg_pipe, "Logistic Regression")
    get_report(grid_xgb.best_estimator_, "XGBoost (Tuned)")
    f1_ens = get_report(ensemble_pipeline, "Ensemble (Final)")

    # Simpan Confusion Matrix Plot
    cm = confusion_matrix(y_test, ensemble_pipeline.predict(X_test))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix Ensemble (F1: {f1_ens:.2f})")
    plt.savefig('reports/confusion_matrix.png')

    # ==========================================
    # 6. INTERPRETASI (SHAP) - Soal 3.4
    # ==========================================
    print("\n5. Menghasilkan Interpretasi SHAP...")
    try:
        # Transformasi data test agar sesuai format model
        X_test_transformed = preprocessor.transform(X_test)
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()
        
        # Dapatkan nama fitur setelah encoding
        ohe_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
        all_feature_names = numeric_features + list(ohe_names)
        
        X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names)

        # Gunakan model XGBoost dari ensemble untuk SHAP (karena Tree-based)
        explainer = shap.TreeExplainer(best_xgb_clf)
        shap_values = explainer.shap_values(X_test_df)
        
        plt.figure()
        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.savefig('reports/shap_summary.png', bbox_inches='tight')
    except Exception as e:
        print(f"Peringatan SHAP: {e}")

    # ==========================================
    # 7. PENYIMPANAN ARTEFAK (Soal 4 & 5)
    # ==========================================
    joblib.dump(ensemble_pipeline, 'models/ensemble_model.pkl')
    # Simpan model individu sebagai cadangan pembanding di app.py
    joblib.dump(grid_xgb.best_estimator_, 'models/xgb_model.pkl')
    joblib.dump(logreg_pipe, 'models/logreg_model.pkl')
    joblib.dump(X.columns.tolist(), 'models/column_names.pkl')

    print("\n🚀 [SELESAI] Semua artefak model disimpan di folder 'models/'.")

if __name__ == "__main__":
    run_modeling_pipeline()