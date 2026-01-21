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

# ==========================================
# 1. SETUP DIREKTORI & DATA LOADING
# ==========================================
print("1. Menyiapkan Lingkungan & Memuat Data...")
for folder in ['models', 'reports']:
    if not os.path.exists(folder):
        os.makedirs(folder)

data_path = 'data/preprocess/dataset_cleaned.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan di {data_path}")

df = pd.read_csv(data_path)
X = df.drop(['ID_Pelanggan', 'Churn'], axis=1)
y = df['Churn']

# Definisi Fitur (13 Fitur Lengkap)
numeric_features = ['Lama_Berlangganan', 'Jarak_Gudang_ke_Rumah', 'Jam_di_Aplikasi', 
                    'Skor_Kepuasan', 'Hari_Sejak_Pesanan_Terakhir', 'Jumlah_Cashback', 'Tingkat_Kota']
categorical_features = ['Perangkat_Login', 'Metode_Pembayaran', 'Jenis_Kelamin', 'Status_Perkawinan', 'Komplain']

# ==========================================
# 2. PREPROCESSING PIPELINE
# ==========================================
# Preprocessor untuk angka dan kategori
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Split Data (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# 3. TRAINING INDIVIDUAL MODELS & TUNING
# ==========================================
print("\n2. Melatih & Optimasi Model Individu...")

# --- Model A: Logistic Regression ---
logreg_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])
logreg_pipe.fit(X_train, y_train)

# --- Model B: XGBoost dengan GridSearchCV ---
xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.1]
}

grid_xgb = GridSearchCV(xgb_pipe, param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_

# ==========================================
# 4. IMPLEMENTASI ENSEMBLE (SOFT VOTING)
# ==========================================
print("\n3. Menggabungkan Model menjadi Ensemble (Soft Voting)...")

ensemble_model = VotingClassifier(
    estimators=[
        ('xgb', best_xgb.named_steps['classifier']),
        ('logreg', logreg_pipe.named_steps['classifier'])
    ],
    voting='soft'
)

# Pipeline Akhir untuk Ensemble
final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('ensemble', ensemble_model)
])

final_pipeline.fit(X_train, y_train)

# ==========================================
# 5. EVALUASI KOMPREHENSIF
# ==========================================
print("\n4. Evaluasi Performa Model...")

def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"\n[{name}] Classification Report:")
    print(classification_report(y_test, y_pred))
    return f1_score(y_test, y_pred)

f1_xgb = evaluate(best_xgb, "XGBoost Only")
f1_ens = evaluate(final_pipeline, "Ensemble (XGB + LogReg)")

# Confusion Matrix untuk Ensemble
cm = confusion_matrix(y_test, final_pipeline.predict(X_test))
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Ensemble Model")
plt.savefig('reports/confusion_matrix_ensemble.png')

# ==========================================
# 6. INTERPRETASI SHAP (KOMPONEN XGBOOST)
# ==========================================
print("\n5. Menghasilkan Interpretasi SHAP...")
try:
    # Transformasi data test untuk SHAP
    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"): X_test_transformed = X_test_transformed.toarray()
    
    # Dapatkan nama kolom fitur
    ohe_features = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(ohe_features)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    # SHAP untuk XGBoost
    explainer = shap.TreeExplainer(best_xgb.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test_df)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_df, show=False)
    plt.savefig('reports/shap_summary_ensemble.png', bbox_inches='tight')
    print("✅ SHAP Plot berhasil disimpan.")
except Exception as e:
    print(f"⚠️ SHAP Error: {e}")

# ==========================================
# 7. PENYIMPANAN ARTEFAK
# ==========================================
joblib.dump(final_pipeline, 'models/ensemble_model.pkl')
joblib.dump(best_xgb, 'models/xgb_model.pkl')
joblib.dump(logreg_pipe, 'models/logreg_model.pkl')
joblib.dump(X.columns.tolist(), 'models/column_names.pkl')

print("\n🚀 [SELESAI] Semua model dan laporan telah disimpan di folder 'models/' dan 'reports/'.")