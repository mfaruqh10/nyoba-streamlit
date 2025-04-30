import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# Set page config
st.set_page_config(page_title="Wine Classification", layout="wide")
st.write("""
# Welcome to my machine learning dashboard

This dashboard created by : [Muhammad Faruq Hizburrabbani](https://www.linkedin.com/in/muhammad-faruq-hizburrabbani-59872734a/)
""")

# Load data
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    df['wine_class'] = df['target'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
    return df, wine

df, wine = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Modeling", "Evaluation"])

# Main content
st.title("Wine Classification Analysis")
st.write(f"Dataset shape: {df.shape}")

if page == "Data Overview":
    st.header("Data Overview")
    
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Description")
    st.write("""
    Dataset ini adalah hasil analisis kimia dari wine yang ditanam di wilayah yang sama di Italia 
    oleh tiga pembudidaya berbeda. Terdapat tiga jenis wine yang dianalisis dengan 13 pengukuran 
    berbeda untuk berbagai konstituen yang ditemukan dalam wine-wine tersebut.
    
    **Karakteristik Dataset:**
    - Jumlah Instansi: 178
    - Jumlah Atribut: 13 atribut numerik prediktif dan kelas
                
    **Informasi Atribut:**
    1. Alkohol
    2. Asam malat
    3. Abu
    4. Alkalinitas abu
    5. Magnesium
    6. Total fenol
    7. Flavanoid
    8. Fenol nonflavanoid
    9. Proantosianin
    10. Intensitas warna
    11. Hue (warna)
    12. OD280/OD315 dari wine yang diencerkan
    13. Prolin
    14. Kelas:
       - class_0
       - class_1
       - class_2
    
    **Informasi Tambahan:**
    - Nilai Atribut yang Hilang: Tidak ada""")
    
    st.subheader("Class Distribution")
    st.bar_chart(df['wine_class'].value_counts())
    st.write("""
    **Distribusi Kelas Dataset Wine:**
    Total sampel: 178 sampel wine dari 3 produsen berbeda
    
    **Pembagian kelas:**
             Class_1 (mayoritas): 71 sampel (39.9%)
             Class_0: 59 sampel (33.1%)
             Class_2 (minoritas): 48 sampel (27%)
    """)

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    st.subheader("Numerical Summary")
    st.write(df.describe())
    
    st.subheader("Correlation Heatmap")
    st.write("""Digunakan Correlation Heatmap untuk mengidentifikasi hubungan antarvariabel, 
             menunjukkan korelasi (linear) antara setiap pasangan fitur numerik, dan seleksi fitur""")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    st.subheader("Feature Distribution by Class")
    st.write("""Feature Distribution digunakan untuk memvisualisasikan distribusi setiap fitur untuk tiap kelas wine (class_0, class_1, class_2) 
             dan mengidentifikasi pola pembeda antarkelas melalui karakteristik kimia wine""")
    feature = st.selectbox("Select feature to visualize", wine.feature_names)
    
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x='wine_class', y=feature, ax=ax)
    st.pyplot(fig)

elif page == "Modeling":
    st.header("Model Training")
    st.write("""Disini digunakan 3 model: random forest, SVM, dan logistic regression""")
    
    # Split data
    X = df[wine.feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection
    model_type = st.selectbox("Select Model", 
                            ["Random Forest", "SVM", "Logistic Regression"])
    
    if model_type == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier(random_state=42)
    elif model_type == "SVM":
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['linear', 'rbf']
        }
        model = SVC(random_state=42)
    else:  # Logistic Regression
        param_grid = {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Hyperparameter tuning
    st.subheader("Hyperparameter Tuning")
    st.write("""Disini digunakan hyperparameter tuning untuk mencari kombinasi parameter terbaik untuk algoritma yang dipilih (Random Forest, SVM, atau Logistic Regression)
              dan mengoptimalkan performa model tanpa overfitting""")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    
    with st.spinner('Training model...'):
        grid_search.fit(X_train, y_train)
    
    st.success("Model training completed!")
    
    st.write("Best parameters:", grid_search.best_params_)
    st.write("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    
    # Save best model
    best_model = grid_search.best_estimator_
    st.session_state['best_model'] = best_model
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test

elif page == "Evaluation":
    st.header("Model Evaluation")
    
    if 'best_model' not in st.session_state:
        st.warning("Please train a model first on the Modeling page.")
    else:
        best_model = st.session_state['best_model']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.subheader("Performance Metrics")
        st.write(f"Test Accuracy: {accuracy:.2f}")
        
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        st.subheader("Confusion Matrix")
        st.write(""" Confusion matrix menggambarkan performa model klasifikasi dalam memprediksi tiga jenis wine (kelas 0, 1, dan 2) dengan sangat akurat. Dari total 36 sampel yang diuji, model berhasil memprediksi dengan benar 35 sampel, menghasilkan akurasi sebesar 97.2%. Khusus untuk kelas 2, model menunjukkan performa sempurna dengan semua 8 sampel teridentifikasi tepat. Kelas 1 juga mencapai prediksi sempurna dimana seluruh 14 sampel dikenali dengan benar. Hanya terdapat satu kesalahan klasifikasi, yaitu satu sampel dari kelas 0 yang salah diprediksi sebagai kelas 1. Hasil ini mengindikasikan bahwa karakteristik kimia wine kelas 2 sangat unik sehingga mudah dibedakan, sementara kelas 0 dan 1 memiliki beberapa kesamaan fitur yang menyebabkan sedikit tumpang tindih dalam klasifikasi. Meskipun demikian, tingkat akurasi yang mencapai 97.2% menunjukkan bahwa model ini sudah sangat handal untuk tugas klasifikasi wine. """)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
        st.pyplot(fig)
        
        st.subheader("Feature Importance")
        st.write(""" Visualisasi feature importance ini mengungkapkan kontribusi relatif setiap fitur kimia dalam model klasifikasi wine, di mana flavanoids muncul sebagai fitur paling determinan dengan skor importance sekitar 0.175. Dominasi flavanoids menunjukkan bahwa senyawa fenolik ini menjadi pembeda utama antar kelas wine, kemungkinan karena kadar kandungannya sangat bervariasi tergantung produsen. Fitur color intensity (0.125) dan proline (0.1) menempati posisi berikutnya, mengindikasikan bahwa intensitas warna dan kadar asam amino proline juga berperan krusial dalam membedakan karakteristik wine. Sementara itu, alcohol (0.075) dan OD280/OD315 (ukuran protein) termasuk fitur pendukung yang cukup berpengaruh, meskipun tidak sekuat tiga fitur utama. """)
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            feat_imp = pd.DataFrame({'Feature': wine.feature_names, 'Importance': importance})
            feat_imp = feat_imp.sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots()
            sns.barplot(data=feat_imp, x='Importance', y='Feature', ax=ax)
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model type.")