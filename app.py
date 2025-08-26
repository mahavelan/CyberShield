# app.py
import streamlit as st
import pandas as pd

# ------------------------
# Title & Description
# ------------------------
st.set_page_config(page_title="Intrusion Detection Web App", layout="wide")
st.title("🚨 Intrusion Detection Web App")
st.write("""
This app detects whether uploaded datasets are **Labeled (Supervised)** or **Unlabeled (Unsupervised)** 
and allows users to choose machine learning / deep learning models for cyber attack detection.
""")

# ------------------------
# File Upload
# ------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Show preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))

    # ------------------------
    # Auto-detect dataset type
    # ------------------------
    label_cols = ['label', 'class', 'attack', 'target']
    found = [col for col in df.columns if col.lower() in label_cols]

    if found:
        dataset_type = "Labeled (Supervised)"
    else:
        dataset_type = "Unlabeled (Unsupervised)"

    st.info(f"🔎 Detected dataset type: **{dataset_type}**")

    # ------------------------
    # Model Selection
    # ------------------------
    st.subheader("🤖 Choose Model")

    if dataset_type == "Labeled (Supervised)":
        model_choice = st.selectbox(
            "Select a Supervised Model",
            ["Logistic Regression", "Random Forest", "SVM", "KNN", "CNN", "LSTM", "BiLSTM", "Hybrid (Ensemble)"]
        )
    else:
        model_choice = st.selectbox(
            "Select an Unsupervised Model",
            ["Autoencoder", "One-Class SVM", "Isolation Forest", "KMeans/DBSCAN", "Hybrid (Consensus)"]
        )

    st.success(f"✅ You selected: **{model_choice}**")

    # ------------------------
    # Placeholder for Next Steps
    # ------------------------
    st.subheader("⚡ Next Step (Coming Soon)")
    st.write("""
    After selecting a model, the app will:
    - Run pre-trained models on your dataset
    - Show Accuracy, Precision, Recall, and F1 (for labeled data)
    - Show Anomaly Detection results (for unlabeled data)
    - Provide Visualizations and Explainable AI insights
    - Allow download of results
    """)

else:
    st.warning("⚠️ Please upload a dataset to continue.")
