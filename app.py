# app.py
"""
CYBER SHIELD — Final Stable Version 🚀
Mini Project Web App
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    IsolationForest, VotingClassifier
)
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------
# Label Auto-Detection
# ---------------------------
def detect_label_column(df):
    keywords = ["label","class","attack","target","category","type","y","outcome","result"]
    candidates = []

    for col in df.columns:
        norm = re.sub(r'[^0-9a-zA-Z]', ' ', str(col)).lower().strip()
        # name match
        score = sum(1 for kw in keywords if kw in norm)

        nunique = df[col].nunique(dropna=True)
        # binary/low unique
        if nunique <= 10:
            score += 2
        elif nunique <= max(0.05 * len(df), 20):
            score += 1

        candidates.append((col, score))

    # sort by score
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    best = candidates[0] if candidates else None
    if best and best[1] > 0:
        return best[0]  # column name
    else:
        return None

def clean_labels(y_raw):
    try:
        if y_raw.dtype == "object" or y_raw.dtype.name == "category":
            y, _ = pd.factorize(y_raw)
            return y
        y = pd.to_numeric(y_raw, errors="coerce")
        if y.isna().any():
            y = y.fillna(0)
        return y.astype(int)
    except:
        return None

def preprocess_data(df, drop_cols=None):
    df = df.copy()
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
    # drop fully empty
    df = df.dropna(axis=1, how="all")
    # encode categoricals
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # fill missing
    df = df.fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)
    return pd.DataFrame(X, columns=df.columns), scaler

def plot_visualization(data, labels, viz_type, title="Visualization"):
    fig, ax = plt.subplots()
    if viz_type == "Bar Chart":
        pd.Series(labels).value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
    elif viz_type == "Pie Chart":
        pd.Series(labels).value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    elif viz_type == "Heatmap":
        cm = confusion_matrix(data, labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    elif viz_type == "Line Chart":
        pd.Series(labels).reset_index(drop=True).plot(kind="line", ax=ax)
    elif viz_type == "Area Chart":
        pd.Series(labels).reset_index(drop=True).plot(kind="area", ax=ax)
    elif viz_type == "Scatter Plot":
        if isinstance(data, pd.DataFrame) and data.shape[1] >= 2:
            ax.scatter(data.iloc[:,0], data.iloc[:,1], c=pd.factorize(labels)[0], cmap="coolwarm")
        else:
            st.warning("Scatter requires at least 2 features.")
    ax.set_title(title)
    st.pyplot(fig)

# ---------------------------
# App Layout
# ---------------------------
st.set_page_config(page_title="CYBER SHIELD", layout="wide")
st.title("🛡️ CYBER SHIELD — Intrusion Detection Web App")

uploaded_file = st.file_uploader("📂 Upload dataset (CSV/XLSX)", type=["csv","xlsx"])

if not uploaded_file:
    st.info("Upload dataset to begin.")
else:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10))

    # ---------------------------
    # Auto-detect supervised vs unsupervised
    # ---------------------------
    label_col = detect_label_column(df)
    if label_col:
        y = clean_labels(df[label_col])
        if y is not None:
            dataset_type = "Supervised"
        else:
            dataset_type = "Unsupervised"
    else:
        dataset_type = "Unsupervised"

    st.success(f"🔍 Auto-detected dataset type: **{dataset_type}**")
    if dataset_type == "Supervised":
        st.info(f"Using label column: **{label_col}**")

    # ---------------------------
    # Model Selection
    # ---------------------------
    st.sidebar.header("⚙️ Options")
    viz_options = ["Bar Chart","Pie Chart","Heatmap","Line Chart","Area Chart","Scatter Plot"]

    if dataset_type == "Supervised":
        models_sup = [
            "Logistic Regression","Decision Tree","Random Forest","KNN",
            "SVM (RBF)","Naive Bayes","Gradient Boosting","AdaBoost","Ridge Classifier"
        ]
        chosen = st.sidebar.multiselect("Choose supervised models", models_sup, default=["Logistic Regression"])
        chosen_hybrid = st.sidebar.multiselect("Hybrid (combine models)", models_sup)
    else:
        models_unsup = [
            "Isolation Forest","One-Class SVM","KMeans","DBSCAN"
        ]
        chosen = st.sidebar.multiselect("Choose unsupervised models", models_unsup, default=["Isolation Forest"])
        chosen_hybrid = st.sidebar.multiselect("Hybrid (combine models)", models_unsup)
        contamination = st.sidebar.slider("Anomaly contamination (for IsoForest/OCSVM)", 0.01, 0.5, 0.05, 0.01)

    viz_choice = st.sidebar.selectbox("Choose Visualization", viz_options)
    run = st.sidebar.button("🚀 Run Models")

    # ---------------------------
    # Execution
    # ---------------------------
    if run:
        if dataset_type == "Supervised":
            X = df.drop(columns=[label_col])
            X_processed, scaler = preprocess_data(X)
            y = clean_labels(df[label_col])

            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.3, random_state=42, stratify=y
            )

            for model in chosen:
                if model == "Logistic Regression":
                    clf = LogisticRegression(max_iter=1000)
                elif model == "Decision Tree":
                    clf = DecisionTreeClassifier()
                elif model == "Random Forest":
                    clf = RandomForestClassifier()
                elif model == "KNN":
                    clf = KNeighborsClassifier()
                elif model == "SVM (RBF)":
                    clf = SVC(kernel="rbf", probability=True)
                elif model == "Naive Bayes":
                    clf = GaussianNB()
                elif model == "Gradient Boosting":
                    clf = GradientBoostingClassifier()
                elif model == "AdaBoost":
                    clf = AdaBoostClassifier()
                elif model == "Ridge Classifier":
                    clf = RidgeClassifier()
                else:
                    continue

                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                rmse = mean_squared_error(y_test, y_pred, squared=False)

                st.subheader(f"📌 {model}")
                st.write(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, RMSE: {rmse:.3f}")
                plot_visualization(y_test, y_pred, viz_choice, title=f"{model} - {viz_choice}")

        else:  # Unsupervised
            X_processed, scaler = preprocess_data(df)

            for model in chosen:
                if model == "Isolation Forest":
                    clf = IsolationForest(contamination=contamination, random_state=42)
                    y_pred = clf.fit_predict(X_processed)
                elif model == "One-Class SVM":
                    clf = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
                    y_pred = clf.fit_predict(X_processed)
                elif model == "KMeans":
                    clf = KMeans(n_clusters=2, random_state=42)
                    y_pred = clf.fit_predict(X_processed)
                elif model == "DBSCAN":
                    clf = DBSCAN()
                    y_pred = clf.fit_predict(X_processed)
                else:
                    continue

                y_pred = np.where(y_pred == -1, "Attack", "Normal")
                st.subheader(f"📌 {model}")
                st.write(pd.Series(y_pred).value_counts())
                plot_visualization(df, y_pred, viz_choice, title=f"{model} - {viz_choice}")
