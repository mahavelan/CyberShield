# app.py
"""
CYBER SHIELD (Final Version 🚀)
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
# Utility Functions
# ---------------------------
def find_label_candidates(df, keywords=None):
    if keywords is None:
        keywords = ["label","class","attack","target","category","type","y","outcome","result"]
    scores = {}
    for col in df.columns:
        norm = re.sub(r'[^0-9a-zA-Z]', ' ', str(col)).lower().strip()
        score = sum(1 for kw in keywords if kw in norm)
        scores[col] = score
    sorted_cands = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_cands

def clean_labels(y_raw):
    try:
        if y_raw.dtype == "object" or y_raw.dtype.name == "category":
            y, _ = pd.factorize(y_raw)
            return y, "categorical"
        y = pd.to_numeric(y_raw, errors="coerce")
        if y.isna().any():
            y = y.fillna(0)
        return y.astype(int), "numeric"
    except Exception as e:
        st.warning(f"⚠️ Label cleaning failed: {e}. Switching to Unsupervised.")
        return None, "failed"

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
            ax.scatter(data.iloc[:,0], data.iloc[:,1], c=labels, cmap="coolwarm")
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
    st.info("Upload dataset to begin. Use demo data if needed.")
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
    candidates = find_label_candidates(df)
    label_col = candidates[0][0] if candidates and candidates[0][1] > 0 else None
    if label_col:
        y, label_type = clean_labels(df[label_col])
        if y is None:
            dataset_type = "Unsupervised"
        else:
            dataset_type = "Supervised"
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
        if st.sidebar.checkbox("Hybrid (combine selected models)"):
            chosen_hybrid = chosen
        else:
            chosen_hybrid = None
    else:
        models_unsup = [
            "Isolation Forest","One-Class SVM","KMeans","DBSCAN"
        ]
        chosen = st.sidebar.multiselect("Choose unsupervised models", models_unsup, default=["Isolation Forest"])
        if st.sidebar.checkbox("Hybrid (combine selected models)"):
            chosen_hybrid = chosen
        else:
            chosen_hybrid = None
        contamination = st.sidebar.slider("Anomaly contamination (Isolation/OCSVM)", 0.01, 0.5, 0.05, 0.01)

    viz_choice = st.sidebar.selectbox("Choose Visualization", viz_options)
    run = st.sidebar.button("🚀 Run Models")

    # ---------------------------
    # Execution
    # ---------------------------
    if run:
        if dataset_type == "Supervised":
            X = df.drop(columns=[label_col])
            X_processed, scaler = preprocess_data(X)
            y, _ = clean_labels(df[label_col])

            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.3, random_state=42, stratify=y
            )

            results = {}
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
                results[model] = (acc, prec, rec, f1, rmse)

                st.subheader(f"📌 {model}")
                st.write(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, RMSE: {rmse:.3f}")

                plot_visualization(y_test, y_pred, viz_choice, title=f"{model} - {viz_choice}")

            # Hybrid (Voting)
            if chosen_hybrid and len(chosen_hybrid) > 1:
                st.subheader("🤝 Hybrid Model (Voting)")
                estimators = []
                for m in chosen_hybrid:
                    if m == "Logistic Regression":
                        estimators.append(("lr", LogisticRegression(max_iter=1000)))
                    elif m == "Random Forest":
                        estimators.append(("rf", RandomForestClassifier()))
                    elif m == "SVM (RBF)":
                        estimators.append(("svm", SVC(probability=True)))
                if estimators:
                    vote = VotingClassifier(estimators=estimators, voting="soft")
                    vote.fit(X_train, y_train)
                    y_pred = vote.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.write(f"Hybrid Accuracy: {acc:.3f}")
                    plot_visualization(y_test, y_pred, viz_choice, title="Hybrid - "+viz_choice)

        else:  # Unsupervised
            X_processed, scaler = preprocess_data(df)
            results = {}
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

            # Hybrid Unsupervised → Majority Voting
            if chosen_hybrid and len(chosen_hybrid) > 1:
                st.subheader("🤝 Hybrid Unsupervised")
                votes = pd.DataFrame()
                for m in chosen_hybrid:
                    if m == "Isolation Forest":
                        clf = IsolationForest(contamination=contamination, random_state=42)
                        votes[m] = np.where(clf.fit_predict(X_processed) == -1, "Attack", "Normal")
                    elif m == "One-Class SVM":
                        clf = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
                        votes[m] = np.where(clf.fit_predict(X_processed) == -1, "Attack", "Normal")
                    elif m == "KMeans":
                        clf = KMeans(n_clusters=2, random_state=42)
                        votes[m] = np.where(clf.fit_predict(X_processed) == 1, "Attack", "Normal")
                final_pred = votes.mode(axis=1)[0]
                st.write(final_pred.value_counts())
                plot_visualization(df, final_pred, viz_choice, title="Hybrid Unsupervised - "+viz_choice)

# ---------------------------
# Help
# ---------------------------
st.markdown("---")
with st.expander("ℹ️ Help & Quickstart"):
    st.write("""
    1. Upload your dataset (CSV/XLSX).
    2. The app auto-detects if it's **Supervised** (has labels) or **Unsupervised**.
    3. Choose models from the sidebar (Hybrid = multiple models combined).
    4. Run → View metrics, predictions, and visualizations.
    5. Download results if needed.
    """)
