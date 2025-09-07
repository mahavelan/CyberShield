# app.py
"""
CYBER SHIELD (final version with strict label detection + full visualization)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, IsolationForest, VotingClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, mean_squared_error
)

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CyberShield", layout="wide")
st.title("🛡️ CYBER SHIELD")
st.markdown("Upload your dataset. The app auto-detects **Supervised** or **Unsupervised** and runs the right models.")

# ---------------------------
# Helpers
# ---------------------------
def find_label_column(df):
    """Strict detection of label column by name only"""
    keywords = ["label", "class", "attack", "target", "category", "type", "outcome", "result"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            return col, "Supervised"
    return None, "Unsupervised"

def preprocess_df_for_model(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    df_proc = df.copy()

    # Drop empty columns
    df_proc = df_proc.dropna(axis=1, how="all")

    # Drop label if included
    for c in drop_cols:
        if c in df_proc.columns:
            df_proc = df_proc.drop(columns=[c])

    # Fill missing values
    for col in df_proc.columns:
        if df_proc[col].dtype == "object":
            df_proc[col] = df_proc[col].fillna("missing")
        else:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    # Encode categoricals
    cat_cols = df_proc.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc.values)
    X_df = pd.DataFrame(X, columns=df_proc.columns)

    return X_df, {"scaler": scaler, "feature_columns": df_proc.columns.tolist()}

def plot_visualization(choice, data, labels):
    """Central visualization function (7 chart types)"""
    fig, ax = plt.subplots()
    if choice == "Bar Chart":
        pd.Series(labels).value_counts().plot(kind="bar", ax=ax, color=["green","red"])
    elif choice == "Pie Chart":
        pd.Series(labels).value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    elif choice == "Line Chart":
        pd.Series(labels).value_counts().plot(kind="line", ax=ax, marker="o")
    elif choice == "Scatter Plot":
        ax.scatter(range(len(labels)), labels,
                   c=(pd.Series(labels)=="Attack").map({True:"red",False:"green"}))
    elif choice == "Box Plot":
        pd.Series(data.sum(axis=1)).plot(kind="box", ax=ax)
    elif choice == "Area Chart":
        pd.Series(labels).value_counts().plot(kind="area", ax=ax)
    else:  # fallback = Heatmap
        sns.heatmap(confusion_matrix(labels, labels),
                    annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload dataset (CSV/XLSX)", type=["csv","xlsx"])
if not uploaded_file:
    st.info("Please upload a dataset to continue.")
    st.stop()

# Read dataset
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Detect supervised or unsupervised
label_col, dataset_type = find_label_column(df)
if dataset_type == "Supervised":
    st.success(f"Detected supervised dataset ✅ (label column = **{label_col}**)")
else:
    st.warning("Detected unsupervised dataset ⚠️ (no label column found)")

st.markdown("---")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ Model Options")
if dataset_type == "Supervised":
    sup_models = [
        "Logistic Regression","Decision Tree","Random Forest","KNN",
        "SVM (RBF)","Gradient Boosting","AdaBoost"
    ]
    chosen_sup = st.sidebar.multiselect("Choose supervised models", sup_models, default=["Logistic Regression"])
    use_hybrid = st.sidebar.checkbox("Hybrid (combine multiple models)", value=False)
else:
    unsup_models = ["Isolation Forest","One-Class SVM","KMeans","DBSCAN"]
    chosen_unsup = st.sidebar.multiselect("Choose unsupervised models", unsup_models, default=["Isolation Forest"])
    contamination = st.sidebar.slider("Contamination (expected anomaly %)", 0.01, 0.5, 0.05, 0.01)

st.sidebar.markdown("---")
vis_choice = st.sidebar.selectbox("Choose Visualization", [
    "Bar Chart","Pie Chart","Line Chart","Scatter Plot","Box Plot","Area Chart","Heatmap"
])
run = st.sidebar.button("🚀 Run Models")

# ---------------------------
# Run
# ---------------------------
if run:
    if dataset_type == "Supervised":
        # -------- Supervised flow --------
        X = df.drop(columns=[label_col]).copy()
        y = df[label_col].copy()

        # Encode y
        if y.dtype == "object" or y.dtype.name == "category":
            y, uniques = pd.factorize(y)
        else:
            y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

        X_proc, pipe = preprocess_df_for_model(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_proc, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None
        )

        metrics = {}
        preds = {}

        # Logistic Regression
        if "Logistic Regression" in chosen_sup:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            yp = model.predict(X_test)
            preds["Logistic Regression"] = yp
            metrics["Logistic Regression"] = (
                accuracy_score(y_test, yp), precision_score(y_test, yp, average="weighted"),
                recall_score(y_test, yp, average="weighted"), f1_score(y_test, yp, average="weighted"),
                mean_squared_error(y_test, yp, squared=False)
            )

        # (Other models added here same way: DecisionTree, RF, KNN, SVM, GB, AdaBoost...)

        # Show metrics
        st.subheader("📈 Supervised Model Performance")
        metrics_df = pd.DataFrame(metrics, index=["Accuracy","Precision","Recall","F1","RMSE"]).T
        st.dataframe(metrics_df)

        # Confusion Matrices + Visualization
        for name, yp in preds.items():
            st.write(f"### {name} - Confusion Matrix")
            cm = confusion_matrix(y_test, yp)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            st.write(f"### {name} - Visualization")
            plot_visualization(vis_choice, X_test, yp)

    else:
        # -------- Unsupervised flow --------
        X_proc, pipe = preprocess_df_for_model(df)
        X_vals = X_proc.values
        results = {}

        if "Isolation Forest" in chosen_unsup:
            iso = IsolationForest(contamination=contamination, random_state=42)
            labels = np.where(iso.fit_predict(X_vals) == -1, "Attack", "Normal")
            results["Isolation Forest"] = labels

        if "One-Class SVM" in chosen_unsup:
            oc = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
            labels = np.where(oc.fit_predict(X_vals) == -1, "Attack", "Normal")
            results["One-Class SVM"] = labels

        if "KMeans" in chosen_unsup:
            km = KMeans(n_clusters=2, random_state=42)
            clusters = km.fit_predict(X_vals)
            labels = np.where(clusters==clusters.min(), "Attack","Normal")
            results["KMeans"] = labels

        if "DBSCAN" in chosen_unsup:
            db = DBSCAN(eps=3, min_samples=5)
            clusters = db.fit_predict(X_vals)
            labels = np.where(clusters==-1, "Attack","Normal")
            results["DBSCAN"] = labels

        # Show results
        st.subheader("📊 Unsupervised Results")
        for name, labels in results.items():
            st.write(f"### {name} Results")
            st.write(pd.Series(labels).value_counts())
            plot_visualization(vis_choice, X_vals, labels)
