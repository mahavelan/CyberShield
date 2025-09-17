import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Supervised models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression

# Unsupervised models
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------- HELPERS -------------------

def normalize_labels(y):
    """Map dataset labels into Normal vs Attack"""
    if y is None:
        return None
    y_norm = []
    for val in y:
        val_str = str(val).lower()
        if val_str in ["benign", "normal", "s", "0"]:
            y_norm.append("Normal")
        else:
            y_norm.append("Attack")
    return np.array(y_norm)

def preprocess_data(df, label_col=None):
    """Preprocess dataset: handle categorical/numeric, impute, scale"""
    if label_col and label_col in df.columns:
        y = df[label_col]
        X = df.drop(columns=[label_col])
    else:
        y = None
        X = df.copy()

    # Replace infinities
    X = X.replace([np.inf, -np.inf], np.nan)

    # Separate categorical & numeric
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    X_proc = preprocessor.fit_transform(X)
    return X_proc, y, df

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def plot_distribution(labels, title="Data Distribution"):
    fig, ax = plt.subplots()
    value_counts = pd.Series(labels).value_counts()
    value_counts.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"])
    ax.set_ylabel("Count")
    ax.set_title(title)
    for i, v in enumerate(value_counts):
        ax.text(i, v + 100, str(v), ha="center")
    st.pyplot(fig)

# ------------------- STREAMLIT APP -------------------

st.title("🛡 CyberShield — Intrusion Detection")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head(10))

    label_col = "Label" if "Label" in df.columns else None
    dtype = "supervised" if label_col else "unsupervised"
    st.info(f"Detected **{dtype}** dataset {'with label column: ' + label_col if label_col else ''}")

    X, y_raw, df_clean = preprocess_data(df, label_col if dtype == "supervised" else None)
    y = normalize_labels(y_raw) if dtype == "supervised" else None

    # ----------------- Visualization -----------------
    st.subheader("📊 Data Visualization")
    if dtype == "supervised":
        plot_distribution(y, title="Supervised Data: Normal vs Attack")
    else:
        st.write("No label found — running unsupervised mode. PCA or clustering results will be shown.")

    # ----------------- Model Options -----------------
    st.subheader("⚙️ Model Selection")
    if dtype == "supervised":
        model_options = ["RandomForest", "GradientBoosting", "SGDClassifier", "LogisticRegression"]
        selected_models = st.multiselect("Choose supervised models to train", model_options, default=["RandomForest"])
    else:
        model_options = ["IsolationForest", "DBSCAN"]
        selected_models = st.multiselect("Choose unsupervised models to run", model_options, default=["IsolationForest"])

    run_btn = st.button("🚀 Run Models")

    # ----------------- Supervised Training -----------------
    if run_btn and dtype == "supervised":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        results = {}

        for model_name in selected_models:
            if model_name == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == "GradientBoosting":
                model = GradientBoostingClassifier()
            elif model_name == "SGDClassifier":
                model = SGDClassifier(max_iter=1000, tol=1e-3)
            elif model_name == "LogisticRegression":
                model = LogisticRegression(max_iter=1000)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = compute_metrics(y_test, preds)
            results[model_name] = metrics

            st.write(f"### ✅ {model_name} Results")
            st.json(metrics)

    # ----------------- Unsupervised Training -----------------
    if run_btn and dtype == "unsupervised":
        if "IsolationForest" in selected_models:
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(X)
            preds = np.where(preds == 1, "Normal", "Attack")
            plot_distribution(preds, "IsolationForest: Normal vs Attack")

        if "DBSCAN" in selected_models:
            db = DBSCAN(eps=0.5, min_samples=5).fit(X[:5000])  # limit size
            preds = np.where(db.labels_ == -1, "Attack", "Normal")
            plot_distribution(preds, "DBSCAN: Normal vs Attack")
