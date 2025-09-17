# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib

# ---------------- CONFIG ----------------
STRICT_LABEL_NAMES = {"label", "class", "attack", "target", "y"}

# ---------------- HELPERS ----------------
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

def detect_label_column(df: pd.DataFrame):
    for col in df.columns:
        if col.strip().lower() in STRICT_LABEL_NAMES:
            return col
    return None

def preprocess_data(df, label_col=None):
    df = df.replace([np.inf, -np.inf], np.nan)  # fix infinite values

    y = df[label_col] if label_col else None
    X = df.drop(columns=[label_col], errors="ignore")

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    ct = ColumnTransformer(transformers, remainder="drop")
    X_proc = ct.fit_transform(X)

    return X_proc, (y.values if y is not None else None), df

# ---------------- MODELS ----------------
def train_supervised(X, y, selected_models):
    available_models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(),
        "SGDClassifier": SGDClassifier(max_iter=1000, tol=1e-3),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }
    results = {}
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    for name in selected_models:
        clf = available_models[name]
        try:
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            results[name] = {
                "metrics": compute_metrics(y_te, pred),
                "model": clf,
                "test_true": y_te,
                "test_pred": pred
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

def train_unsupervised(X):
    results = {}
    try:
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        iso_labels = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
        results["IsolationForest"] = iso_labels
    except Exception as e:
        results["IsolationForest_error"] = str(e)

    try:
        db = DBSCAN(eps=0.5, min_samples=5).fit(X)
        results["DBSCAN"] = db.labels_  # -1 = anomaly
    except Exception as e:
        results["DBSCAN_error"] = str(e)

    return results

# ---------------- VISUALIZATION ----------------
def plot_distribution(labels, title="Data Distribution"):
    fig, ax = plt.subplots()
    value_counts = pd.Series(labels).value_counts()
    value_counts.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"])
    ax.set_ylabel("Count")
    ax.set_title(title)
    st.pyplot(fig)

def plot_multi_graphs(df, label_col, graphs):
    for g in graphs:
        fig, ax = plt.subplots()
        if g == "Histogram":
            df[label_col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Histogram of Labels")
        elif g == "Pie":
            df[label_col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_title("Pie Chart of Labels")
        elif g == "Line":
            df[label_col].value_counts().plot(kind="line", ax=ax)
            ax.set_title("Line Chart of Labels")
        elif g == "Bar":
            df[label_col].value_counts().plot(kind="barh", ax=ax)
            ax.set_title("Bar Chart of Labels")
        st.pyplot(fig)

# ---------------- STREAMLIT APP ----------------
st.title("🛡 CyberShield — Intrusion Detection")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    label_col = detect_label_column(df)
    if label_col:
        st.success(f"Detected **supervised dataset** with label column: `{label_col}`")
        dtype = "supervised"
    else:
        st.warning("No label column detected → treating as **unsupervised dataset**")
        dtype = "unsupervised"

    # Sidebar only visible after dataset upload
    st.sidebar.header("Options")
    hybrid = st.sidebar.checkbox("Enable Hybrid (multiple models)?")
    multi_graph = st.sidebar.checkbox("Enable multiple graphs?")
    graphs = st.sidebar.multiselect("Choose Graphs", ["Histogram", "Pie", "Line", "Bar"]) if multi_graph else []

    run_btn = st.button("Run Analysis")
    if run_btn:
        X, y, df_clean = preprocess_data(df, label_col if dtype == "supervised" else None)

        if dtype == "supervised":
            # Graph from true labels
            st.subheader("📊 Label Distribution")
            plot_distribution(y, "Supervised Data: Normal vs Attack")

            if graphs:
                st.subheader("📊 Additional Visualizations")
                plot_multi_graphs(df_clean, label_col, graphs)

            # Model selection
            available_models = ["RandomForest", "GradientBoosting", "SGDClassifier", "LogisticRegression"]
            if hybrid:
                chosen_models = st.multiselect("Choose models", available_models, default=available_models[:2])
            else:
                chosen_models = [st.selectbox("Choose one model", available_models)]

            # Train & show
            results = train_supervised(X, y, chosen_models)
            for name, res in results.items():
                if "metrics" in res:
                    st.write(f"### {name} Results")
                    st.json(res["metrics"])
                    pred_df = pd.DataFrame({"True": res["test_true"], "Predicted": res["test_pred"]})
                    st.dataframe(pred_df.head(20))
                else:
                    st.error(f"{name} failed: {res.get('error')}")

        else:
            # Run unsupervised models
            st.subheader("📊 Unsupervised Anomaly Detection")
            results = train_unsupervised(X)

            if "IsolationForest" in results:
                iso_labels = results["IsolationForest"]
                plot_distribution(np.where(iso_labels == -1, "Attack/Anomaly", "Normal"),
                                  "Isolation Forest Results")

            if "DBSCAN" in results:
                db_labels = results["DBSCAN"]
                plot_distribution(np.where(db_labels == -1, "Attack/Anomaly", "Normal"),
                                  "DBSCAN Results")
