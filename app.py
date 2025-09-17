# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Supervised models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Unsupervised models
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# -----------------------------------------
# Helper functions
# -----------------------------------------
def detect_label_column(df):
    """Try to find label column automatically"""
    possible = {"label", "class", "target", "y"}
    for col in df.columns:
        if col.strip().lower() in possible:
            return col
    return None

def preprocess_numeric(df, label_col=None):
    """Basic preprocessing: handle NaN + scaling"""
    if label_col:
        X = df.drop(columns=[label_col])
        y = df[label_col]
    else:
        X = df.copy()
        y = None
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

def plot_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

def plot_column(df, col, graph_type):
    fig, ax = plt.subplots()
    if graph_type == "Histogram":
        df[col].plot(kind="hist", bins=20, ax=ax)
    elif graph_type == "Bar":
        df[col].value_counts().plot(kind="bar", ax=ax)
    elif graph_type == "Pie":
        df[col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    elif graph_type == "Line":
        df[col].reset_index(drop=True).plot(kind="line", ax=ax)
    st.pyplot(fig)

# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.title("🛡 CyberShield — Intrusion Detection")

uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("### Dataset Preview", df.head())

    # Detect label column
    label_col = detect_label_column(df)
    if label_col:
        st.success(f"Detected supervised dataset with label column: **{label_col}**")
    else:
        st.warning("No label column detected → running in UNSUPERVISED mode")

    # Column visualization
    st.subheader("Column Visualization")
    col_choice = st.selectbox("Choose a column to visualize", df.columns)
    graph_choice = st.selectbox("Choose graph type", ["Histogram", "Bar", "Pie", "Line"])

    # Run button
    if st.button("Run"):
        st.write("### Preprocessing data...")
        X, y = preprocess_numeric(df, label_col if label_col else None)
        st.write("Shape after preprocessing:", X.shape)

        # Data type recognition
        st.subheader("Data Type Recognition")
        st.dataframe(pd.DataFrame({
            "Column": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns]
        }))

        # Visualization
        st.subheader("Visualization")
        plot_column(df, col_choice, graph_choice)

        # ---------------- SUPERVISED ----------------
        if label_col:
            st.subheader("Supervised Results")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    metrics = compute_metrics(y_test, pred)
                    st.write(f"#### {name} Metrics", metrics)
                    plot_confusion(y_test, pred, name)
                except Exception as e:
                    st.error(f"{name} failed: {e}")

        # ---------------- UNSUPERVISED ----------------
        else:
            st.subheader("Unsupervised Results")
            try:
                iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
                iso_labels = iso.fit_predict(X)
                st.write("Isolation Forest sample:", iso_labels[:20])
            except Exception as e:
                st.error(f"Isolation Forest failed: {e}")

            try:
                db = DBSCAN(eps=0.5, min_samples=5).fit(X)
                st.write("DBSCAN cluster labels (sample):", db.labels_[:20])
            except Exception as e:
                st.error(f"DBSCAN failed: {e}")
