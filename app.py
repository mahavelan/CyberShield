# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix, mean_absolute_error
)

# Classical models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest

# DL (lazy import later)
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Flatten, Dropout

st.set_page_config(page_title="CyberShield", layout="wide")
st.title("🛡 CyberShield — Intrusion Detection")

# ---------------- Utility ----------------
def find_label(df):
    for col in df.columns:
        if str(col).lower() in ["label", "class", "target", "y", "attack"]:
            return col
    return None

def preprocess_numeric(df, label_col=None):
    if label_col:
        X = df.drop(columns=[label_col])
        y = df[label_col]
    else:
        X = df.copy()
        y = None
    X = X.select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def compute_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R²": r2_score(y_true, y_pred)
    }

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

def plot_attack_distribution(labels, title="Attack vs Normal"):
    fig, ax = plt.subplots()
    pd.Series(labels).value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax)
    ax.set_ylabel("")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

# ---------------- UI ----------------
uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
st.write("### Dataset Preview", df.head())

label_col = find_label(df)
dtype = "supervised" if label_col else "unsupervised"
st.info(f"Detected **{dtype.upper()}** dataset{' with label: ' + label_col if label_col else ''}")

# Model choices
sup_models = [
    "Logistic Regression", "Random Forest", "Gradient Boosting",
    "SVM", "Naive Bayes", "Keras-MLP", "Keras-1D-CNN"
]
unsup_models = ["Isolation Forest", "DBSCAN", "KMeans", "Autoencoder"]

hybrid = st.sidebar.checkbox("Enable Hybrid (choose multiple models)")
if dtype == "supervised":
    models = sup_models
else:
    models = unsup_models

if hybrid:
    chosen_models = st.sidebar.multiselect("Choose models", models)
else:
    chosen_models = [st.sidebar.selectbox("Choose model", models)]

graphs = st.sidebar.multiselect("Choose visualizations", ["Confusion Matrix", "Attack Distribution", "Histogram", "Bar", "Line", "Pie"])
run_btn = st.sidebar.button("Run")

# ---------------- Run ----------------
if run_btn:
    try:
        if dtype == "supervised":
            X, y = preprocess_numeric(df, label_col)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            for m in chosen_models:
                st.subheader(f"### {m} Results")
                try:
                    if m == "Logistic Regression":
                        clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                        pred = clf.predict(X_test)
                    elif m == "Random Forest":
                        clf = RandomForestClassifier().fit(X_train, y_train)
                        pred = clf.predict(X_test)
                    elif m == "Gradient Boosting":
                        clf = GradientBoostingClassifier().fit(X_train, y_train)
                        pred = clf.predict(X_test)
                    elif m == "SVM":
                        clf = SVC().fit(X_train, y_train)
                        pred = clf.predict(X_test)
                    elif m == "Naive Bayes":
                        clf = GaussianNB().fit(X_train, y_train)
                        pred = clf.predict(X_test)
                    elif m == "Keras-MLP":
                        model = Sequential([
                            Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
                            Dense(32, activation="relu"),
                            Dense(1, activation="sigmoid")
                        ])
                        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                        model.fit(X_train, y_train, epochs=3, verbose=0)
                        pred = (model.predict(X_test) > 0.5).astype(int).ravel()
                    elif m == "Keras-1D-CNN":
                        X_tr = np.expand_dims(X_train, -1)
                        X_te = np.expand_dims(X_test, -1)
                        model = Sequential([
                            Conv1D(32, 3, activation="relu", input_shape=(X_tr.shape[1],1)),
                            Flatten(),
                            Dense(32, activation="relu"),
                            Dense(1, activation="sigmoid")
                        ])
                        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                        model.fit(X_tr, y_train, epochs=3, verbose=0)
                        pred = (model.predict(X_te) > 0.5).astype(int).ravel()
                    else:
                        st.warning(f"{m} not supported")
                        continue

                    metrics = compute_metrics(y_test, pred)
                    st.json(metrics)

                    if "Confusion Matrix" in graphs:
                        plot_confusion(y_test, pred)
                    if "Attack Distribution" in graphs:
                        plot_attack_distribution(y_test, "True Attack Distribution")
                except Exception as e:
                    st.error(f"{m} failed: {e}")
                    st.text(traceback.format_exc())

        else:  # unsupervised
            X, _ = preprocess_numeric(df)
            for m in chosen_models:
                st.subheader(f"### {m} Results")
                try:
                    if m == "Isolation Forest":
                        clf = IsolationForest(contamination=0.1).fit(X)
                        pred = clf.predict(X)
                        pred = np.where(pred==-1, "Attack", "Normal")
                    elif m == "DBSCAN":
                        pred = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)
                        pred = np.where(pred==-1, "Attack", "Normal")
                    elif m == "KMeans":
                        pred = KMeans(n_clusters=2, random_state=42).fit_predict(X)
                        pred = np.where(pred==0, "Normal", "Attack")
                    elif m == "Autoencoder":
                        inp = Input(shape=(X.shape[1],))
                        encoded = Dense(32, activation="relu")(inp)
                        decoded = Dense(X.shape[1], activation="linear")(encoded)
                        auto = Model(inp, decoded)
                        auto.compile(optimizer="adam", loss="mse")
                        auto.fit(X, X, epochs=3, verbose=0)
                        recon = auto.predict(X)
                        mse = np.mean(np.power(X - recon, 2), axis=1)
                        thresh = np.percentile(mse, 95)
                        pred = np.where(mse > thresh, "Attack", "Normal")
                    else:
                        continue

                    counts = pd.Series(pred).value_counts().to_dict()
                    st.write("Prediction counts:", counts)

                    if "Attack Distribution" in graphs:
                        plot_attack_distribution(pred, "Predicted Attack vs Normal")
                except Exception as e:
                    st.error(f"{m} failed: {e}")
                    st.text(traceback.format_exc())

    except Exception as e:
        st.error("Run failed")
        st.text(traceback.format_exc())
