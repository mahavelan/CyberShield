import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TensorFlow logs

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# Try importing TensorFlow (for deep learning models)
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input, Dropout
    from tensorflow.keras.optimizers import Adam
except Exception:
    USE_TF = False

# Streamlit page setup
st.set_page_config(page_title="CyberShield", layout="wide")
st.title("🛡️ CyberShield Web App")
st.write("Upload your dataset. The app will auto-detect if it is **Supervised** or **Unsupervised**, preprocess it, and run ML/DL models.")

# --------------------
# Helper functions
# --------------------
def find_label_column(df):
    """Try to auto-detect label column from common names."""
    label_keywords = ["label", "class", "target", "y", "attack", "category", "outcome"]
    for col in df.columns:
        if any(kw in col.lower() for kw in label_keywords):
            return col
    return None

def preprocess_data(df, label_col=None):
    """Preprocess dataset automatically: handle NaNs, encode categoricals, scale features."""
    df = df.copy()
    df = df.dropna(axis=1, how="all")  # drop fully empty columns
    df = df.fillna(0)  # fill remaining NaNs
    
    if label_col:
        X = df.drop(columns=[label_col])
        y = df[label_col]
    else:
        X = df
        y = None

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return pd.DataFrame(X_scaled, columns=X.columns), y

def plot_visualization(graph_type, y_true=None, y_pred=None, labels=None):
    """Generate selected visualizations."""
    fig, ax = plt.subplots()

    if graph_type == "Confusion Matrix" and y_true is not None and y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")

    elif graph_type == "Bar Chart" and labels is not None:
        vc = pd.Series(labels).value_counts()
        vc.plot(kind="bar", ax=ax, color=["red", "green"])
        ax.set_ylabel("Count")
        ax.set_title("Bar Chart of Predictions")

    elif graph_type == "Pie Chart" and labels is not None:
        vc = pd.Series(labels).value_counts()
        ax.pie(vc, labels=vc.index, autopct="%1.1f%%", colors=["red", "green"])
        ax.set_title("Pie Chart of Predictions")

    elif graph_type == "Line Chart" and labels is not None:
        pd.Series(labels).reset_index(drop=True).plot(kind="line", ax=ax)
        ax.set_ylabel("Predictions")
        ax.set_title("Line Chart of Predictions")

    elif graph_type == "Histogram" and labels is not None:
        pd.Series(labels).plot(kind="hist", bins=10, ax=ax, color="blue")
        ax.set_title("Histogram of Predictions")

    elif graph_type == "Box Plot" and labels is not None:
        pd.Series(labels).plot(kind="box", ax=ax)
        ax.set_title("Box Plot of Predictions")

    st.pyplot(fig)

# --------------------
# File Upload
# --------------------
uploaded = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx"])

if uploaded:
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Auto-detect label column
    label_col = find_label_column(df)
    if label_col:
        st.success(f"Auto-detected label column: **{label_col}** → Supervised learning")
        dataset_type = "Supervised"
    else:
        st.warning("No label column detected → Unsupervised learning")
        dataset_type = "Unsupervised"

    # --------------------
    # Sidebar controls
    # --------------------
    st.sidebar.header("⚙️ Options")

    # Model selection logic
    hybrid_mode = st.sidebar.checkbox("Enable Hybrid (Multiple Models)?", value=False)

    if dataset_type == "Supervised":
        supervised_models = [
            "Logistic Regression", "Random Forest", "Decision Tree", "KNN", "SVM",
            "Naive Bayes", "Gradient Boosting", "AdaBoost"
        ]
        if USE_TF:
            supervised_models += ["CNN", "LSTM"]

        if hybrid_mode:
            chosen_models = st.sidebar.multiselect("Choose Supervised Models", supervised_models)
        else:
            chosen_models = st.sidebar.selectbox("Choose One Supervised Model", supervised_models)

    else:
        unsupervised_models = ["KMeans", "DBSCAN", "Isolation Forest", "One-Class SVM"]
        if USE_TF:
            unsupervised_models += ["Autoencoder"]

        if hybrid_mode:
            chosen_models = st.sidebar.multiselect("Choose Unsupervised Models", unsupervised_models)
        else:
            chosen_models = st.sidebar.selectbox("Choose One Unsupervised Model", unsupervised_models)

    # Visualization selection
    multi_graph = st.sidebar.checkbox("Enable Multiple Graphs?", value=False)
    graph_options = ["Confusion Matrix", "Bar Chart", "Pie Chart", "Line Chart", "Histogram", "Box Plot"]

    if multi_graph:
        chosen_graphs = st.sidebar.multiselect("Choose Visualizations", graph_options)
    else:
        chosen_graphs = [st.sidebar.selectbox("Choose One Visualization", graph_options)]

    run_btn = st.sidebar.button("🚀 Run Models")

    # --------------------
    # Run Logic
    # --------------------
    if run_btn and chosen_models:
        X_proc, y = preprocess_data(df, label_col)

        # --------------------
        # Supervised
        # --------------------
        if dataset_type == "Supervised":
            X_train, X_test, y_train, y_test = train_test_split(
                X_proc, y, test_size=0.3, random_state=42, stratify=y
            )

            for model_name in chosen_models:
                st.subheader(f"Model: {model_name}")

                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(n_estimators=100)
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif model_name == "KNN":
                    model = KNeighborsClassifier()
                elif model_name == "SVM":
                    model = SVC()
                elif model_name == "Naive Bayes":
                    model = GaussianNB()
                elif model_name == "Gradient Boosting":
                    model = GradientBoostingClassifier()
                elif model_name == "AdaBoost":
                    model = AdaBoostClassifier()
                elif USE_TF and model_name == "CNN":
                    model = Sequential([
                        Input(shape=(X_train.shape[1], 1)),
                        Conv1D(32, 3, activation="relu"),
                        MaxPooling1D(2),
                        Flatten(),
                        Dense(64, activation="relu"),
                        Dense(1, activation="sigmoid")
                    ])
                    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
                    model.fit(np.expand_dims(X_train, -1), y_train, epochs=5, batch_size=32, verbose=0)
                    y_pred = (model.predict(np.expand_dims(X_test, -1)) > 0.5).astype(int).flatten()
                elif USE_TF and model_name == "LSTM":
                    model = Sequential([
                        Input(shape=(X_train.shape[1], 1)),
                        LSTM(32),
                        Dense(1, activation="sigmoid")
                    ])
                    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
                    model.fit(np.expand_dims(X_train, -1), y_train, epochs=5, batch_size=32, verbose=0)
                    y_pred = (model.predict(np.expand_dims(X_test, -1)) > 0.5).astype(int).flatten()
                else:
                    continue

                if model_name not in ["CNN", "LSTM"]:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                rmse = mean_squared_error(y_test, y_pred, squared=False)

                st.write({"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "RMSE": rmse})

                # Visualizations
                for g in chosen_graphs:
                    plot_visualization(g, y_test, y_pred, labels=y_pred)

        # --------------------
        # Unsupervised
        # --------------------
        else:
            X_vals = X_proc.values
            for model_name in chosen_models:
                st.subheader(f"Model: {model_name}")

                if model_name == "KMeans":
                    model = KMeans(n_clusters=2, random_state=42)
                    labels = model.fit_predict(X_vals)
                elif model_name == "DBSCAN":
                    model = DBSCAN(eps=0.5, min_samples=5)
                    labels = model.fit_predict(X_vals)
                elif model_name == "Isolation Forest":
                    model = IsolationForest(contamination=0.1, random_state=42)
                    labels = model.fit_predict(X_vals)
                elif model_name == "One-Class SVM":
                    model = OneClassSVM(nu=0.1, kernel="rbf")
                    labels = model.fit_predict(X_vals)
                elif USE_TF and model_name == "Autoencoder":
                    model = Sequential([
                        Dense(32, activation="relu", input_shape=(X_vals.shape[1],)),
                        Dense(16, activation="relu"),
                        Dense(32, activation="relu"),
                        Dense(X_vals.shape[1], activation="linear")
                    ])
                    model.compile(optimizer="adam", loss="mse")
                    model.fit(X_vals, X_vals, epochs=5, batch_size=32, verbose=0)
                    recon = model.predict(X_vals)
                    err = np.mean(np.square(recon - X_vals), axis=1)
                    thr = np.percentile(err, 90)
                    labels = np.where(err > thr, -1, 1)
                else:
                    continue

                # Normalize labels to Attack/Normal
                labels = np.where(labels == -1, "Attack", "Normal")

                st.write(pd.Series(labels).value_counts())

                # Visualizations
                for g in chosen_graphs:
                    plot_visualization(g, labels=labels)

else:
    st.info("👆 Upload a dataset to get started.")
