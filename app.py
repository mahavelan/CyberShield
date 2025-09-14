# app.py
"""
CyberShield — Robust Streamlit app (single-file)
- Strict label detection with user override
- Auto-route to Supervised / Unsupervised
- Single vs Hybrid models (controlled by checkbox)
- Single vs Multi visualizations (controlled by checkbox)
- Classical ML + Deep (optional) models (MLP, CNN1D, CNN2D, LSTM, BiLSTM, Attention, Softmax)
- Robust preprocessing & error handling
"""

import os, re, traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error

# classical classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Optional TF/Keras models
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, Model
    from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D, Conv2D,
                                         MaxPooling1D, MaxPooling2D, Flatten,
                                         LSTM, Bidirectional, Attention, Softmax, Reshape)
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel("ERROR")
except Exception:
    USE_TF = False

# show detailed errors inside app (better than generic "Error running app")
st.set_option("client.showErrorDetails", True)

# ---------------- Utility functions ----------------
STRICT_LABEL_NAMES = {"label", "class", "target", "attack", "y", "output", "result", "category", "type", "true_label"}

def normalize_name(colname: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(colname)).strip().lower()

def find_label_column_strict(df: pd.DataFrame):
    """Return the column (original name) if an exact normalized name is in STRICT_LABEL_NAMES."""
    for col in df.columns:
        if normalize_name(col) in STRICT_LABEL_NAMES:
            return col
    return None

def safe_read_file(uploaded):
    """Read CSV or Excel robustly."""
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded)

def preprocess_features(df: pd.DataFrame, drop_cols=None):
    """
    Preprocess:
    - drop specified columns and fully empty columns
    - fill numeric NaNs with median, categorical with 'missing'
    - one-hot encode categoricals (drop_first=True)
    - drop constant columns
    - standard scale numeric features
    Returns (X_df, pipeline_info)
    """
    dfc = df.copy()
    if drop_cols:
        dfc = dfc.drop(columns=[c for c in drop_cols if c in dfc.columns], errors='ignore')
    # drop fully empty columns
    dfc = dfc.dropna(axis=1, how='all')
    # fill missing values
    for c in dfc.columns:
        if pd.api.types.is_numeric_dtype(dfc[c]):
            # if column entirely NaN -> fill 0, else median
            if dfc[c].dropna().shape[0] > 0:
                dfc[c] = dfc[c].fillna(dfc[c].median())
            else:
                dfc[c] = dfc[c].fillna(0)
        else:
            dfc[c] = dfc[c].fillna("missing")
    # one-hot encode categoricals
    cat_cols = dfc.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        dfc = pd.get_dummies(dfc, columns=cat_cols, drop_first=True)
    # drop constant columns
    nunq = dfc.nunique(dropna=True)
    const_cols = nunq[nunq <= 1].index.tolist()
    if const_cols:
        dfc = dfc.drop(columns=const_cols, errors='ignore')
    if dfc.shape[1] == 0:
        raise ValueError("No usable features after preprocessing.")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dfc.values)
    X_df = pd.DataFrame(X_scaled, columns=dfc.columns, index=dfc.index)
    pipeline = {"scaler": scaler, "feature_columns": dfc.columns.tolist()}
    return X_df, pipeline

# Basic metric computation
def compute_supervised_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "rmse": mean_squared_error(y_true, y_pred, squared=False)
    }

# plotting helpers
def plot_confusion_mat(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    if labels is not None:
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels, rotation=0)
    st.pyplot(fig)
    plt.close(fig)

def plot_pie_counts(series, title=""):
    fig, ax = plt.subplots()
    vc = series.value_counts()
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def plot_generic(df, kind, x=None, y=None, hue=None, title=None):
    fig, ax = plt.subplots(figsize=(6,4))
    try:
        if kind == "Bar":
            if x and y:
                sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
            else:
                df.select_dtypes(include=np.number).sum().plot(kind='bar', ax=ax)
        elif kind == "Pie":
            col = x if x else df.columns[0]
            s = df[col].value_counts()
            ax.pie(s.values, labels=s.index, autopct="%1.1f%%")
        elif kind == "Heatmap":
            sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm', ax=ax)
        elif kind == "Line":
            if x and y:
                sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
            else:
                df.reset_index(drop=True).plot(ax=ax)
        elif kind == "Histogram":
            if y:
                sns.histplot(data=df, x=y, hue=hue, kde=True, ax=ax)
            else:
                df.select_dtypes(include=np.number).hist(ax=ax)
                st.pyplot(fig); plt.close(fig); return
        elif kind == "Box":
            if y:
                sns.boxplot(data=df, y=y, ax=ax)
            else:
                sns.boxplot(data=df.select_dtypes(include=np.number), ax=ax)
        elif kind == "Scatter":
            if x and y:
                sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
        ax.set_title(title or kind)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not render {kind}: {e}")
    finally:
        plt.close(fig)

# ---------------- Deep model builders ----------------
def build_keras_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'), Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_keras_cnn1d(input_len):
    model = Sequential([
        Input(shape=(input_len,1)),
        Conv1D(32,3,activation='relu'), MaxPooling1D(2),
        Conv1D(64,3,activation='relu'), MaxPooling1D(2),
        Flatten(), Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_keras_cnn2d(shape):
    model = Sequential([
        Input(shape=shape),
        Conv2D(32,(3,3),activation='relu'), MaxPooling2D((2,2)),
        Conv2D(64,(3,3),activation='relu'), MaxPooling2D((2,2)),
        Flatten(), Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_keras_lstm(input_len):
    model = Sequential([
        Input(shape=(input_len,1)),
        LSTM(64), Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_keras_bilstm(input_len):
    model = Sequential([
        Input(shape=(input_len,1)),
        Bidirectional(LSTM(64)), Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_attention_mlp(input_dim):
    # simple demonstration attention block used on dense features (not ideal but ok for demo)
    inp = Input(shape=(input_dim,))
    d = Dense(64, activation='relu')(inp)
    # Attention expects sequence-like; we expand dims to (batch, seq_len=1, features)
    x = tf.expand_dims(d, axis=1)  # shape (batch,1,features)
    att = Attention()([x,x])
    flat = Flatten()(att)
    out = Dense(1, activation='sigmoid')(flat)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_softmax_classifier(input_dim, n_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------- UI & Flow ----------------
st.set_page_config(page_title="CyberShield", layout="wide")
st.title("🛡 CyberShield — Intrusion Detection (Robust)")

uploaded = st.file_uploader("Upload dataset (CSV or XLSX)", type=["csv","xlsx"])
if not uploaded:
    st.info("Upload a dataset to begin. The app will attempt strict label detection (e.g. column name 'label'/'class'). You can override the detected label below.")
    st.stop()

# Read file
try:
    df = safe_read_file(uploaded)
except Exception as e:
    st.error("Failed to read uploaded file.")
    st.code(traceback.format_exc())
    st.stop()

st.subheader("Dataset preview (first 10 rows)")
st.dataframe(df.head(10))

# Strict label detection (normalized)
detected_label = find_label_column_strict(df)
st.write("Detected columns:", list(df.columns))
if detected_label:
    st.success(f"Auto-detected label column (strict match): **{detected_label}**")
else:
    st.info("No strict label column found.")

# Let user override / choose label or choose None (unsupervised)
options = ["-- None / Unlabeled --"] + list(df.columns)
default_idx = options.index(detected_label) if (detected_label and detected_label in options) else 0
label_choice = st.selectbox("Select label column (or choose '-- None / Unlabeled --')", options, index=default_idx)
if label_choice == "-- None / Unlabeled --":
    dataset_type = "Unsupervised"
    label_col = None
    st.warning("Proceeding as Unsupervised dataset (no label chosen).")
else:
    dataset_type = "Supervised"
    label_col = label_choice
    st.success(f"Using **{label_col}** as label (Supervised flow).")

# Sidebar: Model & Visualization Options
st.sidebar.header("Model & Visualization Options")

# Supervised model lists (classical + deep)
SUP_CLASSICAL = ["Logistic Regression", "Random Forest", "Decision Tree", "KNN", "SVM (RBF)", "Naive Bayes", "Gradient Boosting", "AdaBoost", "MLP (sklearn)"]
SUP_DEEP = []
if USE_TF:
    SUP_DEEP = ["Keras-MLP", "Keras-CNN1D", "Keras-CNN2D", "Keras-LSTM", "Keras-BiLSTM", "Keras-Attention", "Keras-Softmax"]

# Unsupervised list
UNSUP_CLASSICAL = ["Isolation Forest", "One-Class SVM", "KMeans", "DBSCAN", "PCA-heuristic"]
UNSUP_DEEP = ["Autoencoder (Keras)"] if USE_TF else []

# Visualization options
GRAPH_OPTIONS = ["Confusion Matrix", "Bar", "Pie", "Line", "Histogram", "Box", "Heatmap"]

# Side selection logic with enforcement of hybrid/multi flags
if dataset_type == "Supervised":
    st.sidebar.subheader("Supervised model selection")
    sup_hybrid = st.sidebar.checkbox("Enable Hybrid Ensemble (choose multiple models)?", value=False, key="sup_hybrid")
    if sup_hybrid:
        chosen_models = st.sidebar.multiselect("Choose 2 or more models for Hybrid", SUP_CLASSICAL + SUP_DEEP)
    else:
        chosen_one = st.sidebar.selectbox("Choose ONE model", SUP_CLASSICAL + SUP_DEEP)
        chosen_models = [chosen_one]
    train_deep = st.sidebar.checkbox("Train deep models (may be slow)", value=False)
else:
    st.sidebar.subheader("Unsupervised model selection")
    unsup_hybrid = st.sidebar.checkbox("Enable Hybrid Consensus (choose multiple models)?", value=False, key="unsup_hybrid")
    if unsup_hybrid:
        chosen_models = st.sidebar.multiselect("Choose 2 or more unsupervised models", UNSUP_CLASSICAL + UNSUP_DEEP)
    else:
        chosen_one = st.sidebar.selectbox("Choose ONE unsupervised model", UNSUP_CLASSICAL + UNSUP_DEEP)
        chosen_models = [chosen_one]
    contamination = st.sidebar.slider("Estimated contamination (anomaly fraction)", 0.001, 0.5, 0.05, 0.001)

# Graph selections
st.sidebar.subheader("Visualizations")
graph_multi = st.sidebar.checkbox("Enable multiple visualizations?", value=False, key="multi_graphs")
if graph_multi:
    chosen_graphs = st.sidebar.multiselect("Choose visualizations (2+)", GRAPH_OPTIONS, default=["Bar", "Pie"])
else:
    gg = st.sidebar.selectbox("Choose one visualization", GRAPH_OPTIONS)
    chosen_graphs = [gg]

run = st.sidebar.button("Run Selected Models")

# Validation of selections
def validate_selections():
    if dataset_type == "Supervised":
        if sup_hybrid:
            if len(chosen_models) < 2:
                st.sidebar.error("Hybrid ON → select at least 2 models.")
                return False
        else:
            if len(chosen_models) != 1:
                st.sidebar.error("Hybrid OFF → select exactly 1 model.")
                return False
    else:
        if unsup_hybrid:
            if len(chosen_models) < 2:
                st.sidebar.error("Hybrid ON → select at least 2 models.")
                return False
        else:
            if len(chosen_models) != 1:
                st.sidebar.error("Hybrid OFF → select exactly 1 model.")
                return False
    if graph_multi:
        if len(chosen_graphs) < 2:
            st.sidebar.error("Multi-graphs ON → select 2 or more visualizations.")
            return False
    else:
        if len(chosen_graphs) != 1:
            st.sidebar.error("Multi-graphs OFF → select exactly 1 visualization.")
            return False
    return True

# Helper to show exceptions inside the app
def show_exception(e):
    st.error("An error occurred:")
    st.code(traceback.format_exception(type(e), e, e.__traceback__))

# ---------------- Run models ----------------
if run:
    if not validate_selections():
        st.stop()

    try:
        if dataset_type == "Supervised":
            # prepare features and labels
            if label_col not in df.columns:
                st.error("Label column not found in dataset — please re-upload or pick the correct column.")
                st.stop()
            X_raw = df.drop(columns=[label_col]).copy()
            y_raw = df[label_col].copy()

            # robust label encoding: always factorize as strings (works for numeric and text labels)
            try:
                y_codes, uniques = pd.factorize(y_raw.astype(str))
            except Exception as ex:
                show_exception(ex)
                st.stop()
            label_mapping = {i: v for i, v in enumerate(uniques)}
            st.write("Label classes detected:", list(uniques))

            # preprocess features (safe)
            try:
                X_proc, pipeline = preprocess_features(X_raw, drop_cols=None)
            except Exception as ex:
                show_exception(ex)
                st.stop()

            # train-test split, stratify when possible
            stratify_param = y_codes if len(np.unique(y_codes)) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(X_proc.values, y_codes, test_size=0.3, random_state=42, stratify=stratify_param)
            st.success("Preprocessing and train/test split complete.")

            preds_store = {}   # model_name -> predictions on X_test
            metrics_summary = {}  # model_name -> metrics dict

            # helper to train classical models
            def run_classical(name, clf):
                try:
                    clf.fit(X_train, y_train)
                    yp = clf.predict(X_test)
                    preds_store[name] = yp
                    metrics_summary[name] = compute_supervised_metrics(y_test, yp)
                except Exception as ex:
                    st.warning(f"{name} failed: {ex}")

            # iterate chosen models
            for m in chosen_models:
                st.info(f"Running {m} ...")
                try:
                    if m == "Logistic Regression":
                        run_classical(m, LogisticRegression(max_iter=1000))
                    elif m == "Random Forest":
                        run_classical(m, RandomForestClassifier(n_estimators=100, random_state=42))
                    elif m == "Decision Tree":
                        run_classical(m, DecisionTreeClassifier(random_state=42))
                    elif m == "KNN":
                        run_classical(m, KNeighborsClassifier())
                    elif m == "SVM (RBF)":
                        run_classical(m, SVC(kernel='rbf', probability=True))
                    elif m == "Naive Bayes":
                        run_classical(m, GaussianNB())
                    elif m == "Gradient Boosting":
                        run_classical(m, GradientBoostingClassifier(random_state=42))
                    elif m == "AdaBoost":
                        run_classical(m, AdaBoostClassifier(random_state=42))
                    elif m == "MLP (sklearn)":
                        run_classical(m, MLPClassifier(max_iter=500))
                    # Deep models (if TF available)
                    elif USE_TF and m == "Keras-MLP":
                        model = build_keras_mlp(X_train.shape[1])
                        epochs = 10 if train_deep else 4
                        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
                        yp = (model.predict(X_test).ravel() >= 0.5).astype(int)
                        preds_store[m] = yp
                        metrics_summary[m] = compute_supervised_metrics(y_test, yp)
                    elif USE_TF and m == "Keras-CNN1D":
                        model = build_keras_cnn1d(X_train.shape[1])
                        Xtr = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        Xte = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                        epochs = 8 if train_deep else 4
                        model.fit(Xtr, y_train, epochs=epochs, batch_size=32, verbose=0)
                        yp = (model.predict(Xte).ravel() >= 0.5).astype(int)
                        preds_store[m] = yp
                        metrics_summary[m] = compute_supervised_metrics(y_test, yp)
                    elif USE_TF and m == "Keras-CNN2D":
                        # attempt square reshape; if not possible, warn and skip
                        n_feats = X_train.shape[1]
                        size = int(np.sqrt(n_feats))
                        if size * size != n_feats:
                            st.warning(f"Cannot reshape features ({n_feats}) into square for CNN2D. Skipping {m}.")
                        else:
                            Xtr = X_train.reshape((-1, size, size, 1))
                            Xte = X_test.reshape((-1, size, size, 1))
                            model = build_keras_cnn2d((size, size, 1))
                            epochs = 8 if train_deep else 4
                            model.fit(Xtr, y_train, epochs=epochs, batch_size=32, verbose=0)
                            yp = (model.predict(Xte).ravel() >= 0.5).astype(int)
                            preds_store[m] = yp
                            metrics_summary[m] = compute_supervised_metrics(y_test, yp)
                    elif USE_TF and m == "Keras-LSTM":
                        model = build_keras_lstm(X_train.shape[1])
                        Xtr = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        Xte = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                        epochs = 6 if train_deep else 3
                        model.fit(Xtr, y_train, epochs=epochs, batch_size=32, verbose=0)
                        yp = (model.predict(Xte).ravel() >= 0.5).astype(int)
                        preds_store[m] = yp
                        metrics_summary[m] = compute_supervised_metrics(y_test, yp)
                    elif USE_TF and m == "Keras-BiLSTM":
                        model = build_keras_bilstm(X_train.shape[1])
                        Xtr = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        Xte = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                        epochs = 6 if train_deep else 3
                        model.fit(Xtr, y_train, epochs=epochs, batch_size=32, verbose=0)
                        yp = (model.predict(Xte).ravel() >= 0.5).astype(int)
                        preds_store[m] = yp
                        metrics_summary[m] = compute_supervised_metrics(y_test, yp)
                    elif USE_TF and m == "Keras-Attention":
                        model = build_attention_mlp(X_train.shape[1])
                        epochs = 6 if train_deep else 3
                        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
                        yp = (model.predict(X_test).ravel() >= 0.5).astype(int)
                        preds_store[m] = yp
                        metrics_summary[m] = compute_supervised_metrics(y_test, yp)
                    elif USE_TF and m == "Keras-Softmax":
                        # multi-class softmax classifier
                        model = build_softmax_classifier(X_train.shape[1], len(uniques))
                        epochs = 8 if train_deep else 4
                        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
                        yp = np.argmax(model.predict(X_test), axis=1)
                        preds_store[m] = yp
                        metrics_summary[m] = compute_supervised_metrics(y_test, yp)
                    else:
                        st.warning(f"Unknown or unavailable model: {m}")
                except Exception as ex:
                    st.warning(f"Model {m} raised an exception: {ex}")
                    st.code(traceback.format_exc())

            # Hybrid majority vote (supervised)
            if sup_hybrid:
                # collect predictions arrays
                arrays = []
                for k, v in preds_store.items():
                    arrays.append(np.asarray(v))
                if len(arrays) >= 2:
                    # majority (mode) across columns
                    arr = np.vstack(arrays).T
                    from scipy.stats import mode
                    mv = mode(arr, axis=1).mode.ravel()
                    preds_store["Hybrid-Majority"] = mv
                    metrics_summary["Hybrid-Majority"] = compute_supervised_metrics(y_test, mv)
                else:
                    st.warning("Hybrid enabled but fewer than 2 models succeeded; hybrid skipped.")

            # Display supervised metrics and visualizations
            if metrics_summary:
                st.header("Supervised — Metrics Summary")
                df_metrics = pd.DataFrame(metrics_summary).T
                st.dataframe(df_metrics.round(4))

                # per-model outputs (confusion matrix + chosen graphs + download)
                for name, yp in preds_store.items():
                    st.subheader(f"Model: {name}")
                    # confusion matrix (if binary or small classes)
                    try:
                        if "Confusion Matrix" in chosen_graphs:
                            plot_confusion_mat(y_test, yp, labels=[label_mapping[i] for i in range(len(uniques))] if len(uniques) <= 20 else None)
                    except Exception:
                        st.write("Could not plot confusion matrix.")
                    # other charts
                    X_test_df = pd.DataFrame(X_test, columns=pipeline["feature_columns"])
                    for g in chosen_graphs:
                        if g == "Confusion Matrix":
                            continue
                        if g in ("Pie", "Bar"):
                            s = pd.Series([label_mapping.get(int(x), str(x)) if str(x).isdigit() else str(x) for x in yp])
                            plot_pie_counts(s, title=f"{name} — {g}")
                        else:
                            plot_generic(X_test_df, g, title=f"{name} — {g}")
                    # prepare and download CSV
                    out = X_test_df.copy()
                    out["Actual"] = [label_mapping.get(int(v), str(v)) if isinstance(v, (np.integer, int)) else str(v) for v in y_test]
                    out["Predicted"] = [label_mapping.get(int(v), str(v)) if isinstance(v, (np.integer, int)) and int(v) in label_mapping else str(v) for v in yp]
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(f"Download results - {name}", data=csv, file_name=f"results_{name}.csv", mime="text/csv")
            else:
                st.warning("No metrics were produced — check model logs above.")
        else:
            # ---------------- Unsupervised flow ----------------
            st.header("Unsupervised flow — preprocessing")
            try:
                X_proc, pipeline = preprocess_features(df, drop_cols=None)
            except Exception as ex:
                show_exception(ex); st.stop()
            X_vals = X_proc.values
            st.success("Preprocessing complete.")

            unsup_results = {}
            percent_flagged = {}

            for m in chosen_models:
                st.info(f"Running {m} ...")
                try:
                    if m == "Isolation Forest":
                        iso = IsolationForest(contamination=contamination, random_state=42)
                        pred = iso.fit_predict(X_vals)  # -1 anomaly
                        labels = np.where(pred == -1, "Attack", "Normal")
                    elif m == "One-Class SVM":
                        oc = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
                        pred = oc.fit_predict(X_vals)
                        labels = np.where(pred == -1, "Attack", "Normal")
                    elif m == "KMeans":
                        km = KMeans(n_clusters=2, random_state=42)
                        cl = km.fit_predict(X_vals)
                        counts = pd.Series(cl).value_counts()
                        small = counts.idxmin()
                        labels = np.where(cl == small, "Attack", "Normal")
                    elif m == "DBSCAN":
                        db = DBSCAN(eps=0.5, min_samples=5)
                        cl = db.fit_predict(X_vals)
                        labels = np.where(cl == -1, "Attack", "Normal")
                    elif m == "PCA-heuristic":
                        pca = PCA(n_components=min(5, X_vals.shape[1]))
                        comp = pca.fit_transform(X_vals)
                        dist = np.linalg.norm(comp - comp.mean(axis=0), axis=1)
                        thr = np.percentile(dist, 100 * (1 - contamination))
                        labels = np.where(dist >= thr, "Attack", "Normal")
                    elif USE_TF and m == "Autoencoder (Keras)":
                        # small autoencoder
                        ae = Sequential([
                            Input(shape=(X_vals.shape[1],)),
                            Dense(max(16, X_vals.shape[1]//2), activation='relu'),
                            Dense(max(8, X_vals.shape[1]//4), activation='relu'),
                            Dense(max(16, X_vals.shape[1]//2), activation='relu'),
                            Dense(X_vals.shape[1], activation='linear'),
                        ])
                        ae.compile(optimizer=Adam(1e-3), loss='mse')
                        epochs = 10 if train_deep else 5
                        ae.fit(X_vals, X_vals, epochs=epochs, batch_size=128, verbose=0)
                        recon = ae.predict(X_vals)
                        rec_err = np.mean(np.square(recon - X_vals), axis=1)
                        thr = np.percentile(rec_err, 100 * (1 - contamination))
                        labels = np.where(rec_err >= thr, "Attack", "Normal")
                    else:
                        st.warning(f"Unrecognized unsupervised model: {m}")
                        continue

                    unsup_results[m] = labels
                    percent_flagged[m] = (labels == "Attack").sum() / len(labels)

                    s = pd.Series(labels)
                    st.write(f"{m} — Attack: {(s=='Attack').sum()} | Normal: {(s=='Normal').sum()}  ({(s=='Attack').mean():.2%} flagged)")
                    # visualizations
                    for g in chosen_graphs:
                        if g in ("Pie", "Bar"):
                            plot_pie_counts(s, title=f"{m} — {g}")
                        else:
                            plot_generic(X_proc, g, title=f"{m} — {g}")
                    # download
                    out = df.copy()
                    out[f"Pred_{m}"] = labels
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(f"Download predictions ({m})", data=csv, file_name=f"unsup_{m}.csv", mime="text/csv")

                except Exception as ex:
                    st.warning(f"{m} failed: {ex}")
                    st.code(traceback.format_exc())

            # Consensus/hybrid across unsupervised
            if (dataset_type == "Unsupervised") and (('unsup_hybrid' in st.session_state and st.session_state.get('unsup_hybrid')) or ('sup_hybrid' in st.session_state and st.session_state.get('sup_hybrid'))):
                # only perform consensus if >=2 model outputs available
                if len(unsup_results) >= 2:
                    df_flags = pd.DataFrame(unsup_results)
                    flags_num = df_flags.applymap(lambda x: 1 if str(x).lower().startswith("attack") else 0)
                    cons_score = flags_num.mean(axis=1)
                    cons_pred = np.where(cons_score >= 0.5, "Attack", "Normal")
                    st.subheader("Consensus (majority) across unsupervised models")
                    s = pd.Series(cons_pred)
                    st.write(f"Consensus — Attack: {(s=='Attack').sum()} | Normal: {(s=='Normal').sum()}")
                    for g in chosen_graphs:
                        if g in ("Pie", "Bar"):
                            plot_pie_counts(s, title=f"Consensus — {g}")
                        else:
                            plot_generic(X_proc, g, title=f"Consensus — {g}")
                    out = df.copy()
                    out["Consensus_Pred"] = cons_pred
                    st.download_button("Download Consensus predictions", data=out.to_csv(index=False).encode('utf-8'), file_name="unsup_consensus.csv", mime="text/csv")
                else:
                    st.warning("Consensus requested but fewer than 2 unsupervised outputs available.")

    except Exception as e:
        st.error("An unexpected error occurred during run.")
        st.code(traceback.format_exc())

# ---------------- Help / Troubleshooting ----------------
st.markdown("---")
with st.expander("Help & Troubleshooting"):
    st.markdown("""
    **Quick guidance & common fixes**
    - If the app auto-detected the wrong label (e.g. 'syn_flag_count'), use the **Select label column** dropdown and pick the correct `label` or `class` column, or choose `-- None / Unlabeled --` for unsupervised.
    - To avoid `IntCastingNaNError` we do **not** force-cast label column to int; labels are factorized with `pd.factorize()` which handles strings/numbers/NaNs robustly.
    - If CNN2D option is chosen, the number of features must be a perfect square (e.g. 16, 25, 36) — otherwise CNN2D will be skipped automatically with a warning.
    - If TensorFlow isn't installed or environment can't use GPU, deep model options won't appear (or will raise warnings). For demo keep `train_deep` off or use small epochs.
    - If you see `Error running app` in streamlit cloud logs, open the logs (Manage app → Logs). This app prints full stack traces in the UI to make debugging easier.
    - If a model fails, check the printed stack trace block shown under the model's warning.
    """)
