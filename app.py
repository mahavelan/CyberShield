import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF logs

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix

# Classical ML
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Unsupervised
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, Model, Input
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, LSTM, Bidirectional, Attention, Softmax, Reshape
    from tensorflow.keras.optimizers import Adam
except Exception:
    USE_TF = False

# ---------------------------
# Utility
# ---------------------------
STRICT_LABEL_NAMES = {"label", "class", "target", "attack", "y"}

def normalize_name(colname: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(colname)).strip().lower()

def find_label_column_strict(df: pd.DataFrame):
    for col in df.columns:
        if normalize_name(col) in STRICT_LABEL_NAMES:
            return col
    return None

def preprocess_features(df: pd.DataFrame, drop_cols=None):
    dfc = df.copy()
    if drop_cols:
        dfc = dfc.drop(columns=[c for c in drop_cols if c in dfc.columns], errors="ignore")
    dfc = dfc.dropna(axis=1, how="all")

    for c in dfc.columns:
        if pd.api.types.is_numeric_dtype(dfc[c]):
            dfc[c] = dfc[c].fillna(dfc[c].median())
        else:
            dfc[c] = dfc[c].fillna("missing")

    dfc = pd.get_dummies(dfc, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dfc.values)
    return pd.DataFrame(X_scaled, columns=dfc.columns, index=dfc.index)

# ---------------------------
# Deep models
# ---------------------------
def build_keras_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_cnn1d(input_len):
    model = Sequential([
        Input(shape=(input_len, 1)),
        Conv1D(32, 3, activation="relu"),
        MaxPooling1D(2),
        Conv1D(64, 3, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_cnn2d(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_lstm(input_len):
    model = Sequential([
        Input(shape=(input_len, 1)),
        LSTM(64),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_bilstm(input_len):
    model = Sequential([
        Input(shape=(input_len, 1)),
        Bidirectional(LSTM(64)),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_attention(input_len):
    inputs = Input(shape=(input_len, 1))
    lstm_out = LSTM(64, return_sequences=True)(inputs)
    attn_out = Attention()([lstm_out, lstm_out])
    x = Flatten()(attn_out)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ---------------------------
# Metrics & plotting
# ---------------------------
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
    }

def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="CyberShield", layout="wide")
st.title("🛡 CyberShield — Intrusion Detection")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.write(df.head())

label_col = find_label_column_strict(df)
dataset_type = "Supervised" if label_col else "Unsupervised"

st.sidebar.header("Options")

if dataset_type == "Supervised":
    st.sidebar.write("Supervised models")
    classical = ["LogReg","RandomForest","DecisionTree","KNN","SVM","NaiveBayes","GB","AdaBoost","MLP"]
    deep = []
    if USE_TF:
        deep = ["Keras-MLP","CNN-1D","CNN-2D","LSTM","BiLSTM","Attention"]
    use_hybrid = st.sidebar.checkbox("Hybrid (multiple models)?", value=False)
    if use_hybrid:
        chosen_models = st.sidebar.multiselect("Choose models", classical+deep)
    else:
        chosen_models = [st.sidebar.selectbox("Choose one", classical+deep)]
    train_deep_toggle = st.sidebar.checkbox("Train deep models (slow but accurate)?", value=False)
else:
    st.sidebar.write("Unsupervised models")
    chosen_models = [st.sidebar.selectbox("Choose one", ["IsolationForest","KMeans","DBSCAN","PCA"])]

if st.sidebar.button("Run"):
    if dataset_type == "Supervised":
        X = df.drop(columns=[label_col])
        y = df[label_col].factorize()[0]
        X_proc = preprocess_features(X)
        X_train,X_test,y_train,y_test = train_test_split(X_proc,y,test_size=0.3,random_state=42)

        results = {}
        for m in chosen_models:
            if m == "LogReg":
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train,y_train); yp=model.predict(X_test)
            elif m == "RandomForest":
                model=RandomForestClassifier(); model.fit(X_train,y_train); yp=model.predict(X_test)
            elif m == "DecisionTree":
                model=DecisionTreeClassifier(); model.fit(X_train,y_train); yp=model.predict(X_test)
            elif m == "KNN":
                model=KNeighborsClassifier(); model.fit(X_train,y_train); yp=model.predict(X_test)
            elif m == "SVM":
                model=SVC(probability=True); model.fit(X_train,y_train); yp=model.predict(X_test)
            elif m == "NaiveBayes":
                model=GaussianNB(); model.fit(X_train,y_train); yp=model.predict(X_test)
            elif m == "GB":
                model=GradientBoostingClassifier(); model.fit(X_train,y_train); yp=model.predict(X_test)
            elif m == "AdaBoost":
                model=AdaBoostClassifier(); model.fit(X_train,y_train); yp=model.predict(X_test)
            elif m == "MLP":
                model=MLPClassifier(max_iter=500); model.fit(X_train,y_train); yp=model.predict(X_test)
            elif USE_TF and m=="Keras-MLP":
                model=build_keras_mlp(X_train.shape[1])
                epochs=10 if train_deep_toggle else 3
                model.fit(X_train,y_train,epochs=epochs,verbose=0)
                yp=(model.predict(X_test).ravel()>=0.5).astype(int)
            elif USE_TF and m=="CNN-1D":
                model=build_cnn1d(X_train.shape[1])
                Xtr=X_train.values.reshape(-1,X_train.shape[1],1)
                Xte=X_test.values.reshape(-1,X_test.shape[1],1)
                epochs=10 if train_deep_toggle else 3
                model.fit(Xtr,y_train,epochs=epochs,verbose=0)
                yp=(model.predict(Xte).ravel()>=0.5).astype(int)
            elif USE_TF and m=="CNN-2D":
                side=int(np.sqrt(X_train.shape[1]))
                if side*side!=X_train.shape[1]:
                    st.warning("CNN-2D requires square input features."); continue
                model=build_cnn2d((side,side,1))
                Xtr=X_train.values.reshape(-1,side,side,1)
                Xte=X_test.values.reshape(-1,side,side,1)
                epochs=10 if train_deep_toggle else 3
                model.fit(Xtr,y_train,epochs=epochs,verbose=0)
                yp=(model.predict(Xte).ravel()>=0.5).astype(int)
            elif USE_TF and m=="LSTM":
                model=build_lstm(X_train.shape[1])
                Xtr=X_train.values.reshape(-1,X_train.shape[1],1)
                Xte=X_test.values.reshape(-1,X_test.shape[1],1)
                epochs=10 if train_deep_toggle else 3
                model.fit(Xtr,y_train,epochs=epochs,verbose=0)
                yp=(model.predict(Xte).ravel()>=0.5).astype(int)
            elif USE_TF and m=="BiLSTM":
                model=build_bilstm(X_train.shape[1])
                Xtr=X_train.values.reshape(-1,X_train.shape[1],1)
                Xte=X_test.values.reshape(-1,X_test.shape[1],1)
                epochs=10 if train_deep_toggle else 3
                model.fit(Xtr,y_train,epochs=epochs,verbose=0)
                yp=(model.predict(Xte).ravel()>=0.5).astype(int)
            elif USE_TF and m=="Attention":
                model=build_attention(X_train.shape[1])
                Xtr=X_train.values.reshape(-1,X_train.shape[1],1)
                Xte=X_test.values.reshape(-1,X_test.shape[1],1)
                epochs=10 if train_deep_toggle else 3
                model.fit(Xtr,y_train,epochs=epochs,verbose=0)
                yp=(model.predict(Xte).ravel()>=0.5).astype(int)
            else:
                continue
            results[m]=compute_metrics(y_test,yp)
            st.subheader(m)
            st.write(results[m])
            plot_confusion(y_test,yp)
