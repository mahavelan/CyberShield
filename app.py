# app.py
"""
CYBERSHIELD — Intrusion Detection Web App
- Strict label detection by name only (no heuristics)
- Auto-route to supervised / unsupervised
- Enforce: single model vs hybrid model (checkbox)
- Enforce: single visualization vs multi-graphs (checkbox)
- Supports deep models: CNN-1D, CNN-2D, LSTM, RNN, BLSTM, Attention, Softmax
"""

import os, re, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error

# Classical supervised
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Unsupervised
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, Model
    from tensorflow.keras.layers import (Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten,
                                         Conv2D, MaxPooling2D, Reshape,
                                         LSTM, SimpleRNN, Bidirectional, Layer)
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel("ERROR")
except Exception:
    USE_TF = False

# ---------------------------
# Utils
# ---------------------------
STRICT_LABEL_NAMES = {"label","class","target","attack","y","output","result","category","type","true_label"}

def normalize_name(c): return re.sub(r"[^0-9a-zA-Z]+","",str(c)).lower()
def find_label_column_strict(df): 
    for col in df.columns:
        if normalize_name(col) in STRICT_LABEL_NAMES:
            return col
    return None
def safe_read_file(uploaded):
    return pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

def preprocess_features(df, drop_cols=None):
    dfc = df.copy()
    if drop_cols: dfc = dfc.drop(columns=[c for c in drop_cols if c in dfc.columns], errors='ignore')
    dfc = dfc.dropna(axis=1, how="all")
    for c in dfc.columns:
        if pd.api.types.is_numeric_dtype(dfc[c]):
            dfc[c] = dfc[c].fillna(dfc[c].median() if dfc[c].dropna().size else 0)
        else:
            dfc[c] = dfc[c].fillna("missing")
    dfc = pd.get_dummies(dfc, drop_first=True)
    nunq = dfc.nunique()
    dfc = dfc.drop(columns=nunq[nunq<=1].index, errors="ignore")
    X_scaled = StandardScaler().fit_transform(dfc.values)
    return pd.DataFrame(X_scaled, columns=dfc.columns), {"feature_columns":dfc.columns.tolist()}

def compute_metrics(y_true, y_pred):
    return dict(
        accuracy=accuracy_score(y_true,y_pred),
        precision=precision_score(y_true,y_pred,average="weighted",zero_division=0),
        recall=recall_score(y_true,y_pred,average="weighted",zero_division=0),
        f1=f1_score(y_true,y_pred,average="weighted",zero_division=0),
        rmse=mean_squared_error(y_true,y_pred,squared=False)
    )

def plot_confusion(y_true,y_pred,labels=None):
    cm=confusion_matrix(y_true,y_pred)
    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
    st.pyplot(fig); plt.close(fig)

# ---------------------------
# Keras models
# ---------------------------
if USE_TF:
    def build_mlp(input_dim, num_classes=2):
        out_units = num_classes if num_classes>2 else 1
        out_act   = "softmax" if num_classes>2 else "sigmoid"
        loss      = "sparse_categorical_crossentropy" if num_classes>2 else "binary_crossentropy"
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128,activation="relu"),
            Dropout(0.2),
            Dense(64,activation="relu"),
            Dense(out_units,activation=out_act)
        ])
        model.compile(optimizer=Adam(1e-3),loss=loss,metrics=["accuracy"])
        return model

    def build_cnn1d(input_len,num_classes=2):
        out_units = num_classes if num_classes>2 else 1
        out_act   = "softmax" if num_classes>2 else "sigmoid"
        loss      = "sparse_categorical_crossentropy" if num_classes>2 else "binary_crossentropy"
        model=Sequential([
            Input(shape=(input_len,1)),
            Conv1D(32,3,activation="relu"), MaxPooling1D(2),
            Conv1D(64,3,activation="relu"), MaxPooling1D(2),
            Flatten(), Dense(64,activation="relu"),
            Dense(out_units,activation=out_act)
        ])
        model.compile(optimizer=Adam(1e-3),loss=loss,metrics=["accuracy"])
        return model

    def build_cnn2d(side,num_classes=2):
        out_units=num_classes if num_classes>2 else 1
        out_act="softmax" if num_classes>2 else "sigmoid"
        loss="sparse_categorical_crossentropy" if num_classes>2 else "binary_crossentropy"
        model=Sequential([
            Input(shape=(side,side,1)),
            Conv2D(32,(3,3),activation="relu"), MaxPooling2D((2,2)),
            Conv2D(64,(3,3),activation="relu"), MaxPooling2D((2,2)),
            Flatten(), Dense(64,activation="relu"),
            Dense(out_units,activation=out_act)
        ])
        model.compile(optimizer=Adam(1e-3),loss=loss,metrics=["accuracy"])
        return model

    def build_lstm(timesteps,features,num_classes=2):
        out_units=num_classes if num_classes>2 else 1
        out_act="softmax" if num_classes>2 else "sigmoid"
        loss="sparse_categorical_crossentropy" if num_classes>2 else "binary_crossentropy"
        inp=Input(shape=(timesteps,features))
        x=LSTM(64)(inp)
        out=Dense(out_units,activation=out_act)(x)
        model=Model(inp,out)
        model.compile(optimizer=Adam(1e-3),loss=loss,metrics=["accuracy"])
        return model

    def build_rnn(timesteps,features,num_classes=2):
        out_units=num_classes if num_classes>2 else 1
        out_act="softmax" if num_classes>2 else "sigmoid"
        loss="sparse_categorical_crossentropy" if num_classes>2 else "binary_crossentropy"
        inp=Input(shape=(timesteps,features))
        x=SimpleRNN(64)(inp)
        out=Dense(out_units,activation=out_act)(x)
        model=Model(inp,out)
        model.compile(optimizer=Adam(1e-3),loss=loss,metrics=["accuracy"])
        return model

    def build_blstm(timesteps,features,num_classes=2):
        out_units=num_classes if num_classes>2 else 1
        out_act="softmax" if num_classes>2 else "sigmoid"
        loss="sparse_categorical_crossentropy" if num_classes>2 else "binary_crossentropy"
        inp=Input(shape=(timesteps,features))
        x=Bidirectional(LSTM(64))(inp)
        out=Dense(out_units,activation=out_act)(x)
        model=Model(inp,out)
        model.compile(optimizer=Adam(1e-3),loss=loss,metrics=["accuracy"])
        return model

    class AttentionLayer(Layer):
        def build(self,input_shape):
            self.W=self.add_weight("att_w",shape=(input_shape[-1],),initializer="random_normal",trainable=True)
        def call(self,inputs):
            score=tf.tensordot(inputs,self.W,axes=[2,0])
            weights=tf.nn.softmax(score,axis=1)
            return tf.reduce_sum(inputs*tf.expand_dims(weights,-1),axis=1)

    def build_attention(timesteps,features,num_classes=2):
        out_units=num_classes if num_classes>2 else 1
        out_act="softmax" if num_classes>2 else "sigmoid"
        loss="sparse_categorical_crossentropy" if num_classes>2 else "binary_crossentropy"
        inp=Input(shape=(timesteps,features))
        x=LSTM(64,return_sequences=True)(inp)
        x=AttentionLayer()(x)
        out=Dense(out_units,activation=out_act)(x)
        model=Model(inp,out)
        model.compile(optimizer=Adam(1e-3),loss=loss,metrics=["accuracy"])
        return model

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="CyberShield", layout="wide")
st.title("🛡 CyberShield — Intrusion Detection Web App")

uploaded=st.file_uploader("Upload CSV/XLSX dataset",type=["csv","xlsx"])
if not uploaded: st.stop()
df=safe_read_file(uploaded)
st.dataframe(df.head())

label_col=find_label_column_strict(df)
dataset_type="Supervised" if label_col else "Unsupervised"
st.write("Detected type:",dataset_type)

# Sidebar
if dataset_type=="Supervised":
    SUP_CLASSICAL=["Logistic Regression","Random Forest","Decision Tree","KNN","SVM (RBF)","Naive Bayes","Gradient Boosting","AdaBoost","MLP (sklearn)"]
    SUP_DEEP=[]
    if USE_TF:
        SUP_DEEP=["Keras-MLP","Keras-1D-CNN","Keras-2D-CNN","Keras-LSTM","Keras-RNN","Keras-BLSTM","Keras-Attention"]
    hybrid=st.sidebar.checkbox("Enable Hybrid?",value=False)
    models=st.sidebar.multiselect("Choose models",SUP_CLASSICAL+SUP_DEEP) if hybrid else [st.sidebar.selectbox("Choose ONE model",SUP_CLASSICAL+SUP_DEEP)]
else:
    st.sidebar.write("Unsupervised models same as before…")
