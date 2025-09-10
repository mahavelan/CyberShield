# app.py
"""
CYBERSHIELD — Streamlit Intrusion Detection Web App
- Strict label detection
- Auto supervised / unsupervised routing
- Hybrid models only if checkbox ON
- Multiple visualizations only if checkbox ON
- Includes CNN1D, CNN2D, LSTM, RNN, BLSTM, Attention+Softmax
"""

import os, re, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

# deep learning
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, Input, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D,
        Flatten, LSTM, SimpleRNN, Bidirectional, Attention, Softmax
    )
    from tensorflow.keras.optimizers import Adam
except Exception:
    USE_TF = False

# --------------------
# Utilities
# --------------------
STRICT_LABELS = {"label","class","target","attack","y","output","result","category","type","true_label"}

def normalize_name(c): return re.sub(r"[^0-9a-zA-Z]+","",c).lower()

def find_label_column(df):
    for c in df.columns:
        if normalize_name(c) in STRICT_LABELS:
            return c
    return None

def preprocess_features(df, drop_cols=None):
    df = df.copy()
    if drop_cols: df.drop(columns=[c for c in drop_cols if c in df], inplace=True, errors="ignore")
    df.dropna(axis=1, how="all", inplace=True)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = df[c].fillna("missing")
    cat_cols = df.select_dtypes(include=["object","category"]).columns
    if len(cat_cols)>0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df.values)
    return pd.DataFrame(Xs, columns=df.columns), {"scaler":scaler,"feature_columns":df.columns}

def metrics_report(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true,y_pred),
        "precision": precision_score(y_true,y_pred,average="weighted",zero_division=0),
        "recall": recall_score(y_true,y_pred,average="weighted",zero_division=0),
        "f1": f1_score(y_true,y_pred,average="weighted",zero_division=0),
        "rmse": mean_squared_error(y_true,y_pred,squared=False)
    }

def majority_vote(preds):
    return pd.DataFrame(preds).mode(axis=1)[0].values

def plot_confusion(y_true,y_pred,labels=None):
    cm = confusion_matrix(y_true,y_pred)
    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    if labels is not None:
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    st.pyplot(fig); plt.close(fig)

def plot_generic(df, kind, x=None, y=None, title=None):
    fig,ax=plt.subplots()
    if kind=="Bar": df.sum().plot(kind="bar",ax=ax)
    elif kind=="Pie": df.iloc[:,0].value_counts().plot(kind="pie",ax=ax,autopct="%1.1f%%")
    elif kind=="Line": df.plot(ax=ax)
    elif kind=="Histogram": df.hist(ax=ax)
    elif kind=="Box": df.plot(kind="box",ax=ax)
    elif kind=="Heatmap": sns.heatmap(df.corr(),cmap="coolwarm",ax=ax)
    ax.set_title(title or kind)
    st.pyplot(fig); plt.close(fig)

# --------------------
# Deep models
# --------------------
def keras_mlp(input_dim):
    m=Sequential([Input(shape=(input_dim,)),Dense(128,activation="relu"),Dropout(0.2),
                  Dense(64,activation="relu"),Dense(1,activation="sigmoid")])
    m.compile(Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"]); return m

def keras_cnn1d(input_dim):
    m=Sequential([Input(shape=(input_dim,1)),Conv1D(32,3,activation="relu"),MaxPooling1D(2),
                  Conv1D(64,3,activation="relu"),MaxPooling1D(2),Flatten(),
                  Dense(64,activation="relu"),Dense(1,activation="sigmoid")])
    m.compile(Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"]); return m

def keras_cnn2d(input_shape):
    m=Sequential([Input(shape=input_shape),Conv2D(32,(3,3),activation="relu"),MaxPooling2D((2,2)),
                  Conv2D(64,(3,3),activation="relu"),MaxPooling2D((2,2)),Flatten(),
                  Dense(64,activation="relu"),Dense(1,activation="sigmoid")])
    m.compile(Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"]); return m

def keras_lstm(input_dim):
    m=Sequential([Input(shape=(1,input_dim)),LSTM(64),Dense(1,activation="sigmoid")])
    m.compile(Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"]); return m

def keras_rnn(input_dim):
    m=Sequential([Input(shape=(1,input_dim)),SimpleRNN(64),Dense(1,activation="sigmoid")])
    m.compile(Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"]); return m

def keras_blstm(input_dim):
    m=Sequential([Input(shape=(1,input_dim)),Bidirectional(LSTM(64)),Dense(1,activation="sigmoid")])
    m.compile(Adam(1e-3),loss="binary_crossentropy",metrics=["accuracy"]); return m

def keras_attention(input_dim):
    inp=Input(shape=(1,input_dim))
    x=LSTM(64,return_sequences=True)(inp)
    att=Attention()([x,x])
    x=tf.reduce_mean(att,axis=1)
    out=Dense(1,activation="softmax")(x)
    m=Model(inputs=inp,outputs=out)
    m.compile(Adam(1e-3),loss="categorical_crossentropy",metrics=["accuracy"])
    return m

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="CyberShield",layout="wide")
st.title("🛡 CyberShield IDS")

uploaded=st.file_uploader("Upload dataset",type=["csv","xlsx"])
if not uploaded: st.stop()
df=pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
st.write(df.head())

label_col=find_label_column(df)
dataset_type="Supervised" if label_col else "Unsupervised"
st.info(f"Detected: {dataset_type}")

st.sidebar.header("Options")
if dataset_type=="Supervised":
    models=["Logistic Regression","Random Forest","Decision Tree","KNN","SVM","Naive Bayes",
            "Gradient Boosting","AdaBoost","MLP","Keras-MLP","CNN1D","CNN2D",
            "LSTM","RNN","BLSTM","Attention+Softmax"]
else:
    models=["Isolation Forest","One-Class SVM","KMeans","DBSCAN","PCA"]

hybrid=st.sidebar.checkbox("Enable Hybrid?")
chosen=models
if hybrid:
    chosen=st.sidebar.multiselect("Choose 2+ models",models)
else:
    chosen=[st.sidebar.selectbox("Choose one model",models)]

multi_graph=st.sidebar.checkbox("Enable Multiple Graphs?")
graphs=["Confusion Matrix","Bar","Pie","Line","Histogram","Box","Heatmap"]
chosen_graphs=st.sidebar.multiselect("Choose graphs",graphs) if multi_graph else [st.sidebar.selectbox("Choose one graph",graphs)]

if st.sidebar.button("Run"):
    if dataset_type=="Supervised":
        X=df.drop(columns=[label_col]); y=df[label_col]
        y,y_classes=pd.factorize(y)
        X_proc,_=preprocess_features(X)
        Xtr,Xte,ytr,yte=train_test_split(X_proc,y,test_size=0.3,random_state=42,stratify=y)
        preds={}; metrics={}
        for m in chosen:
            if m=="Logistic Regression": clf=LogisticRegression(max_iter=1000); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif m=="Random Forest": clf=RandomForestClassifier(); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif m=="Decision Tree": clf=DecisionTreeClassifier(); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif m=="KNN": clf=KNeighborsClassifier(); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif m=="SVM": clf=SVC(); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif m=="Naive Bayes": clf=GaussianNB(); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif m=="Gradient Boosting": clf=GradientBoostingClassifier(); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif m=="AdaBoost": clf=AdaBoostClassifier(); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif m=="MLP": clf=MLPClassifier(max_iter=500); clf.fit(Xtr,ytr); yp=clf.predict(Xte)
            elif USE_TF and m=="Keras-MLP": mdl=keras_mlp(Xtr.shape[1]); mdl.fit(Xtr,ytr,epochs=3,verbose=0); yp=(mdl.predict(Xte).ravel()>0.5).astype(int)
            elif USE_TF and m=="CNN1D": mdl=keras_cnn1d(Xtr.shape[1]); mdl.fit(Xtr.reshape(-1,Xtr.shape[1],1),ytr,epochs=3,verbose=0); yp=(mdl.predict(Xte.reshape(-1,Xte.shape[1],1)).ravel()>0.5).astype(int)
            elif USE_TF and m=="CNN2D": side=int(np.sqrt(Xtr.shape[1])); mdl=keras_cnn2d((side,side,1)); Xtr2=Xtr.values.reshape(-1,side,side,1); Xte2=Xte.values.reshape(-1,side,side,1); mdl.fit(Xtr2,ytr,epochs=3,verbose=0); yp=(mdl.predict(Xte2).ravel()>0.5).astype(int)
            elif USE_TF and m=="LSTM": mdl=keras_lstm(Xtr.shape[1]); mdl.fit(Xtr.reshape(-1,1,Xtr.shape[1]),ytr,epochs=3,verbose=0); yp=(mdl.predict(Xte.reshape(-1,1,Xte.shape[1])).ravel()>0.5).astype(int)
            elif USE_TF and m=="RNN": mdl=keras_rnn(Xtr.shape[1]); mdl.fit(Xtr.reshape(-1,1,Xtr.shape[1]),ytr,epochs=3,verbose=0); yp=(mdl.predict(Xte.reshape(-1,1,Xte.shape[1])).ravel()>0.5).astype(int)
            elif USE_TF and m=="BLSTM": mdl=keras_blstm(Xtr.shape[1]); mdl.fit(Xtr.reshape(-1,1,Xtr.shape[1]),ytr,epochs=3,verbose=0); yp=(mdl.predict(Xte.reshape(-1,1,Xte.shape[1])).ravel()>0.5).astype(int)
            elif USE_TF and m=="Attention+Softmax": mdl=keras_attention(Xtr.shape[1]); yoh=tf.keras.utils.to_categorical(ytr); mdl.fit(Xtr.reshape(-1,1,Xtr.shape[1]),yoh,epochs=3,verbose=0); yp=np.argmax(mdl.predict(Xte.reshape(-1,1,Xte.shape[1])),axis=1)
            else: continue
            preds[m]=yp; metrics[m]=metrics_report(yte,yp)
        if hybrid and len(preds)>=2:
            preds["Hybrid"]=majority_vote(list(preds.values()))
            metrics["Hybrid"]=metrics_report(yte,preds["Hybrid"])
        st.dataframe(pd.DataFrame(metrics).T)
        for m,yp in preds.items():
            st.subheader(m)
            if "Confusion Matrix" in chosen_graphs: plot_confusion(yte,yp,list(y_classes))
            for g in chosen_graphs:
                if g!="Confusion Matrix": plot_generic(pd.DataFrame(Xte),g,title=f"{m}-{g}")
