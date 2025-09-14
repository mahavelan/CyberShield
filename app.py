import os, re, warnings, traceback
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF logs
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential, Model, Input
    from tensorflow.keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, LSTM, Bidirectional, Attention, Softmax
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel("ERROR")
except Exception:
    USE_TF = False

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
STRICT_LABEL_NAMES = {"label","class","target","attack","y","output","result","category","type","true_label"}

def normalize_name(col): return re.sub(r"[^0-9a-zA-Z]+","",str(col)).lower()

def find_label_column(df):
    for col in df.columns:
        if normalize_name(col) in STRICT_LABEL_NAMES:
            return col
    return None

def safe_read(uploaded):
    return pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)

def preprocess_features(df, drop_cols=None):
    dfc = df.copy()
    if drop_cols: dfc.drop(columns=[c for c in drop_cols if c in dfc], inplace=True)
    dfc.dropna(axis=1, how="all", inplace=True)

    for c in dfc.columns:
        if pd.api.types.is_numeric_dtype(dfc[c]):
            dfc[c] = dfc[c].fillna(dfc[c].median() if dfc[c].dropna().size>0 else 0)
        else:
            dfc[c] = dfc[c].fillna("missing")

    cat_cols = dfc.select_dtypes(include=["object","category"]).columns
    if len(cat_cols): dfc = pd.get_dummies(dfc, columns=cat_cols, drop_first=True)

    nunq = dfc.nunique(); const_cols = nunq[nunq<=1].index
    dfc.drop(columns=const_cols, inplace=True, errors="ignore")

    scaler = StandardScaler()
    X = scaler.fit_transform(dfc.values)
    return pd.DataFrame(X, columns=dfc.columns), {"scaler":scaler,"features":dfc.columns.tolist()}

# ---------------------------------------------------
# Deep model builders
# ---------------------------------------------------
def build_mlp(input_dim):
    m = Sequential([Input(shape=(input_dim,)),Dense(128,activation="relu"),Dropout(0.2),Dense(64,activation="relu"),Dense(1,activation="sigmoid")])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]); return m

def build_cnn1d(input_len):
    m = Sequential([Input(shape=(input_len,1)),Conv1D(32,3,activation="relu"),MaxPooling1D(2),Conv1D(64,3,activation="relu"),MaxPooling1D(2),Flatten(),Dense(64,activation="relu"),Dense(1,activation="sigmoid")])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]); return m

def build_cnn2d(input_shape):
    m = Sequential([Input(shape=input_shape),Conv2D(32,(3,3),activation="relu"),MaxPooling2D((2,2)),Conv2D(64,(3,3),activation="relu"),MaxPooling2D((2,2)),Flatten(),Dense(64,activation="relu"),Dense(1,activation="sigmoid")])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]); return m

def build_lstm(input_len):
    m = Sequential([Input(shape=(input_len,1)),LSTM(64),Dense(1,activation="sigmoid")])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]); return m

def build_blstm(input_len):
    m = Sequential([Input(shape=(input_len,1)),Bidirectional(LSTM(64)),Dense(1,activation="sigmoid")])
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]); return m

def build_attention(input_len):
    inputs = Input(shape=(input_len,1))
    x = LSTM(64, return_sequences=True)(inputs)
    attn = Attention()([x,x])
    x = Flatten()(attn)
    outputs = Dense(1,activation="sigmoid")(x)
    m = Model(inputs, outputs)
    m.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m

# ---------------------------------------------------
# Metrics & plots
# ---------------------------------------------------
def metrics(y_true,y_pred):
    return {
        "accuracy":accuracy_score(y_true,y_pred),
        "precision":precision_score(y_true,y_pred,average="weighted",zero_division=0),
        "recall":recall_score(y_true,y_pred,average="weighted",zero_division=0),
        "f1":f1_score(y_true,y_pred,average="weighted",zero_division=0),
        "rmse":mean_squared_error(y_true,y_pred,squared=False)
    }

def plot_confusion(y_true,y_pred,labels=None):
    cm = confusion_matrix(y_true,y_pred)
    fig,ax=plt.subplots()
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    if labels is not None: ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    st.pyplot(fig); plt.close(fig)

def plot_generic(df, kind, title=""):
    fig,ax=plt.subplots()
    try:
        if kind=="Bar": df.sum().plot(kind="bar",ax=ax)
        elif kind=="Pie": df.iloc[:,0].value_counts().plot(kind="pie",autopct="%1.1f%%",ax=ax)
        elif kind=="Heatmap": sns.heatmap(df.corr(numeric_only=True),cmap="coolwarm",ax=ax)
        elif kind=="Line": df.reset_index(drop=True).plot(ax=ax)
        elif kind=="Histogram": df.hist(ax=ax)
        elif kind=="Box": sns.boxplot(data=df,ax=ax)
        ax.set_title(title or kind); st.pyplot(fig)
    except Exception as e: st.warning(f"Plot {kind} failed: {e}")
    finally: plt.close(fig)

# ---------------------------------------------------
# UI
# ---------------------------------------------------
st.set_page_config(page_title="CyberShield",layout="wide")
st.title("🛡 CyberShield — Intrusion Detection Web App")

uploaded = st.file_uploader("Upload CSV/Excel dataset",type=["csv","xlsx"])
if not uploaded: st.stop()
try: df = safe_read(uploaded)
except Exception as e: st.error(f"Read error: {e}"); st.stop()
st.dataframe(df.head())

label_col = find_label_column(df)
dataset_type = "Supervised" if label_col else "Unsupervised"
st.info(f"Detected: {dataset_type} mode" + (f" (label: {label_col})" if label_col else ""))

# Sidebar
st.sidebar.header("Options")
if dataset_type=="Supervised":
    SUP_CLASSICAL=["Logistic Regression","Random Forest","Decision Tree","KNN","SVM","Naive Bayes","Gradient Boosting","AdaBoost","MLP (sklearn)"]
    SUP_DEEP=[]
    if USE_TF: SUP_DEEP=["Keras-MLP","Keras-1D-CNN","Keras-2D-CNN","LSTM","BiLSTM","Attention"]
    hybrid=st.sidebar.checkbox("Hybrid Ensemble?",key="h_sup")
    chosen=st.sidebar.multiselect("Models",SUP_CLASSICAL+SUP_DEEP) if hybrid else [st.sidebar.selectbox("Model",SUP_CLASSICAL+SUP_DEEP)]
    train_deep=st.sidebar.checkbox("Train deep models",False)
else:
    UNSUP_CLASSICAL=["Isolation Forest","One-Class SVM","KMeans","DBSCAN","PCA-heuristic"]
    UNSUP_DEEP=["Autoencoder"] if USE_TF else []
    hybrid=st.sidebar.checkbox("Hybrid Consensus?",key="h_unsup")
    chosen=st.sidebar.multiselect("Models",UNSUP_CLASSICAL+UNSUP_DEEP) if hybrid else [st.sidebar.selectbox("Model",UNSUP_CLASSICAL+UNSUP_DEEP)]
    contamination=st.sidebar.slider("Contamination",0.001,0.5,0.05,0.01)

multi_graph=st.sidebar.checkbox("Multiple Graphs?",key="multi_g")
GRAPH_OPTIONS=["Confusion Matrix","Bar","Pie","Line","Histogram","Box","Heatmap"]
graphs=st.sidebar.multiselect("Graphs",GRAPH_OPTIONS) if multi_graph else [st.sidebar.selectbox("Graph",GRAPH_OPTIONS)]
run=st.sidebar.button("Run")

# ---------------------------------------------------
# Run
# ---------------------------------------------------
if run:
    try:
        if dataset_type=="Supervised":
            X_raw=df.drop(columns=[label_col]); y_raw=df[label_col]
            y_codes,uniques=pd.factorize(y_raw.astype(str)); label_map={i:v for i,v in enumerate(uniques)}
            X_proc,pipeline=preprocess_features(X_raw)
            X_train,X_test,y_train,y_test=train_test_split(X_proc,y_codes,test_size=0.3,random_state=42,stratify=y_codes)
            preds={}; mets={}

            def run_clf(name,clf):
                clf.fit(X_train,y_train); yp=clf.predict(X_test); preds[name]=yp; mets[name]=metrics(y_test,yp)

            for m in chosen:
                if m=="Logistic Regression": run_clf(m,LogisticRegression(max_iter=1000))
                elif m=="Random Forest": run_clf(m,RandomForestClassifier())
                elif m=="Decision Tree": run_clf(m,DecisionTreeClassifier())
                elif m=="KNN": run_clf(m,KNeighborsClassifier())
                elif m=="SVM": run_clf(m,SVC())
                elif m=="Naive Bayes": run_clf(m,GaussianNB())
                elif m=="Gradient Boosting": run_clf(m,GradientBoostingClassifier())
                elif m=="AdaBoost": run_clf(m,AdaBoostClassifier())
                elif m=="MLP (sklearn)": run_clf(m,MLPClassifier(max_iter=500))
                elif USE_TF and m=="Keras-MLP":
                    model=build_mlp(X_train.shape[1]); model.fit(X_train,y_train,epochs=3 if not train_deep else 10,verbose=0)
                    yp=(model.predict(X_test).ravel()>=0.5).astype(int); preds[m]=yp; mets[m]=metrics(y_test,yp)
                elif USE_TF and m=="Keras-1D-CNN":
                    model=build_cnn1d(X_train.shape[1]); Xtr=X_train[...,None]; Xte=X_test[...,None]
                    model.fit(Xtr,y_train,epochs=3 if not train_deep else 8,verbose=0)
                    yp=(model.predict(Xte).ravel()>=0.5).astype(int); preds[m]=yp; mets[m]=metrics(y_test,yp)
                elif USE_TF and m=="Keras-2D-CNN":
                    side=int(np.sqrt(X_train.shape[1])); Xtr=X_train[:,:side*side].reshape((-1,side,side,1))
                    Xte=X_test[:,:side*side].reshape((-1,side,side,1))
                    model=build_cnn2d((side,side,1)); model.fit(Xtr,y_train,epochs=3,verbose=0)
                    yp=(model.predict(Xte).ravel()>=0.5).astype(int); preds[m]=yp; mets[m]=metrics(y_test,yp)
                elif USE_TF and m=="LSTM":
                    model=build_lstm(X_train.shape[1]); Xtr=X_train[...,None]; Xte=X_test[...,None]
                    model.fit(Xtr,y_train,epochs=3,verbose=0)
                    yp=(model.predict(Xte).ravel()>=0.5).astype(int); preds[m]=yp; mets[m]=metrics(y_test,yp)
                elif USE_TF and m=="BiLSTM":
                    model=build_blstm(X_train.shape[1]); Xtr=X_train[...,None]; Xte=X_test[...,None]
                    model.fit(Xtr,y_train,epochs=3,verbose=0)
                    yp=(model.predict(Xte).ravel()>=0.5).astype(int); preds[m]=yp; mets[m]=metrics(y_test,yp)
                elif USE_TF and m=="Attention":
                    model=build_attention(X_train.shape[1]); Xtr=X_train[...,None]; Xte=X_test[...,None]
                    model.fit(Xtr,y_train,epochs=3,verbose=0)
                    yp=(model.predict(Xte).ravel()>=0.5).astype(int); preds[m]=yp; mets[m]=metrics(y_test,yp)

            st.dataframe(pd.DataFrame(mets).T)
            for name,yp in preds.items():
                if "Confusion Matrix" in graphs: plot_confusion(y_test,yp,[label_map[i] for i in range(len(uniques))])
                for g in graphs: 
                    if g!="Confusion Matrix": plot_generic(pd.DataFrame(X_test,columns=pipeline["features"]),g,title=f"{name}-{g}")

        else: # unsupervised
            X_proc,pipeline=preprocess_features(df); X=X_proc.values
            unsup_preds={}
            for m in chosen:
                if m=="Isolation Forest": p=IsolationForest(contamination=contamination).fit_predict(X); lbl=np.where(p==-1,"Attack","Normal")
                elif m=="One-Class SVM": p=OneClassSVM(nu=contamination).fit_predict(X); lbl=np.where(p==-1,"Attack","Normal")
                elif m=="KMeans": cl=KMeans(2).fit_predict(X); lbl=np.where(cl==pd.Series(cl).value_counts().idxmin(),"Attack","Normal")
                elif m=="DBSCAN": cl=DBSCAN().fit_predict(X); lbl=np.where(cl==-1,"Attack","Normal")
                elif m=="PCA-heuristic": comp=PCA(5).fit_transform(X); dist=np.linalg.norm(comp-comp.mean(0),axis=1); thr=np.percentile(dist,100*(1-contamination)); lbl=np.where(dist>=thr,"Attack","Normal")
                elif USE_TF and m=="Autoencoder":
                    ae=Sequential([Input((X.shape[1],)),Dense(16,activation="relu"),Dense(X.shape[1],activation="linear")]); ae.compile("adam","mse")
                    ae.fit(X,X,epochs=5,verbose=0); rec=ae.predict(X); err=np.mean((rec-X)**2,axis=1); thr=np.percentile(err,100*(1-contamination)); lbl=np.where(err>=thr,"Attack","Normal")
                else: continue
                unsup_preds[m]=lbl
                s=pd.Series(lbl); st.write(f"{m}: Attack={sum(s=='Attack')} Normal={sum(s=='Normal')}")
                for g in graphs: plot_generic(X_proc,g,title=f"{m}-{g}")
    except Exception as e:
        st.error("Error: "+str(e))
        st.text(traceback.format_exc())
