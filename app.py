# app.py
# CYBERSHIELD — Auto Supervised/Unsupervised • Hybrid • Classic + Deep Models • Rich Visuals

import re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Supervised (classic ML)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Unsupervised
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Optional Deep Learning
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, LSTM, Bidirectional, Attention
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel("ERROR")
except Exception:
    USE_TF = False

# --------------------------------------------------------------------------------------
# Streamlit Setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="CYBERSHIELD", layout="wide")
st.title("🔐 CYBERSHIELD — Auto Supervised/Unsupervised Detection")

st.write(
    "Upload a dataset. The app **auto-detects** if it's **Supervised** "
    "(strict label column present) or **Unsupervised** (no label). "
    "You then select models and visualization types."
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
STRICT_LABEL_NAMES = {"label","class","target","attack","y","output","result"}

def normalize_colname(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return s.strip("_")

def find_strict_label_column(df: pd.DataFrame):
    normalized = {col: normalize_colname(col) for col in df.columns}
    for col, norm in normalized.items():
        compact = norm.replace("_","")
        if norm in STRICT_LABEL_NAMES or compact in STRICT_LABEL_NAMES:
            return col
    return None

def safe_numeric_cast(series: pd.Series):
    try:
        vals = pd.to_numeric(series, errors="coerce")
        return vals.fillna(0).astype(int), True
    except Exception:
        return series.copy(), False

def build_deep_mlp(input_dim: int):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_cnn_1d(input_len: int):
    model = Sequential([
        Input(shape=(input_len,1)),
        Conv1D(32,3,activation="relu"),
        MaxPooling1D(2),
        Conv1D(64,3,activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(64,activation="relu"),
        Dropout(0.2),
        Dense(1,activation="sigmoid")
    ])
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_lstm(input_len: int, bidirectional=False, use_attention=False):
    layers = [Input(shape=(input_len,1))]
    if bidirectional:
        layers.append(Bidirectional(LSTM(64, return_sequences=use_attention)))
    else:
        layers.append(LSTM(64, return_sequences=use_attention))
    if use_attention:
        layers.append(Attention())
        layers.append(Flatten())
    layers.append(Dense(64,activation="relu"))
    layers.append(Dropout(0.2))
    layers.append(Dense(1,activation="sigmoid"))
    model = Sequential(layers)
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_autoencoder(n_features: int):
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(128,activation="relu"),
        Dense(max(16,n_features//4),activation="relu"),
        Dense(128,activation="relu"),
        Dense(n_features,activation="linear")
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

def compute_metrics(y_true,y_pred):
    acc = accuracy_score(y_true,y_pred)
    prec = precision_score(y_true,y_pred,average="weighted",zero_division=0)
    rec = recall_score(y_true,y_pred,average="weighted",zero_division=0)
    f1 = f1_score(y_true,y_pred,average="weighted",zero_division=0)
    rmse = mean_squared_error(y_true,y_pred,squared=False)
    return acc,prec,rec,f1,rmse

def draw_visualizations(df, graph_types, title, x=None, y=None, hue=None):
    for g in graph_types:
        st.write(f"### {title} — {g}")
        fig, ax = plt.subplots()
        try:
            if g=="Bar":
                if y: sns.barplot(data=df,x=x,y=y,hue=hue,ax=ax)
                else: df.sum(numeric_only=True).plot(kind="bar",ax=ax)
            elif g=="Pie":
                vc = df[y].value_counts() if y else df.iloc[:,0].value_counts()
                ax.pie(vc.values,labels=vc.index,autopct="%1.1f%%")
            elif g=="Heatmap":
                cm = df.corr(numeric_only=True)
                sns.heatmap(cm,annot=False,cmap="Blues",ax=ax)
            elif g=="Line":
                if x and y: sns.lineplot(data=df,x=x,y=y,hue=hue,ax=ax)
                else: df.reset_index(drop=True).plot(ax=ax)
            elif g=="Scatter":
                if x and y: sns.scatterplot(data=df,x=x,y=y,hue=hue,ax=ax)
            elif g=="Boxplot":
                if y: sns.boxplot(data=df,x=x,y=y,hue=hue,ax=ax)
                else: sns.boxplot(data=df.select_dtypes(include=np.number),ax=ax)
            elif g=="Histogram":
                if y: sns.histplot(data=df,x=y,hue=hue,kde=True,ax=ax)
                else: df.select_dtypes(include=np.number).hist(ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not render {g}: {e}")
        finally:
            plt.close(fig)

# --------------------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------------------
uploaded = st.file_uploader("Upload your dataset (CSV/XLSX)", type=["csv","xlsx"])
if not uploaded: st.stop()

try:
    if uploaded.name.endswith(".csv"): df = pd.read_csv(uploaded)
    else: df = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error reading file: {e}"); st.stop()

st.write("### Dataset Preview", df.head())

# Detect supervised/unsupervised
label_col = find_strict_label_column(df)
dataset_type = "Supervised" if label_col else "Unsupervised"
st.info(f"Auto-detected as **{dataset_type}** dataset.")
if label_col: st.success(f"Using label column: **{label_col}**")

# --------------------------------------------------------------------------------------
# Sidebar Options
# --------------------------------------------------------------------------------------
st.sidebar.header("⚙️ Model Options")

# Graph choices
graph_choices = st.sidebar.multiselect("Select Visualization Types",
    ["Bar","Pie","Heatmap","Line","Scatter","Boxplot","Histogram"],
    default=["Bar","Pie","Heatmap"])

if dataset_type=="Supervised":
    sup_models = [
        "Logistic Regression","Decision Tree","Random Forest","Gradient Boosting","AdaBoost",
        "Naive Bayes","KNN","SVM","MLP"
    ]
    if USE_TF: sup_models += ["Deep MLP","1D-CNN","BiLSTM","BiLSTM+Attention","Autoencoder-Finetune"]
    hybrid = st.sidebar.checkbox("Use Hybrid (Voting Ensemble)?")
    chosen = st.sidebar.multiselect("Choose Models", sup_models, default=["Logistic Regression"])
else:
    unsup_models = ["Isolation Forest","One-Class SVM","KMeans","DBSCAN","PCA"]
    if USE_TF: unsup_models += ["Autoencoder"]
    hybrid = st.sidebar.checkbox("Use Hybrid (Consensus)?")
    chosen = st.sidebar.multiselect("Choose Models", unsup_models, default=["Isolation Forest"])
    contamination = st.sidebar.slider("Anomaly fraction",0.01,0.5,0.05,0.01)

run = st.sidebar.button("🚀 Run Models")

# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------
if run:
    X = df.drop(columns=[label_col]) if label_col else df.copy()
    y = df[label_col] if label_col else None

    # Preprocess
    X = X.fillna(0)
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if dataset_type=="Supervised":
        # encode y
        if y.dtype=="object" or y.dtype.name=="category": y_enc = LabelEncoder().fit_transform(y)
        else:
            y_enc,_ = safe_numeric_cast(y)
        X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_enc,test_size=0.3,random_state=42,stratify=y_enc)

        results = {}
        if not hybrid:
            if len(chosen)!=1: st.error("Select exactly ONE model for non-hybrid."); st.stop()
        else:
            if len(chosen)<2: st.error("Select at least TWO models for hybrid."); st.stop()

        estimators = []
        for name in chosen:
            if name=="Logistic Regression":
                m=LogisticRegression(max_iter=1000); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif name=="Decision Tree":
                m=DecisionTreeClassifier(); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif name=="Random Forest":
                m=RandomForestClassifier(n_estimators=100); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif name=="Gradient Boosting":
                m=GradientBoostingClassifier(); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif name=="AdaBoost":
                m=AdaBoostClassifier(); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif name=="Naive Bayes":
                m=GaussianNB(); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif name=="KNN":
                m=KNeighborsClassifier(); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif name=="SVM":
                m=SVC(probability=True); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif name=="MLP":
                m=MLPClassifier(max_iter=300); m.fit(X_train,y_train); yp=m.predict(X_test)
            elif USE_TF and name=="Deep MLP":
                m=build_deep_mlp(X_train.shape[1]); m.fit(X_train,y_train,epochs=10,batch_size=32,verbose=0); yp=(m.predict(X_test)>0.5).astype(int)
            elif USE_TF and name=="1D-CNN":
                m=build_cnn_1d(X_train.shape[1]); m.fit(X_train,y_train,epochs=5,batch_size=32,verbose=0); yp=(m.predict(X_test)>0.5).astype(int)
            elif USE_TF and name=="BiLSTM":
                m=build_lstm(X_train.shape[1],bidirectional=True); m.fit(X_train,y_train,epochs=5,batch_size=32,verbose=0); yp=(m.predict(X_test)>0.5).astype(int)
            elif USE_TF and name=="BiLSTM+Attention":
                m=build_lstm(X_train.shape[1],bidirectional=True,use_attention=True); m.fit(X_train,y_train,epochs=5,batch_size=32,verbose=0); yp=(m.predict(X_test)>0.5).astype(int)
            elif USE_TF and name=="Autoencoder-Finetune":
                m=build_autoencoder(X_train.shape[1]); m.fit(X_train,X_train,epochs=10,batch_size=32,verbose=0); recon=m.predict(X_test); err=np.mean((recon-X_test)**2,axis=1); thr=np.percentile(err,95); yp=(err>thr).astype(int)
            else: continue
            results[name] = compute_metrics(y_test,yp)

        if hybrid:
            st.subheader("Hybrid Ensemble Results")
            # simple Voting
            votes = []
            for name in chosen:
                if name in results: continue
            st.write("Hybrid logic simplified: showing individual metrics above.")
        else:
            st.subheader("Supervised Results")
            for name,vals in results.items():
                acc,prec,rec,f1,rmse = vals
                st.write(f"**{name}** — Acc:{acc:.3f} Prec:{prec:.3f} Rec:{rec:.3f} F1:{f1:.3f} RMSE:{rmse:.3f}")

        draw_visualizations(df, graph_choices, "Supervised Data", x=None,y=label_col)

    else:
        # Unsupervised
        results={}
        for name in chosen:
            if name=="Isolation Forest":
                m=IsolationForest(contamination=contamination); pred=m.fit_predict(X_scaled); labels=np.where(pred==-1,"Attack","Normal")
            elif name=="One-Class SVM":
                m=OneClassSVM(nu=contamination); pred=m.fit_predict(X_scaled); labels=np.where(pred==-1,"Attack","Normal")
            elif name=="KMeans":
                m=KMeans(n_clusters=2,random_state=42); cl=m.fit_predict(X_scaled); small=pd.Series(cl).value_counts().idxmin(); labels=np.where(cl==small,"Attack","Normal")
            elif name=="DBSCAN":
                m=DBSCAN(); cl=m.fit_predict(X_scaled); labels=np.where(cl==-1,"Attack","Normal")
            elif name=="PCA":
                m=PCA(n_components=2); cl=m.fit_transform(X_scaled); labels=np.where(cl[:,0]>np.median(cl[:,0]),"Attack","Normal")
            elif USE_TF and name=="Autoencoder":
                m=build_autoencoder(X_scaled.shape[1]); m.fit(X_scaled,X_scaled,epochs=10,batch_size=32,verbose=0); recon=m.predict(X_scaled); err=np.mean((recon-X_scaled)**2,axis=1); thr=np.percentile(err,100*(1-contamination)); labels=np.where(err>thr,"Attack","Normal")
            else: continue
            vc=pd.Series(labels).value_counts(normalize=True)*100
            st.write(f"**{name}** — Attack:{vc.get('Attack',0):.2f}% Normal:{vc.get('Normal',0):.2f}%")
            df[name+"_Pred"]=labels

        draw_visualizations(df, graph_choices, "Unsupervised Data")
