# app.py
"""
CYBER SHIELD (final, extended)
- Auto-detect dataset type (supervised vs unsupervised)
- Supervised: Logistic Regression, Decision Tree, KNN, Random Forest, SVM, Naive Bayes, Gradient Boosting, AdaBoost, MLP (Neural Net)
- Unsupervised: Isolation Forest, One-Class SVM, KMeans, Autoencoder (if TF), Consensus, DBSCAN, PCA anomaly
- Hybrid: user can choose multiple models for ensemble
- Preprocessing: handle missing values, categorical encoding, scaling
- Evaluation: Accuracy, Precision, Recall, F1, RMSE (supervised)
- Visualization: heatmap, bar, pie, line, scatter, histogram
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, IsolationForest, VotingClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# TensorFlow for autoencoder (optional)
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
except Exception:
    USE_TF = False

# ---------------------------
# Helpers
# ---------------------------
def find_label_candidates(df, keywords=None):
    if keywords is None:
        keywords = ["label", "class", "attack", "target", "category", "type", "y", "outcome", "result"]
    scores = {}
    for col in df.columns:
        norm = re.sub(r"[^0-9a-zA-Z]", " ", str(col)).lower()
        score = sum(1 for kw in keywords if kw in norm)
        scores[col] = score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

def preprocess_features(df, drop_cols=None):
    if drop_cols is None: drop_cols = []
    df_proc = df.copy()
    for c in drop_cols:
        if c in df_proc.columns:
            df_proc = df_proc.drop(columns=[c])
    cat_cols = df_proc.select_dtypes(include=["object","category"]).columns
    if len(cat_cols) > 0:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
    df_proc = df_proc.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_proc)
    return pd.DataFrame(X_scaled, columns=df_proc.columns)

def build_autoencoder(n_features):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(n_features,)),
        Dense(max(8, n_features//4), activation="relu"),
        Dense(64, activation="relu"),
        Dense(n_features, activation="linear")
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="Cyber Shield", layout="wide")
st.title("🔎 CYBER SHIELD - Intrusion Detection Web App")

uploaded_file = st.file_uploader("📂 Upload dataset (CSV/XLSX)", type=["csv","xlsx"])
if not uploaded_file:
    st.info("Upload dataset to begin.")
    st.stop()

# Load data
try:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Detect label column
candidates = find_label_candidates(df)
auto_label = candidates[0][0] if candidates and candidates[0][1] > 0 else None
if auto_label and auto_label in df.columns:
    label_col = auto_label
    dataset_type = "Supervised"
    st.success(f"Auto-detected label column: **{label_col}**")
else:
    label_col = None
    dataset_type = "Unsupervised"
    st.warning("No label column detected. Proceeding as Unsupervised dataset.")

# ---------------------------
# Sidebar model selection
# ---------------------------
st.sidebar.header("⚙️ Model Options")
viz_options = ["Heatmap","Bar Chart","Pie Chart","Line Chart","Scatter Plot","Histogram"]

if dataset_type == "Supervised":
    sup_models = ["Logistic Regression","Decision Tree","KNN","Random Forest","SVM","Naive Bayes","Gradient Boosting","AdaBoost","MLP (Neural Net)"]
    chosen_models = st.sidebar.multiselect("Choose Supervised Models", sup_models, default=["Logistic Regression"])
    hybrid = st.sidebar.checkbox("Use Hybrid (ensemble of selected models)")
    chosen_viz = st.sidebar.selectbox("Choose Visualization", viz_options)
else:
    unsup_models = ["Isolation Forest","One-Class SVM","KMeans","DBSCAN","PCA Anomaly"]
    if USE_TF: unsup_models += ["Autoencoder"]
    unsup_models += ["Consensus (hybrid)"]
    chosen_models = st.sidebar.multiselect("Choose Unsupervised Models", unsup_models, default=["Isolation Forest"])
    chosen_viz = st.sidebar.selectbox("Choose Visualization", viz_options)
    contamination = st.sidebar.slider("Contamination (expected anomaly %)", 0.01,0.5,0.1,0.01)

run = st.sidebar.button("🚀 Run Models")

# ---------------------------
# Run
# ---------------------------
if run:
    if dataset_type == "Supervised":
        X = df.drop(columns=[label_col])
        y = df[label_col]

        if y.dtype == "object" or y.dtype.name == "category":
            y, uniques = pd.factorize(y)

        X_proc = preprocess_features(X)
        X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.3, random_state=42, stratify=y)

        results = {}
        metrics = {}
        models_dict = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC(probability=True),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "MLP (Neural Net)": MLPClassifier(max_iter=500)
        }

        estimators = []
        for name in chosen_models:
            st.info(f"Training {name}...")
            model = models_dict[name]
            model.fit(X_train, y_train)
            yp = model.predict(X_test)
            results[name] = yp
            estimators.append((name, model))
            metrics[name] = {
                "Accuracy": accuracy_score(y_test, yp),
                "Precision": precision_score(y_test, yp, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, yp, average="weighted", zero_division=0),
                "F1": f1_score(y_test, yp, average="weighted", zero_division=0),
                "RMSE": mean_squared_error(y_test, yp, squared=False)
            }

        if hybrid and len(estimators) > 1:
            st.info("Running Hybrid Ensemble...")
            vote = VotingClassifier(estimators=estimators, voting="soft")
            vote.fit(X_train,y_train)
            yp = vote.predict(X_test)
            results["Hybrid Ensemble"] = yp
            metrics["Hybrid Ensemble"] = {
                "Accuracy": accuracy_score(y_test, yp),
                "Precision": precision_score(y_test, yp, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, yp, average="weighted", zero_division=0),
                "F1": f1_score(y_test, yp, average="weighted", zero_division=0),
                "RMSE": mean_squared_error(y_test, yp, squared=False)
            }

        st.subheader("📈 Metrics Summary")
        st.dataframe(pd.DataFrame(metrics).T)

        # Visualization
        if chosen_viz == "Heatmap":
            for name, yp in results.items():
                cm = confusion_matrix(y_test, yp)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
        elif chosen_viz == "Bar Chart":
            dfm = pd.DataFrame(metrics).T
            dfm["Accuracy"].plot(kind="bar", legend=False)
            st.pyplot(plt.gcf())
        elif chosen_viz == "Pie Chart":
            cm = confusion_matrix(y_test, list(results.values())[0])
            vals = [cm[0,0]+cm[0,1], cm[1,0]+cm[1,1]]
            plt.pie(vals, labels=["Class0","Class1"], autopct="%1.1f%%")
            st.pyplot(plt.gcf())
        elif chosen_viz == "Line Chart":
            pd.DataFrame(metrics).T[["Accuracy","F1"]].plot(kind="line", marker="o")
            st.pyplot(plt.gcf())
        elif chosen_viz == "Scatter Plot":
            dfm = pd.DataFrame(metrics).T
            plt.scatter(dfm["Accuracy"], dfm["F1"])
            plt.xlabel("Accuracy"); plt.ylabel("F1")
            st.pyplot(plt.gcf())
        elif chosen_viz == "Histogram":
            pd.DataFrame(metrics).T["Accuracy"].plot(kind="hist", bins=5)
            st.pyplot(plt.gcf())

    # ---------------------------
    # Unsupervised
    # ---------------------------
    else:
        X_proc = preprocess_features(df)
        X_vals = X_proc.values
        unsup_results = {}
        percent_flagged = {}

        if "Isolation Forest" in chosen_models:
            iso = IsolationForest(contamination=contamination, random_state=42)
            pred = iso.fit_predict(X_vals)
            labels = np.where(pred==-1,"Attack","Normal")
            df_iso = df.copy(); df_iso["Pred_IsolationForest"] = labels
            unsup_results["Isolation Forest"] = df_iso
            percent_flagged["Isolation Forest"] = (labels=="Attack").sum()/len(labels)

        if "One-Class SVM" in chosen_models:
            oc = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
            pred = oc.fit_predict(X_vals)
            labels = np.where(pred==-1,"Attack","Normal")
            df_oc = df.copy(); df_oc["Pred_OneClassSVM"] = labels
            unsup_results["One-Class SVM"] = df_oc
            percent_flagged["One-Class SVM"] = (labels=="Attack").sum()/len(labels)

        if "KMeans" in chosen_models:
            km = KMeans(n_clusters=2, random_state=42)
            clusters = km.fit_predict(X_vals)
            small = pd.Series(clusters).value_counts().idxmin()
            labels = np.where(clusters==small,"Attack","Normal")
            df_km = df.copy(); df_km["Pred_KMeans"] = labels
            unsup_results["KMeans"] = df_km
            percent_flagged["KMeans"] = (labels=="Attack").sum()/len(labels)

        if "DBSCAN" in chosen_models:
            db = DBSCAN(eps=0.5, min_samples=5).fit(X_vals)
            labels = np.where(db.labels_==-1,"Attack","Normal")
            df_db = df.copy(); df_db["Pred_DBSCAN"] = labels
            unsup_results["DBSCAN"] = df_db
            percent_flagged["DBSCAN"] = (labels=="Attack").sum()/len(labels)

        if "PCA Anomaly" in chosen_models:
            pca = PCA(n_components=2)
            X_p = pca.fit_transform(X_vals)
            rec = pca.inverse_transform(X_p)
            err = np.mean((X_vals - rec)**2, axis=1)
            thr = np.percentile(err, 100*(1-contamination))
            labels = np.where(err>=thr,"Attack","Normal")
            df_pca = df.copy(); df_pca["Pred_PCA"] = labels
            unsup_results["PCA Anomaly"] = df_pca
            percent_flagged["PCA Anomaly"] = (labels=="Attack").sum()/len(labels)

        if USE_TF and "Autoencoder" in chosen_models:
            ae = build_autoencoder(X_vals.shape[1])
            ae.fit(X_vals,X_vals,epochs=10,batch_size=64,verbose=0)
            rec = ae.predict(X_vals)
            err = np.mean((X_vals - rec)**2, axis=1)
            thr = np.percentile(err, 100*(1-contamination))
            labels = np.where(err>=thr,"Attack","Normal")
            df_ae = df.copy(); df_ae["Pred_AE"] = labels
            unsup_results["Autoencoder"] = df_ae
            percent_flagged["Autoencoder"] = (labels=="Attack").sum()/len(labels)

        if "Consensus (hybrid)" in chosen_models and len(unsup_results)>0:
            cons = df.copy()
            flag_cols = []
            for name, dfm in unsup_results.items():
                col = [c for c in dfm.columns if c.startswith("Pred_")][0]
                cons[name] = dfm[col]
                flag_cols.append(name)
            flags = cons[flag_cols].applymap(lambda x: 1 if str(x).lower().startswith("attack") else 0)
            cons["Consensus_Pred"] = np.where(flags.mean(axis=1)>=0.5,"Attack","Normal")
            unsup_results["Consensus"] = cons
            percent_flagged["Consensus"] = (cons["Consensus_Pred"]=="Attack").sum()/len(cons)

        st.subheader("📊 Unsupervised Results")
        st.dataframe(pd.DataFrame(percent_flagged, index=["% Attacks"]).T*100)

        # Visualization
        if chosen_viz == "Pie Chart":
            for name, df_out in unsup_results.items():
                pcol = [c for c in df_out.columns if c.startswith("Pred_")][-1]
                vc = df_out[pcol].value_counts()
                plt.pie(vc, labels=vc.index, autopct="%1.1f%%", colors=["red","green"])
                plt.title(name)
                st.pyplot(plt.gcf())
        elif chosen_viz == "Bar Chart":
            pd.Series(percent_flagged).plot(kind="bar")
            plt.ylabel("% Attack")
            st.pyplot(plt.gcf())
        elif chosen_viz == "Line Chart":
            pd.Series(percent_flagged).plot(kind="line", marker="o")
            plt.ylabel("% Attack")
            st.pyplot(plt.gcf())
        elif chosen_viz == "Scatter Plot":
            vals = pd.Series(percent_flagged)
            plt.scatter(vals.index, vals.values)
            plt.ylabel("% Attack"); plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
        elif chosen_viz == "Histogram":
            vals = pd.Series(percent_flagged)
            vals.plot(kind="hist", bins=5)
            st.pyplot(plt.gcf())
        elif chosen_viz == "Heatmap":
            vals = pd.Series(percent_flagged).to_frame("Attack%")
            sns.heatmap(vals, annot=True, cmap="Reds")
            st.pyplot(plt.gcf())

# ---------------------------
# Help
# ---------------------------
st.markdown("---")
with st.expander("📘 Help & Guide"):
    st.write("""
    Steps to use:
    1. Upload CSV/XLSX dataset.
    2. App auto-detects if dataset is **Supervised (labeled)** or **Unsupervised (unlabeled)**.
    3. Select models and visualization type from sidebar.
    4. Press **Run Models**.
    5. View metrics (supervised) or anomaly counts (unsupervised).
    6. Download predictions if needed.
    """)
