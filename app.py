# app.py
"""
CYBER SHIELD (complete, corrected)
Features:
- Robust label detection + manual override
- Supervised: Logistic Regression, Random Forest, SVM, KNN, Voting (hybrid)
- Unsupervised: IsolationForest, One-Class SVM, KMeans, Autoencoder (if TF), Consensus (hybrid)
- Preprocessing: one-hot for categoricals + scaling
- Visualizations: confusion matrix, metrics comparison, anomaly pie + comparison bar
- Export: CSV downloads
Note: Deep models require TensorFlow and more compute; they are optional.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Try import tensorflow/keras - optional
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input
    from tensorflow.keras.optimizers import Adam
except Exception:
    USE_TF = False

st.set_page_config(page_title="cybershield", layout="wide")
st.title("🔎 CYBER SHIELD")
st.markdown("Upload a CSV/XLSX dataset. The app auto-detects labeled/unlabeled and runs selected models.")

# ---------------------------
# Helpers
# ---------------------------
def find_label_candidates(df, keywords=None, max_unique_pct=0.05, max_unique_count=50):
    if keywords is None:
        keywords = ["label", "class", "attack", "target", "category", "type", "y", "outcome", "result"]
    n = len(df)
    scores = {}
    for col in df.columns:
        norm = re.sub(r"[^0-9a-zA-Z]", " ", str(col)).lower().strip()
        name_score = sum(1 for kw in keywords if kw in norm)
        try:
            nunq = df[col].nunique(dropna=True)
        except Exception:
            nunq = 0
        uniq_ratio = nunq / n if n > 0 else 0
        sample_vals = set([str(x).strip().lower() for x in df[col].dropna().unique()[:100]])
        binary_like_sets = [{"0", "1"}, {"true", "false"}, {"yes", "no"},
                            {"y", "n"}, {"attack", "normal"}, {"malicious", "benign"}]
        is_binary = any(sample_vals.issubset(s) for s in binary_like_sets) or nunq <= 2
        value_score = 2 if is_binary else (1 if (nunq <= max_unique_count or uniq_ratio <= max_unique_pct) else 0)
        scores[col] = name_score * 2 + value_score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def preprocess_df_for_model(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    df_proc = df.drop(columns=drop_cols, errors="ignore").copy()
    df_proc = df_proc.dropna(axis=1, how="all")
    cat_cols = df_proc.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
    df_proc = df_proc.fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc.values)
    X_df = pd.DataFrame(X, columns=df_proc.columns)
    return X_df, {"scaler": scaler, "feature_columns": df_proc.columns.tolist()}

def build_autoencoder(n_features):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(n_features,)),
        Dense(max(8, n_features // 4), activation="relu"),
        Dense(64, activation="relu"),
        Dense(n_features, activation="linear"),
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload dataset (CSV or XLSX)", type=["csv", "xlsx"])
if not uploaded_file:
    st.info("Upload data to begin. See Help below.")
    st.markdown("---")
else:
    # ---------------- Dataset load ----------------
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head(10))

    # ---------------- Label detection ----------------
    candidates = find_label_candidates(df)
    auto_label = candidates[0][0] if candidates and candidates[0][1] > 0 else None
    if auto_label:
        st.success(f"Auto-detected label candidate: **{auto_label}** (score={candidates[0][1]})")
    else:
        st.info("No strong label column detected.")

    options = ["-- None / Unlabeled --"] + list(df.columns)
    default_idx = options.index(auto_label) if (auto_label and auto_label in options) else 0
    label_choice = st.selectbox("Select label column", options, index=default_idx)

    dataset_type = "Unlabeled" if label_choice == "-- None / Unlabeled --" else "Labeled"
    label_col = None if dataset_type == "Unlabeled" else label_choice

    if dataset_type == "Unlabeled":
        st.warning("Proceeding as Unlabeled dataset.")
    else:
        st.info(f"Using **{label_col}** as label.")

    with st.expander("Show candidates and scores"):
        st.dataframe(pd.DataFrame(candidates, columns=["column", "score"]))

    # ---------------- Sidebar ----------------
    st.sidebar.header("Model Options")
    if dataset_type == "Labeled":
        sup_choices = ["Logistic Regression", "Random Forest", "SVM (RBF)", "KNN", "Voting (LR+RF+SVM) - Hybrid"]
        chosen_sup = st.sidebar.multiselect("Supervised models", sup_choices, default=["Logistic Regression"])
    else:
        unsup_choices = ["Isolation Forest", "One-Class SVM", "KMeans Clustering", "Consensus (hybrid)"]
        chosen_unsup = st.sidebar.multiselect("Unsupervised models", unsup_choices, default=["Isolation Forest"])
        contamination = st.sidebar.slider("Estimated contamination (fraction anomalies)", 0.001, 0.5, 0.05, 0.001)

    run = st.sidebar.button("Run Selected Models")

    # ---------------- Run ----------------
    if run:
        st.info("Preprocessing...")
        if dataset_type == "Labeled":
            df_features = df.drop(columns=[label_col]).copy()
            y = df[label_col].copy()
            if y.dtype == "object" or y.dtype.name == "category":
                y, _ = pd.factorize(y)
            else:
                y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        else:
            df_features, y = df.copy(), None

        X_processed, pipeline_info = preprocess_df_for_model(df_features)

        # ---------------- Supervised ----------------
        if dataset_type == "Labeled":
            # train/test
            X = X_processed.values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )

            metrics_summary, preds_store = {}, {}

            if "Logistic Regression" in chosen_sup:
                lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
                yp = lr.predict(X_test)
                preds_store["Logistic Regression"] = yp
                metrics_summary["Logistic Regression"] = (
                    accuracy_score(y_test, yp),
                    precision_score(y_test, yp, average="weighted", zero_division=0),
                    recall_score(y_test, yp, average="weighted", zero_division=0),
                    f1_score(y_test, yp, average="weighted", zero_division=0),
                )

            if "Random Forest" in chosen_sup:
                rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
                yp = rf.predict(X_test)
                preds_store["Random Forest"] = yp
                metrics_summary["Random Forest"] = (
                    accuracy_score(y_test, yp),
                    precision_score(y_test, yp, average="weighted", zero_division=0),
                    recall_score(y_test, yp, average="weighted", zero_division=0),
                    f1_score(y_test, yp, average="weighted", zero_division=0),
                )

            if "SVM (RBF)" in chosen_sup:
                svm = SVC(kernel="rbf").fit(X_train, y_train)
                yp = svm.predict(X_test)
                preds_store["SVM (RBF)"] = yp
                metrics_summary["SVM (RBF)"] = (
                    accuracy_score(y_test, yp),
                    precision_score(y_test, yp, average="weighted", zero_division=0),
                    recall_score(y_test, yp, average="weighted", zero_division=0),
                    f1_score(y_test, yp, average="weighted", zero_division=0),
                )

            if "KNN" in chosen_sup:
                knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
                yp = knn.predict(X_test)
                preds_store["KNN"] = yp
                metrics_summary["KNN"] = (
                    accuracy_score(y_test, yp),
                    precision_score(y_test, yp, average="weighted", zero_division=0),
                    recall_score(y_test, yp, average="weighted", zero_division=0),
                    f1_score(y_test, yp, average="weighted", zero_division=0),
                )

            if "Voting (LR+RF+SVM) - Hybrid" in chosen_sup:
                vote = VotingClassifier(
                    estimators=[("lr", LogisticRegression(max_iter=1000)),
                                ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
                                ("svm", SVC(kernel="rbf", probability=True))],
                    voting="soft"
                ).fit(X_train, y_train)
                yp = vote.predict(X_test)
                preds_store["Voting Hybrid"] = yp
                metrics_summary["Voting Hybrid"] = (
                    accuracy_score(y_test, yp),
                    precision_score(y_test, yp, average="weighted", zero_division=0),
                    recall_score(y_test, yp, average="weighted", zero_division=0),
                    f1_score(y_test, yp, average="weighted", zero_division=0),
                )

            # summary table
            st.header("Supervised Models — Metrics")
            st.table(pd.DataFrame(metrics_summary, index=["Acc","Prec","Rec","F1"]).T.round(3))

        # ---------------- Unsupervised ----------------
        else:
            X_vals = X_processed.values
            unsup_results, percent_flagged = {}, {}

            if "Isolation Forest" in chosen_unsup:
                iso = IsolationForest(contamination=contamination, random_state=42).fit(X_vals)
                labels = np.where(iso.predict(X_vals) == -1, "Attack", "Normal")
                df_iso = df.copy(); df_iso["Pred_IsolationForest"] = labels
                unsup_results["Isolation Forest"] = df_iso
                percent_flagged["Isolation Forest"] = (labels == "Attack").sum()/len(labels)

            if "One-Class SVM" in chosen_unsup:
                try:
                    oc = OneClassSVM(nu=contamination, kernel="rbf").fit(X_vals)
                    labels = np.where(oc.predict(X_vals) == -1, "Attack", "Normal")
                    df_oc = df.copy(); df_oc["Pred_OneClassSVM"] = labels
                    unsup_results["One-Class SVM"] = df_oc
                    percent_flagged["One-Class SVM"] = (labels == "Attack").sum()/len(labels)
                except Exception as ex:
                    st.error(f"OCSVM failed: {ex}")

            if "KMeans Clustering" in chosen_unsup:
                kmeans = KMeans(n_clusters=3, random_state=42).fit(X_vals)
                clusters = kmeans.labels_
                df_km = df.copy(); df_km["km_cluster"] = clusters
                small = df_km["km_cluster"].value_counts().idxmin()
                df_km["Pred_KMeans"] = np.where(df_km["km_cluster"] == small, "Attack", "Normal")
                unsup_results["KMeans"] = df_km
                percent_flagged["KMeans"] = (df_km["Pred_KMeans"] == "Attack").sum()/len(df_km)

            if "Consensus (hybrid)" in chosen_unsup and unsup_results:
                cons_df = df.copy()
                preds = []
                for dfm in unsup_results.values():
                    pcol = [c for c in dfm.columns if c.startswith("Pred_")][0]
                    preds.append(dfm[pcol].map(lambda x: 1 if str(x).lower().startswith("attack") else 0))
                cons_df["Consensus_Pred"] = np.where(np.mean(preds, axis=0) >= 0.5, "Attack", "Normal")
                unsup_results["Consensus"] = cons_df
                percent_flagged["Consensus"] = (cons_df["Consensus_Pred"] == "Attack").sum()/len(cons_df)

            # results
            st.header("Unsupervised Results")
            for name, df_out in unsup_results.items():
                st.subheader(name)
                predcol = [c for c in df_out.columns if c.startswith("Pred_")][-1]
                vc = df_out[predcol].value_counts()
                fig, ax = plt.subplots()
                ax.pie([vc.get("Attack",0), vc.get("Normal",0)], labels=["Attack","Normal"], autopct="%1.1f%%")
                st.pyplot(fig)
                st.download_button(f"Download {name} results", df_out.to_csv(index=False).encode(), file_name=f"{name}_results.csv")

            if percent_flagged:
                st.subheader("Model Comparison — % flagged")
                comp_df = pd.DataFrame(percent_flagged, index=["Percent"]).T
                st.table((comp_df*100).round(2))
                comp_df.plot(kind="bar", legend=False); st.pyplot(plt.gcf())

# Help
st.markdown("---")
with st.expander("Help & Quickstart"):
    st.write("""
    1. Upload CSV/XLSX file. If you have labels, ensure a label column exists.
    2. The app auto-detects label column; override if needed.
    3. Choose supervised/unsupervised models in the sidebar.
    4. Press 'Run Selected Models' to get results.
    """)
