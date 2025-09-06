# app.py
"""
CYBER SHIELD — Mini Project Web App
-----------------------------------
Features:
- Auto detect Supervised (Labeled) or Unsupervised (Unlabeled) dataset
- Automatic preprocessing: missing values, encoding categoricals, scaling
- Supervised models: Logistic Regression, Random Forest, SVM, KNN, Decision Tree, Naive Bayes
- Unsupervised models: Isolation Forest, One-Class SVM, KMeans
- Hybrid option: user selects multiple models → majority voting
- Metrics: Accuracy, Precision, Recall, F1, RMSE (for supervised)
- Attack/Normal counts + anomaly visualization (for unsupervised)
- Visualization: heatmap, bar chart, pie chart, line plot, scatter plot, histogram
- CSV download of predictions
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

st.set_page_config(page_title="CYBER SHIELD", layout="wide")
st.title("🔎 CYBER SHIELD — Intrusion Detection Mini Project")

st.markdown("""
Upload your dataset (CSV/XLSX).  
The app will **auto-detect** whether it is Supervised (Labeled) or Unsupervised (Unlabeled),  
clean & preprocess it, then train models and show results with visualizations.
""")

# ---------------------------
# Helper: detect label column
# ---------------------------
def find_label_candidates(df, keywords=None):
    if keywords is None:
        keywords = ["label", "class", "attack", "target", "category", "type", "y", "outcome", "result"]
    scores = {}
    for col in df.columns:
        norm = re.sub(r'[^0-9a-zA-Z]', ' ', str(col)).lower().strip()
        score = sum(1 for kw in keywords if kw in norm)
        scores[col] = score
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates

# ---------------------------
# Preprocess function
# ---------------------------
def preprocess_df_for_model(df, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    df_proc = df.copy()
    # drop unwanted cols
    for c in drop_cols:
        if c in df_proc.columns:
            df_proc = df_proc.drop(columns=[c])
    # handle missing
    for col in df_proc.columns:
        if df_proc[col].dtype in ["float64", "int64"]:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        else:
            df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])
    # encode categoricals
    cat_cols = df_proc.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc.values)
    X_df = pd.DataFrame(X, columns=df_proc.columns)
    return X_df, {"scaler": scaler, "feature_columns": df_proc.columns.tolist()}

# ---------------------------
# Upload dataset
# ---------------------------
uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # auto detect label
    candidates = find_label_candidates(df)
    label_col = None
    if candidates and candidates[0][1] > 0:
        label_col = candidates[0][0]

    # decide supervised/unsupervised
    if label_col:
        dataset_type = "Supervised"
        st.success(f"✅ Detected as Supervised dataset (label column: **{label_col}**) ")
    else:
        dataset_type = "Unsupervised"
        st.warning("⚠️ No label column detected. Proceeding as Unsupervised dataset.")

    # Sidebar controls
    st.sidebar.header("⚙️ Model Options")
    if dataset_type == "Supervised":
        sup_models = ["Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree", "Naive Bayes"]
        chosen_sup = st.sidebar.multiselect("Select supervised models", sup_models, default=["Logistic Regression"])
        hybrid = st.sidebar.checkbox("Enable Hybrid (combine multiple models)")
    else:
        unsup_models = ["Isolation Forest", "One-Class SVM", "KMeans"]
        chosen_unsup = st.sidebar.multiselect("Select unsupervised models", unsup_models, default=["Isolation Forest"])
        contamination = st.sidebar.slider("Estimated contamination (attack %)", 0.01, 0.5, 0.1, 0.01)

    # Visualization choice
    viz_options = ["Heatmap", "Bar Chart", "Pie Chart", "Line Plot", "Scatter Plot", "Histogram"]
    chosen_viz = st.sidebar.selectbox("📈 Choose visualization type", viz_options)

    st.sidebar.markdown("---")
    run = st.sidebar.button("🚀 Run Models")

    if run:
        st.info("🔄 Preprocessing data...")
        if dataset_type == "Supervised":
            X = df.drop(columns=[label_col]).copy()
            y = df[label_col].copy()
            if y.dtype == "object" or y.dtype.name == "category":
                y, uniques = pd.factorize(y)
            else:
                y = pd.to_numeric(y, errors="coerce")
                y = y.fillna(0).replace([np.inf, -np.inf], 0)
                y = y.astype(int)
        else:
            X = df.copy()
            y = None

        X_processed, pipe_info = preprocess_df_for_model(X)

        # ---------------------------
        # SUPERVISED
        # ---------------------------
        if dataset_type == "Supervised":
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42, stratify=y)
            metrics_summary = {}
            preds_store = {}

            def evaluate_model(name, model):
                model.fit(X_train, y_train)
                yp = model.predict(X_test)
                preds_store[name] = yp
                acc = accuracy_score(y_test, yp)
                prec = precision_score(y_test, yp, average="weighted", zero_division=0)
                rec = recall_score(y_test, yp, average="weighted", zero_division=0)
                f1 = f1_score(y_test, yp, average="weighted", zero_division=0)
                rmse = np.sqrt(mean_squared_error(y_test, yp))
                metrics_summary[name] = (acc, prec, rec, f1, rmse)

            if "Logistic Regression" in chosen_sup:
                evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000))
            if "Random Forest" in chosen_sup:
                evaluate_model("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))
            if "SVM" in chosen_sup:
                evaluate_model("SVM", SVC(kernel="rbf"))
            if "KNN" in chosen_sup:
                evaluate_model("KNN", KNeighborsClassifier(n_neighbors=5))
            if "Decision Tree" in chosen_sup:
                evaluate_model("Decision Tree", DecisionTreeClassifier(random_state=42))
            if "Naive Bayes" in chosen_sup:
                evaluate_model("Naive Bayes", GaussianNB())

            # Hybrid voting
            if hybrid and len(chosen_sup) > 1:
                estimators = []
                if "Logistic Regression" in chosen_sup:
                    estimators.append(("lr", LogisticRegression(max_iter=1000)))
                if "Random Forest" in chosen_sup:
                    estimators.append(("rf", RandomForestClassifier(n_estimators=100, random_state=42)))
                if "SVM" in chosen_sup:
                    estimators.append(("svm", SVC(kernel="rbf", probability=True)))
                if "KNN" in chosen_sup:
                    estimators.append(("knn", KNeighborsClassifier(n_neighbors=5)))
                if "Decision Tree" in chosen_sup:
                    estimators.append(("dt", DecisionTreeClassifier(random_state=42)))
                if "Naive Bayes" in chosen_sup:
                    estimators.append(("nb", GaussianNB()))
                vote = VotingClassifier(estimators=estimators, voting="hard")
                evaluate_model("Hybrid Voting", vote)

            # Show results
            st.subheader("📊 Supervised Results — Metrics")
            metrics_df = pd.DataFrame.from_dict(
                {m: {"Accuracy": v[0], "Precision": v[1], "Recall": v[2], "F1": v[3], "RMSE": v[4]} for m, v in metrics_summary.items()},
                orient="index"
            )
            st.table(metrics_df.round(3))

            # Confusion matrix + visualization
            for name, yp in preds_store.items():
                st.markdown(f"### {name} — Confusion Matrix")
                cm = confusion_matrix(y_test, yp)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)

            # Visualization option
            st.subheader("📈 Visualization")
            if chosen_viz == "Bar Chart":
                metrics_df["Accuracy"].plot(kind="bar")
                plt.ylabel("Accuracy")
                st.pyplot(plt)
            elif chosen_viz == "Pie Chart":
                sizes = metrics_df["Accuracy"]
                plt.pie(sizes, labels=metrics_df.index, autopct="%1.1f%%")
                st.pyplot(plt)
            elif chosen_viz == "Line Plot":
                metrics_df["Accuracy"].plot(kind="line", marker="o")
                st.pyplot(plt)
            elif chosen_viz == "Scatter Plot":
                plt.scatter(metrics_df.index, metrics_df["Accuracy"])
                plt.ylabel("Accuracy")
                st.pyplot(plt)
            elif chosen_viz == "Histogram":
                plt.hist(metrics_df["Accuracy"], bins=5)
                plt.xlabel("Accuracy")
                st.pyplot(plt)

        # ---------------------------
        # UNSUPERVISED
        # ---------------------------
        else:
            results_unsup = {}
            percent_flagged = {}

            if "Isolation Forest" in chosen_unsup:
                iso = IsolationForest(contamination=contamination, random_state=42)
                pred = iso.fit_predict(X_processed)
                labels = np.where(pred == -1, "Attack", "Normal")
                df_out = df.copy()
                df_out["Prediction"] = labels
                results_unsup["Isolation Forest"] = df_out
                percent_flagged["Isolation Forest"] = (labels == "Attack").sum() / len(labels)

            if "One-Class SVM" in chosen_unsup:
                oc = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
                pred = oc.fit_predict(X_processed)
                labels = np.where(pred == -1, "Attack", "Normal")
                df_out = df.copy()
                df_out["Prediction"] = labels
                results_unsup["One-Class SVM"] = df_out
                percent_flagged["One-Class SVM"] = (labels == "Attack").sum() / len(labels)

            if "KMeans" in chosen_unsup:
                km = KMeans(n_clusters=2, random_state=42)
                clusters = km.fit_predict(X_processed)
                small = pd.Series(clusters).value_counts().idxmin()
                labels = np.where(clusters == small, "Attack", "Normal")
                df_out = df.copy()
                df_out["Prediction"] = labels
                results_unsup["KMeans"] = df_out
                percent_flagged["KMeans"] = (labels == "Attack").sum() / len(labels)

            st.subheader("📊 Unsupervised Results")
            for name, df_out in results_unsup.items():
                st.write(f"### {name}")
                st.dataframe(df_out.head())
                vc = df_out["Prediction"].value_counts()
                attacks = vc.get("Attack", 0)
                normals = vc.get("Normal", 0)
                st.success(f"🔴 Attacks: {attacks}, 🟢 Normal: {normals}")

                # Visualization
                if chosen_viz == "Pie Chart":
                    plt.pie([attacks, normals], labels=["Attack", "Normal"], autopct="%1.1f%%", colors=["red", "green"])
                    st.pyplot(plt)
                elif chosen_viz == "Bar Chart":
                    plt.bar(["Attack", "Normal"], [attacks, normals], color=["red", "green"])
                    st.pyplot(plt)
                elif chosen_viz == "Line Plot":
                    plt.plot(["Attack", "Normal"], [attacks, normals], marker="o")
                    st.pyplot(plt)
                elif chosen_viz == "Scatter Plot":
                    plt.scatter(["Attack", "Normal"], [attacks, normals], color=["red", "green"])
                    st.pyplot(plt)
                elif chosen_viz == "Histogram":
                    plt.hist([attacks, normals], bins=2)
                    st.pyplot(plt)
                elif chosen_viz == "Heatmap":
                    cm = np.array([[attacks, normals]])
                    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
                    st.pyplot(plt)

            if percent_flagged:
                st.subheader("Comparison — % flagged as Attack")
                comp_df = pd.DataFrame.from_dict({k: v for k, v in percent_flagged.items()}, orient="index", columns=["% Attack"])
                st.table((comp_df * 100).round(2))

# ---------------------------
# HELP
# ---------------------------
st.markdown("---")
with st.expander("📘 Help & Quickstart"):
    st.write("""
    **Steps to use CYBER SHIELD:**
    1. Upload dataset (CSV/XLSX).
    2. App auto-detects if dataset is Supervised or Unsupervised.
    3. Choose models from sidebar.
    4. Choose visualization format.
    5. Press "Run Models".
    6. Download predictions if needed.

    - Supervised → shows Accuracy, Precision, Recall, F1, RMSE, confusion matrices.
    - Unsupervised → shows Attacks/Normal counts, anomaly detection.
    - Hybrid → majority voting across selected models.
    """)
