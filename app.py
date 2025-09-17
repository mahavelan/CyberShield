import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import joblib

# ----------------- CONFIG -----------------
MAX_TRAIN_ROWS = 20000
HASHER_FEATURES = 32
PCA_VARIANCE = 0.95
LOW_CARDINALITY_THRESHOLD = 50
HIGH_CARDINALITY_THRESHOLD = 200
TOP_K_FOR_CHART = 15

# ----------------- HELPERS -----------------
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

def extract_timestamp_features(df, ts_col="Timestamp"):
    if ts_col in df.columns:
        try:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            df["ts_hour"] = ts.dt.hour.fillna(-1).astype(int)
            df["ts_day"] = ts.dt.day.fillna(-1).astype(int)
        except Exception:
            df["ts_hour"] = -1
            df["ts_day"] = -1
    return df

# ----------------- PREPROCESS -----------------
def harmonize_and_preprocess(dfs, label_col="Label", sample_limit=MAX_TRAIN_ROWS, perform_pca=True):
    df_all = pd.concat(dfs, ignore_index=True)
    drop_cols = ["Flow ID", "Source IP", "Destination IP"]
    df_all = df_all.drop(columns=[c for c in drop_cols if c in df_all.columns], errors="ignore")
    df_all = extract_timestamp_features(df_all, ts_col="Timestamp")
    if "Timestamp" in df_all.columns:
        df_all = df_all.drop(columns=["Timestamp"])
    y = df_all[label_col] if label_col in df_all.columns else None
    X = df_all.drop(columns=[label_col], errors="ignore")

    dtype_info = pd.DataFrame({'Column': X.columns, 'Type': [str(X[c].dtype) for c in X.columns]})
    X = X.replace([np.inf, -np.inf], np.nan)

    object_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    low_card_cols, med_card_cols, high_card_cols = [], [], []
    for col in object_cols:
        nunq = X[col].nunique(dropna=True)
        if nunq <= LOW_CARDINALITY_THRESHOLD:
            low_card_cols.append(col)
        elif nunq <= HIGH_CARDINALITY_THRESHOLD:
            med_card_cols.append(col)
        else:
            high_card_cols.append(col)

    label_encoders = {}
    for col in med_card_cols:
        le = LabelEncoder()
        X[col] = X[col].astype(str).fillna("nan")
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if low_card_cols:
        transformers.append(("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), low_card_cols))

    column_transformer = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    X_proc = column_transformer.fit_transform(X)

    if med_card_cols:
        X_med = X[med_card_cols].to_numpy(dtype=float)
        X_proc = np.hstack([X_proc, X_med]) if X_proc.size else X_med
    if high_card_cols:
        rows = X[high_card_cols].fillna("nan").astype(str).apply(
            lambda r: [f"{c}={r[c]}" for c in high_card_cols], axis=1
        )
        hasher = FeatureHasher(n_features=HASHER_FEATURES, input_type='string')
        X_hash = hasher.transform(rows).toarray()
        X_proc = np.hstack([X_proc, X_hash]) if X_proc.size else X_hash

    if sample_limit and X_proc.shape[0] > sample_limit:
        if y is not None:
            stratify = y if len(pd.Series(y).unique()) > 1 else None
            X_proc, _, y, _ = train_test_split(X_proc, y, train_size=sample_limit, stratify=stratify, random_state=42)
        else:
            X_proc, _ = train_test_split(X_proc, train_size=sample_limit, random_state=42)

    pca = None
    if perform_pca and X_proc.shape[1] > 50:
        try:
            pca = PCA(n_components=PCA_VARIANCE, svd_solver='full')
            X_final = pca.fit_transform(X_proc)
        except Exception:
            pca = PCA(n_components=30)
            X_final = pca.fit_transform(X_proc)
    else:
        X_final = X_proc

    return X_final, (y.values if y is not None else None), {"pca": pca, "dtype_info": dtype_info, "df_all": df_all}

# ----------------- SUPERVISED -----------------
def train_supervised(X, y, selected_models):
    available_models = {
        "SGDClassifier": SGDClassifier(max_iter=1000, tol=1e-3),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier()
    }
    results = {}
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    for name in selected_models:
        clf = available_models[name]
        try:
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            results[name] = {
                "metrics": compute_metrics(y_te, pred),
                "test_true": y_te,
                "test_pred": pred
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

# ----------------- UNSUPERVISED -----------------
def train_unsupervised(X):
    res = {}
    try:
        iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        iso.fit(X)
        res['isolation_forest'] = iso
    except Exception as e:
        res['isolation_forest_error'] = str(e)
    try:
        sample_size = min(30000, X.shape[0])
        Xc = X[np.random.choice(X.shape[0], sample_size, replace=False)]
        db = DBSCAN(eps=0.5, min_samples=5).fit(Xc)
        res['dbscan'] = db
    except Exception as e:
        res['dbscan_error'] = str(e)
    return res

# ----------------- STREAMLIT APP -----------------
st.title("CyberShield: Supervised & Unsupervised Intrusion Detection")

uploaded_files = st.file_uploader("Upload CSV datasets", type=["csv"], accept_multiple_files=True)

selected_models = st.multiselect("Select supervised models to run", ["SGDClassifier", "RandomForest", "GradientBoosting"], default=["RandomForest"])
selected_graph = st.selectbox("Select graph type for visualization", ["Bar", "Pie", "Scatter", "PCA"])
run_btn = st.button("Run Models")

if uploaded_files and run_btn:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    X, y, artifacts = harmonize_and_preprocess(dfs, label_col="Label", sample_limit=MAX_TRAIN_ROWS, perform_pca=True)

    st.write("Processed shape:", X.shape)
    st.subheader("Data Type Recognition")
    st.dataframe(artifacts["dtype_info"])

    # Column visualization
    df_all = artifacts["df_all"]
    vis_col = st.selectbox("Select column for visualization", df_all.columns)
    fig, ax = plt.subplots()
    chart_data = get_chart_data(df_all[vis_col])
    if selected_graph == "Bar":
        chart_data.plot(kind="bar", ax=ax)
    elif selected_graph == "Pie":
        chart_data.plot(kind="pie", ax=ax, autopct='%1.1f%%')
    elif selected_graph == "Scatter":
        num_cols = df_all.select_dtypes(include=np.number).columns.tolist()
        if vis_col in num_cols and len(num_cols) > 1:
            other = [c for c in num_cols if c != vis_col][0]
            ax.scatter(df_all[vis_col], df_all[other], alpha=0.5)
    elif selected_graph == "PCA" and artifacts["pca"] is not None:
        pc_df = pd.DataFrame(artifacts["pca"].transform(df_all.drop(columns=["Label"], errors="ignore")), columns=["PC1", "PC2"])
        ax.scatter(pc_df["PC1"], pc_df["PC2"], alpha=0.5)
    st.pyplot(fig)

    # --- SUPERVISED ---
    if y is not None:
        st.subheader("Supervised Results")
        results = train_supervised(X, y, selected_models)
        for name, res in results.items():
            if "metrics" in res:
                st.write(f"### {name} Metrics")
                st.json(res["metrics"])
            else:
                st.error(f"{name} failed: {res.get('error')}")
    else:
        # --- UNSUPERVISED ---
        st.subheader("Unsupervised Results")
        unsup = train_unsupervised(X)
        st.write({k: str(v) for k,v in unsup.items()})
