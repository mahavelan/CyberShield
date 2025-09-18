import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
MAX_TRAIN_ROWS = 20000
HASHER_FEATURES = 32
PCA_VARIANCE = 0.95
LOW_CARDINALITY_THRESHOLD = 50
HIGH_CARDINALITY_THRESHOLD = 200
TOP_K_FOR_CHART = 15
MODEL_STORE_PATH = "models"

os.makedirs(MODEL_STORE_PATH, exist_ok=True)

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
            df["ts_month"] = ts.dt.month.fillna(-1).astype(int)
        except Exception:
            df["ts_hour"] = -1
            df["ts_day"] = -1
            df["ts_month"] = -1
    return df

def detect_label_column(df):
    # Expand auto-detection to things like "Label 0", "Label1", "Class", etc.
    candidates = [c for c in df.columns if c.strip().lower() in ("label", "target", "class", "attack")]
    # Also match columns *starting* with typical label keywords
    if not candidates:
        candidates = [c for c in df.columns if c.strip().lower().startswith(("label", "class", "target", "attack"))]
    # If still no candidates, try the last column as a fallback
    if not candidates and len(df.columns) > 0:
        candidates = [df.columns[-1]]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        return st.selectbox("Multiple possible label columns detected, please select the label column", candidates)
    else:
        return st.text_input("No label column detected automatically. Enter label column name if exists", value="")

def get_chart_data(col_data):
    vc = col_data.value_counts()
    if len(vc) > TOP_K_FOR_CHART:
        top_n = vc[:TOP_K_FOR_CHART]
        others_sum = vc[TOP_K_FOR_CHART:].sum()
        top_n["Others"] = others_sum
        return top_n
    return vc

def save_model(model_name, model, artifacts):
    joblib.dump((model, artifacts), os.path.join(MODEL_STORE_PATH, f"{model_name}.joblib"))

def load_model(model_name):
    path = os.path.join(MODEL_STORE_PATH, f"{model_name}.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None, None

def harmonize_and_preprocess(df, label_col, sample_limit=MAX_TRAIN_ROWS, perform_pca=True, artifacts=None):
    drop_cols = ["Flow ID", "Source IP", "Destination IP"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    df = extract_timestamp_features(df, ts_col="Timestamp")
    if "Timestamp" in df.columns:
        df.drop(columns=["Timestamp"], inplace=True)

    y = df[label_col] if label_col in df.columns else None
    X = df.drop(columns=[label_col], errors="ignore")

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

    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
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
            lambda r: [f"{c}={r[c]}" for c in high_card_cols], axis=1)
        hasher = FeatureHasher(n_features=HASHER_FEATURES, input_type='string')
        X_hash = hasher.transform(rows).toarray()
        X_proc = np.hstack([X_proc, X_hash]) if X_proc.size else X_hash

    if sample_limit and X_proc.shape[0] > sample_limit:
        stratify = y if (y is not None and len(pd.Series(y).unique()) > 1) else None
        if y is not None:
            X_proc, _, y, _ = train_test_split(
                X_proc, y, train_size=sample_limit, stratify=stratify, random_state=42)
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

    if artifacts is None:
        artifacts = {}
    artifacts.update({
        "pca": pca,
        "column_transformer": column_transformer,
        "label_encoders": label_encoders,
        "dtype_info": dtype_info,
        "df_all": df
    })
    return X_final, (y.values if y is not None else None), artifacts

def train_supervised(X, y, selected_models, previous_models=None, artifacts=None):
    available_models = {
        "SGDClassifier": SGDClassifier(max_iter=1000, tol=1e-3),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier()
    }
    results = {}
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    for name in selected_models:
        model, _ = (previous_models.get(name) if previous_models else (None, None))
        clf = available_models[name] if model is None else model
        try:
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
            results[name] = {
                "metrics": compute_metrics(y_te, pred),
                "model": clf,
                "test_true": y_te,
                "test_pred": pred
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

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

def map_attack_scenarios(y):
    if y is None:
        return None
    mapping = {
        "BENIGN": "No attack",
        "DDoS": "Volumetric attack",
        "DoS Hulk": "Application-layer DoS",
        "PortScan": "Reconnaissance",
        "FTP-Patator": "Brute force attack"
    }
    return [mapping.get(label, "Other/Unknown") for label in y]

st.title("CyberShield: AI-Driven Self-Learning Vulnerability Classification")
st.markdown("""
This app accepts datasets with varied structures and labels, dynamically learns to classify vulnerabilities per row, and supports ongoing model improvement over uploads.
""")

uploaded_files = st.file_uploader("Upload CSV datasets", type=["csv"], accept_multiple_files=True)

label_col = None
selected_models = []
selected_graph = None

if uploaded_files:
    try:
        first_df = pd.read_csv(io.StringIO(uploaded_files[0].getvalue().decode("utf-8")))
    except Exception as e:
        st.error(f"Failed to read first uploaded file: {e}")
        first_df = None

    if first_df is not None:
        st.subheader("Uploaded Dataset Preview (first file)")
        st.dataframe(first_df.head())
        label_col = detect_label_column(first_df)
        if label_col:
            st.success(f"Detected/selected label column: {label_col}")
        else:
            st.warning("No label column specified or detected, supervised training will be skipped.")
        st.subheader("Detected Data Types")
        dtype_info = pd.DataFrame({
            "Column": first_df.columns,
            "Type": [str(first_df[c].dtype) for c in first_df.columns]
        })
        st.dataframe(dtype_info)

    selected_models = st.multiselect(
        "Select supervised models to run", ["SGDClassifier", "RandomForest", "GradientBoosting"], default=["RandomForest"]
    )
    selected_graph = st.selectbox(
        "Select graph type for visualization",
        ["Bar", "Pie", "Scatter", "PCA"]
    )

run_btn = st.button("Run Models")

if uploaded_files and run_btn:
    dfs = []
    for f in uploaded_files:
        try:
            content = f.getvalue().decode("utf-8")
            df = pd.read_csv(io.StringIO(content))
            dfs.append(df)
        except Exception as e:
            st.error(f"Failed reading file {f.name}: {e}")
            continue

    X_full = []
    y_full = []
    artifacts = None
    previous_models = {}

    for model_name in selected_models:
        mod, arts = load_model(model_name)
        if mod is not None:
            previous_models[model_name] = (mod, arts)

    for df in dfs:
        if label_col and label_col not in df.columns:
            st.warning(f"Label column '{label_col}' not found in dataset {df}. Skipping supervised training for that file.")
        X, y, arts = harmonize_and_preprocess(df, label_col if label_col in df.columns else None, artifacts=artifacts)
        artifacts = arts
        if y is not None:
            X_full.append(X)
            y_full.extend(y)
        else:
            X_full.append(X)

    X_all = np.vstack(X_full)
    y_all = np.array(y_full) if y_full else None

    st.write(f"Processed total shape: {X_all.shape}")

    # Visualization
    st.subheader("Column Visualization")
    df_all = artifacts["df_all"] if artifacts else None
    if df_all is not None and selected_graph is not None:
        vis_col = st.selectbox("Select column for visualization", df_all.columns.tolist())
        if vis_col:
            col_data = df_all[vis_col]
            fig, ax = plt.subplots()
            chart_data = get_chart_data(col_data)

            if selected_graph == "Bar":
                chart_data.plot(kind="bar", ax=ax)
                ax.set_title(f"Bar Chart: ({vis_col}) Top {TOP_K_FOR_CHART} + Others")
                ax.set_ylabel("Count")
                ax.set_xlabel(vis_col)
            elif selected_graph == "Pie":
                chart_data.plot(kind="pie", ax=ax, autopct='%1.1f%%')
                ax.set_ylabel('')
                ax.set_title(f"Pie Chart: ({vis_col}) Top {TOP_K_FOR_CHART} + Others")
            elif selected_graph == "Scatter":
                num_cols = df_all.select_dtypes(include=np.number).columns.tolist()
                other_num = [c for c in num_cols if c != vis_col]
                if vis_col in num_cols and other_num:
                    ax.scatter(df_all[vis_col], df_all[other_num[0]], alpha=0.5)
                    ax.set_xlabel(vis_col)
                    ax.set_ylabel(other_num[0])
                    ax.set_title(f"Scatter: {vis_col} vs {other_num[0]}")
            elif selected_graph == "PCA":
                if artifacts and artifacts.get("pca") is not None:
                    pc_df = pd.DataFrame(artifacts["pca"].transform(df_all.drop(columns=[label_col], errors="ignore")), columns=[f"PC{i+1}" for i in range(artifacts["pca"].n_components_)])
                    if label_col in df_all.columns:
                        ax.scatter(pc_df["PC1"], pc_df["PC2"], c=pd.factorize(df_all[label_col])[0], cmap='tab10', alpha=0.5)
                    else:
                        ax.scatter(pc_df["PC1"], pc_df["PC2"], alpha=0.5)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_title("PCA Plot")
            st.pyplot(fig)
            st.write(f"Showing {selected_graph} for: {vis_col} (summary for categorical data)")

    # Supervised
    if y_all is not None:
        st.subheader("Supervised Results")
        st.markdown("Models train on labeled data and classify each row into vulnerability classes.")
        st.write("Label distribution:", pd.Series(y_all).value_counts().to_dict())
        results = train_supervised(X_all, y_all, selected_models, previous_models=previous_models, artifacts=artifacts)
        for name, res in results.items():
            if "metrics" in res:
                st.subheader(f"{name} Results")
                st.json(res["metrics"])
                st.write(f"Classified test set for {name}:")
                pred_df = pd.DataFrame({
                    'True_Label': res["test_true"],
                    'Predicted_Label': res["test_pred"]
                })
                st.dataframe(pred_df.head(50))
                csv = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(label=f"Download {name} Predictions as CSV", data=csv, file_name=f'{name}_test_predictions.csv', mime='text/csv')
            else:
                st.error(f"{name} failed: {res.get('error')}")
        best_name = max([n for n in results if "metrics" in results[n]], key=lambda nm: results[n]["metrics"]["f1"], default=None)
        if best_name:
            save_model(best_name, results[best_name]['model'], artifacts)
            st.success(f"Best model saved/updated: {best_name}")
        scenarios = map_attack_scenarios(y_all)
        if scenarios:
            st.write("Attack scenario mapping (sample):", scenarios[:10])

    # Unsupervised
    st.subheader("Unsupervised Results")
    st.markdown("These models do not require labels and detect anomalies or clusters.")
    unsup = train_unsupervised(X_all)
    st.write({k: str(type(v)) if not isinstance(v, str) else v for k,v in unsup.items()})

    if 'isolation_forest' in unsup and hasattr(unsup['isolation_forest'], 'predict'):
        iso_labels = unsup['isolation_forest'].predict(X_all)  # -1: anomaly, 1: normal
        st.write("Isolation Forest Predictions (sample):")
        st.dataframe(pd.DataFrame({'value': iso_labels[:50]}))
        iso_df = pd.DataFrame({'IsolationForest_Label': iso_labels})
        iso_csv = iso_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Isolation Forest Results CSV", data=iso_csv, file_name='isolation_forest_results.csv', mime='text/csv')

    if 'dbscan' in unsup and hasattr(unsup['dbscan'], 'labels_'):
        db_labels = unsup['dbscan'].labels_
        st.write("DBSCAN Cluster Labels (sample):")
        st.dataframe(pd.DataFrame({'value': db_labels[:50]}))
        db_df = pd.DataFrame({'DBSCAN_Label': db_labels})
        db_csv = db_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download DBSCAN Results CSV", data=db_csv, file_name='dbscan_results.csv', mime='text/csv')
