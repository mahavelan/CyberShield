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
MAX_TRAIN_ROWS = 20000  # Downsample for i3 machine
HASHER_FEATURES = 32
PCA_VARIANCE = 0.95
LOW_CARDINALITY_THRESHOLD = 50
HIGH_CARDINALITY_THRESHOLD = 200
TOP_K_FOR_CHART = 15  # Show only top 15

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
    # Data type recognition UI data
    dtype_info = pd.DataFrame({
        'Column': X.columns,
        'Type': [str(X[c].dtype) for c in X.columns]
    })
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
            X_proc, _, y, _ = train_test_split(
                X_proc, y, train_size=sample_limit, stratify=stratify, random_state=42
            )
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
    return X_final, (y.values if y is not None else None), {"pca": pca, "column_transformer": column_transformer, "label_encoders": label_encoders, "dtype_info": dtype_info, "df_all": df_all}

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
                "model": clf,
                "test_true": y_te,
                "test_pred": pred
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    return results

# ----------------- UNSUPERVISED -----------------
def train_unsupervised(X, y=None):
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

# ----------------- ATTACK SCENARIO MAPPING -----------------
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

# ----------------- CHART SUMMARIZATION -----------------
def get_chart_data(col_data):
    vc = col_data.value_counts()
    if len(vc) > TOP_K_FOR_CHART:
        top_n = vc[:TOP_K_FOR_CHART]
        summed = vc[TOP_K_FOR_CHART:].sum()
        top_n["Others"] = summed
        return top_n
    return vc

# ----------------- STREAMLIT APP -----------------
st.title("CyberShield: Supervised & Unsupervised Intrusion Detection")

st.markdown("**This app helps you detect intrusions and analyze network activity using supervised and unsupervised ML models. Choose models, view column summaries, and download your results!**")

uploaded_files = st.file_uploader(
    "Upload one or more CSV datasets", type=["csv"], accept_multiple_files=True
)

selected_models = st.multiselect(
    "Select supervised models to run", ["SGDClassifier", "RandomForest", "GradientBoosting"], default=["RandomForest"]
)

selected_graph = st.selectbox(
    "Select graph type for visualization",
    ["Bar", "Pie", "Scatter", "PCA"]
)

run_btn = st.button("Run Models")

if uploaded_files and run_btn:
    dfs = [pd.read_csv(f) for f in uploaded_files]
    st.write(f"Loaded {len(dfs)} dataset(s)")
    X, y, artifacts = harmonize_and_preprocess(dfs, label_col="Label", sample_limit=MAX_TRAIN_ROWS, perform_pca=True)
    st.write("Processed shape:", X.shape)

    # Data type recognition UI
    st.subheader("Data Type Recognition")
    st.write("Detected data types for each column in your dataset:")
    st.dataframe(artifacts["dtype_info"])

    # Graph selection UI
    st.subheader("Column Visualization")
    st.markdown("*Visualizes the distribution of selected column data; summarization is applied for high-cardinality columns.*")
    df_all = artifacts["df_all"]
    colnames = df_all.columns.tolist()
    vis_col = st.selectbox("Select column for visualization", colnames)
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
            if artifacts["pca"] is not None:
                pc_df = pd.DataFrame(artifacts["pca"].transform(df_all.drop(columns=["Label"], errors="ignore")), columns=[f"PC{i+1}" for i in range(artifacts["pca"].n_components_)])
                if 'Label' in df_all.columns:
                    ax.scatter(pc_df["PC1"], pc_df["PC2"], c=pd.factorize(df_all["Label"])[0], cmap='tab10', alpha=0.5)
                else:
                    ax.scatter(pc_df["PC1"], pc_df["PC2"], alpha=0.5)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title("PCA Plot")
        st.pyplot(fig)
        st.write(f"Showing {selected_graph} for: {vis_col} (top {TOP_K_FOR_CHART} + 'Others' shown for categorical data)")

    # --- SUPERVISED ---
    if y is not None:
        st.subheader("Supervised Results")
        st.markdown("These are results from models trained on known labels. Each row is classified, and you can download predictions.")
        st.write("Label distribution:", pd.Series(y).value_counts().to_dict())
        results = train_supervised(X, y, selected_models)
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
                st.download_button(
                    label=f"Download {name} Predictions as CSV",
                    data=csv,
                    file_name=f'{name}_test_predictions.csv',
                    mime='text/csv'
                )
            else:
                st.error(f"{name} failed: {res.get('error')}")
        best_name = max([n for n in results if "metrics" in results[n]], key=lambda nm: results[n]["metrics"]["f1"], default=None)
        if best_name:
            joblib.dump(results[best_name]["model"], "best_model.joblib")
            st.success(f"Best model saved: {best_name}")
        scenarios = map_attack_scenarios(y)
        if scenarios:
            st.write("Attack scenario mapping (sample):", scenarios[:10])

    # --- UNSUPERVISED ---
    st.subheader("Unsupervised Results")
    st.markdown("These results are from anomaly and clustering models that do not use labels. Outliers or clusters are identified automatically.")
    unsup = train_unsupervised(X, y)
    st.write({k: str(type(v)) if not isinstance(v, str) else v for k,v in unsup.items()})
    import io
    if 'isolation_forest' in unsup and hasattr(unsup['isolation_forest'], 'predict'):
        iso_labels = unsup['isolation_forest'].predict(X) # -1: anomaly, 1: normal
        st.write("Isolation Forest Predictions (sample):")
        st.dataframe(pd.DataFrame({'value': iso_labels[:50]}))
        iso_df = pd.DataFrame({'IsolationForest_Label': iso_labels})
        iso_csv = iso_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Isolation Forest Results CSV",
            data=iso_csv,
            file_name='isolation_forest_results.csv',
            mime='text/csv'
        )
    if 'dbscan' in unsup and hasattr(unsup['dbscan'], 'labels_'):
        db_labels = unsup['dbscan'].labels_
        st.write("DBSCAN Cluster Labels (sample):")
        st.dataframe(pd.DataFrame({'value': db_labels[:50]}))
        db_df = pd.DataFrame({'DBSCAN_Label': db_labels})
        db_csv = db_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download DBSCAN Results CSV",
            data=db_csv,
            file_name='dbscan_results.csv',
            mime='text/csv'
        )

