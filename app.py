# app.py
"""
CYBERSHIELD — full Streamlit app (single file)
- Strict label detection by name only (no heuristics)
- Auto-route to supervised / unsupervised
- Single vs Hybrid model selection enforced by checkbox
- Single vs Multiple visualizations enforced by checkbox
- Automatic preprocessing, metrics, visualizations, CSV downloads
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF logs (INFO/WARNING)

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# supervised classical
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# unsupervised classical
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

# Optional deep learning
USE_TF = True
try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel("ERROR")
except Exception:
    USE_TF = False

# ---------------------------
# Utility functions
# ---------------------------
STRICT_LABEL_NAMES = {
    "label", "class", "target", "attack", "y", "output", "result", "category", "type", "true_label"
}

def normalize_name(colname: str) -> str:
    """Normalize column name: remove non-alphanumeric, lowercase."""
    return re.sub(r"[^0-9a-zA-Z]+", "", str(colname)).strip().lower()

def find_label_column_strict(df: pd.DataFrame):
    """
    Return column name if a strict label name exists (normalized).
    Only exact normalized-name matches are accepted.
    """
    for col in df.columns:
        if normalize_name(col) in STRICT_LABEL_NAMES:
            return col
    return None

def safe_read_file(uploaded):
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    else:
        return pd.read_excel(uploaded)

def preprocess_features(df: pd.DataFrame, drop_cols=None):
    """
    Automatic preprocessing:
    - drop fully empty columns
    - drop drop_cols
    - fill numeric NaN with median; categorical NaN with 'missing'
    - one-hot encode categorical columns (drop_first=True)
    - drop constant columns
    - Standard scale numeric features
    Returns (X_df, pipeline_info)
    """
    dfc = df.copy()
    if drop_cols:
        dfc = dfc.drop(columns=[c for c in drop_cols if c in dfc.columns], errors='ignore')
    # drop fully empty columns
    dfc = dfc.dropna(axis=1, how='all')
    # fill missing: numeric -> median, others -> 'missing'
    for c in dfc.columns:
        if pd.api.types.is_numeric_dtype(dfc[c]):
            if dfc[c].dropna().shape[0] > 0:
                dfc[c] = dfc[c].fillna(dfc[c].median())
            else:
                dfc[c] = dfc[c].fillna(0)
        else:
            dfc[c] = dfc[c].fillna("missing")
    # one-hot encode categoricals
    cat_cols = dfc.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        dfc = pd.get_dummies(dfc, columns=cat_cols, drop_first=True)
    # drop constant columns
    nunq = dfc.nunique(dropna=True)
    const_cols = nunq[nunq <= 1].index.tolist()
    if const_cols:
        dfc = dfc.drop(columns=const_cols, errors='ignore')
    if dfc.shape[1] == 0:
        raise ValueError("No usable features after preprocessing.")
    # scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dfc.values)
    X_df = pd.DataFrame(X_scaled, columns=dfc.columns, index=dfc.index)
    pipeline = {"scaler": scaler, "feature_columns": dfc.columns.tolist()}
    return X_df, pipeline

def build_keras_mlp(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_keras_cnn(input_len):
    model = Sequential([
        Input(shape=(input_len,1)),
        Conv1D(32,3,activation='relu'),
        MaxPooling1D(2),
        Conv1D(64,3,activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64,activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def majority_vote_predictions(pred_list):
    """
    pred_list: list of 1D arrays (same length) with predicted labels (ints or strings)
    returns: majority-vote labels (same type)
    """
    df = pd.DataFrame(pred_list).T  # each column = one model
    # For numeric-coded supervised labels, majority by value
    result = df.mode(axis=1)[0].values
    return result

def compute_supervised_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "rmse": rmse}

def plot_confusion_mat(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if labels is not None:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    st.pyplot(fig)
    plt.close(fig)

def plot_pie_bar_from_series(s: pd.Series, title=""):
    fig, ax = plt.subplots()
    vc = s.value_counts()
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

def plot_generic(df, kind, x=None, y=None, hue=None, title=None):
    fig, ax = plt.subplots()
    try:
        if kind == "Bar":
            if x and y:
                sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
            else:
                df.select_dtypes(include=np.number).sum().plot(kind='bar', ax=ax)
        elif kind == "Pie":
            if x:
                s = df[x].value_counts()
                ax.pie(s.values, labels=s.index, autopct="%1.1f%%")
            else:
                df.iloc[:, 0].value_counts().plot(kind='pie', ax=ax)
        elif kind == "Heatmap":
            sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='coolwarm', ax=ax)
        elif kind == "Line":
            if x and y:
                sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
            else:
                df.reset_index(drop=True).plot(ax=ax)
        elif kind == "Scatter":
            if x and y:
                sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
        elif kind == "Histogram":
            if y:
                sns.histplot(data=df, x=y, hue=hue, kde=True, ax=ax)
            else:
                df.select_dtypes(include=np.number).hist(ax=ax)
                st.pyplot(fig)
                plt.close(fig)
                return
        elif kind == "Box":
            if y:
                sns.boxplot(data=df, y=y, ax=ax)
            else:
                sns.boxplot(data=df.select_dtypes(include=np.number), ax=ax)
        ax.set_title(title or kind)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not render {kind}: {e}")
    finally:
        plt.close(fig)

# ---------------------------
# UI: upload and auto-detect
# ---------------------------
st.set_page_config(page_title="CyberShield", layout="wide")
st.title("🛡 CyberShield — Intrusion Detection Web App (Auto Detect Label)")

uploaded = st.file_uploader("Upload CSV or Excel file (dataset)", type=["csv", "xlsx"])
if uploaded is None:
    st.info("Please upload a dataset (CSV/XLSX). The app will auto-detect if it's supervised or unsupervised based on a strict label column name.")
    st.stop()

# read
try:
    df = safe_read_file(uploaded)
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()

st.subheader("Dataset preview (first 10 rows)")
st.dataframe(df.head(10))

# strict label detection
label_col = find_label_column_strict(df)
if label_col:
    st.success(f"Auto-detected label column (strict name match): **{label_col}** — routing to Supervised flow.")
    dataset_type = "Supervised"
else:
    st.info("No strict label column found → routing to Unsupervised flow.")
    dataset_type = "Unsupervised"

# show column list for reference (read-only)
with st.expander("Columns in uploaded dataset"):
    st.write(list(df.columns))

# ---------------------------
# Sidebar: model & graph options
# ---------------------------
st.sidebar.header("Model & Visualization Options")

if dataset_type == "Supervised":
    st.sidebar.subheader("Supervised models")
    SUP_CLASSICAL = [
        "Logistic Regression", "Random Forest", "Decision Tree",
        "KNN", "SVM (RBF)", "Naive Bayes", "Gradient Boosting", "AdaBoost", "MLP (sklearn)"
    ]
    SUP_DEEP = []
    if USE_TF:
        SUP_DEEP = ["Keras-MLP", "Keras-1D-CNN"]
    hybrid_checkbox = st.sidebar.checkbox("Enable Hybrid Ensemble (choose multiple models)?", value=False, key="hybrid_sup")
    if hybrid_checkbox:
        chosen_models = st.sidebar.multiselect("Choose 2 or more models for Hybrid", SUP_CLASSICAL + SUP_DEEP)
    else:
        chosen_one = st.sidebar.selectbox("Choose ONE model", SUP_CLASSICAL + SUP_DEEP)
        chosen_models = [chosen_one]
    train_deep_toggle = st.sidebar.checkbox("Train deep models (if selected) — may be slow", value=False)
else:
    st.sidebar.subheader("Unsupervised models")
    UNSUP_CLASSICAL = ["Isolation Forest", "One-Class SVM", "KMeans", "DBSCAN", "PCA-heuristic"]
    UNSUP_DEEP = []
    if USE_TF:
        UNSUP_DEEP = ["Autoencoder (Keras)"]
    hybrid_checkbox = st.sidebar.checkbox("Enable Hybrid Consensus (choose multiple models)?", value=False, key="hybrid_unsup")
    if hybrid_checkbox:
        chosen_models = st.sidebar.multiselect("Choose 2 or more unsupervised models", UNSUP_CLASSICAL + UNSUP_DEEP)
    else:
        chosen_one = st.sidebar.selectbox("Choose ONE unsupervised model", UNSUP_CLASSICAL + UNSUP_DEEP)
        chosen_models = [chosen_one]
    contamination = st.sidebar.slider("Estimated contamination (anomaly fraction)", min_value=0.001, max_value=0.5, value=0.05, step=0.001)

# Visualization options
st.sidebar.subheader("Visualizations")
graph_multi_checkbox = st.sidebar.checkbox("Enable multiple visualizations?", value=False, key="multi_graphs")
GRAPH_OPTIONS = ["Confusion Matrix", "Bar", "Pie", "Line", "Histogram", "Box", "Heatmap"]
if graph_multi_checkbox:
    chosen_graphs = st.sidebar.multiselect("Choose visualizations (2+)", GRAPH_OPTIONS, default=["Bar","Pie"])
else:
    chosen_graph = st.sidebar.selectbox("Choose one visualization", GRAPH_OPTIONS)
    chosen_graphs = [chosen_graph]

# Run button
run_btn = st.sidebar.button("Run")

# sanity checks on selection counts
def validate_selections():
    if dataset_type == "Supervised":
        if hybrid_checkbox:
            if len(chosen_models) < 2:
                st.sidebar.error("Hybrid ON → select at least 2 models.")
                return False
        else:
            if len(chosen_models) != 1:
                st.sidebar.error("Hybrid OFF → select exactly 1 model.")
                return False
    else:
        if hybrid_checkbox:
            if len(chosen_models) < 2:
                st.sidebar.error("Hybrid ON → select at least 2 models.")
                return False
        else:
            if len(chosen_models) != 1:
                st.sidebar.error("Hybrid OFF → select exactly 1 model.")
                return False

    if graph_multi_checkbox:
        if len(chosen_graphs) < 2:
            st.sidebar.error("Multi-graphs ON → select 2 or more visualizations.")
            return False
    else:
        if len(chosen_graphs) != 1:
            st.sidebar.error("Multi-graphs OFF → select exactly 1 visualization.")
            return False
    return True

# ---------------------------
# Run models
# ---------------------------
if run_btn:
    if not validate_selections():
        st.stop()

    try:
        if dataset_type == "Supervised":
            # prepare X and y
            if label_col not in df.columns:
                st.error("Label column not found after reading file — please re-upload.")
                st.stop()
            X_raw = df.drop(columns=[label_col])
            y_raw = df[label_col]
            # factorize labels (robust)
            y_codes, uniques = pd.factorize(y_raw.astype(str))
            label_mapping = {i: v for i, v in enumerate(uniques)}
            st.write(f"Label classes detected: {list(uniques)}")
            # preprocess X
            X_proc, pipeline = preprocess_features(X_raw, drop_cols=None)
            # split
            stratify_param = y_codes if len(np.unique(y_codes)) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(X_proc.values, y_codes, test_size=0.3, random_state=42, stratify=stratify_param)
            st.success("Preprocessing & train-test split done.")
            # container to store model predictions and metrics
            preds = {}
            metrics = {}

            # helper for classical training
            def run_classical(name, clf):
                try:
                    clf.fit(X_train, y_train)
                    yp = clf.predict(X_test)
                    preds[name] = yp
                    metrics[name] = compute_supervised_metrics(y_test, yp)
                except Exception as e:
                    st.warning(f"{name} failed: {e}")

            # run each selected model
            for m in chosen_models:
                if m == "Logistic Regression":
                    run_classical(m, LogisticRegression(max_iter=1000))
                elif m == "Random Forest":
                    run_classical(m, RandomForestClassifier(n_estimators=100, random_state=42))
                elif m == "Decision Tree":
                    run_classical(m, DecisionTreeClassifier(random_state=42))
                elif m == "KNN":
                    run_classical(m, KNeighborsClassifier())
                elif m == "SVM (RBF)":
                    run_classical(m, SVC(kernel='rbf', probability=True))
                elif m == "Naive Bayes":
                    run_classical(m, GaussianNB())
                elif m == "Gradient Boosting":
                    run_classical(m, GradientBoostingClassifier(random_state=42))
                elif m == "AdaBoost":
                    run_classical(m, AdaBoostClassifier(random_state=42))
                elif m == "MLP (sklearn)":
                    run_classical(m, MLPClassifier(max_iter=500))
                elif USE_TF and m == "Keras-MLP":
                    try:
                        model = build_keras_mlp(X_train.shape[1])
                        if train_deep_toggle:
                            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                        else:
                            model.fit(X_train, y_train, epochs=4, batch_size=32, verbose=0)
                        yp = (model.predict(X_test).ravel() >= 0.5).astype(int)
                        preds[m] = yp
                        metrics[m] = compute_supervised_metrics(y_test, yp)
                    except Exception as e:
                        st.warning(f"Keras-MLP failed: {e}")
                elif USE_TF and m == "Keras-1D-CNN":
                    try:
                        model = build_keras_cnn(X_train.shape[1])
                        Xtr = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                        Xte = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                        if train_deep_toggle:
                            model.fit(Xtr, y_train, epochs=8, batch_size=32, verbose=0)
                        else:
                            model.fit(Xtr, y_train, epochs=4, batch_size=32, verbose=0)
                        yp = (model.predict(Xte).ravel() >= 0.5).astype(int)
                        preds[m] = yp
                        metrics[m] = compute_supervised_metrics(y_test, yp)
                    except Exception as e:
                        st.warning(f"Keras-1D-CNN failed: {e}")
                else:
                    st.warning(f"Unknown model selected: {m}")

            # If hybrid enabled: combine model predictions by majority vote
            if hybrid_checkbox:
                # ensure at least 2 model preds exist
                pred_arrays = []
                pred_names = []
                for k, v in preds.items():
                    pred_arrays.append(np.asarray(v))
                    pred_names.append(k)
                if len(pred_arrays) >= 2:
                    mv = majority_vote_predictions(pred_arrays)
                    preds["Hybrid-Majority"] = mv
                    metrics["Hybrid-Majority"] = compute_supervised_metrics(y_test, mv)
                else:
                    st.warning("Hybrid selected but fewer than 2 models produced predictions; cannot perform hybrid majority vote.")

            # Display results
            if metrics:
                st.header("Supervised — Metrics Summary")
                rows = []
                for name, met in metrics.items():
                    rows.append([name, met['accuracy'], met['precision'], met['recall'], met['f1'], met['rmse']])
                res_df = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "RMSE"]).set_index("Model")
                st.dataframe(res_df.round(4))

                # For each model: confusion matrix, visualizations, download
                for name, ypred in preds.items():
                    st.subheader(f"Model: {name}")
                    # confusion matrix with label names (original)
                    try:
                        plot_confusion_mat(y_test, ypred, labels=[label_mapping[i] for i in range(len(uniques))] if len(uniques) <= 10 else None)
                    except Exception:
                        st.write("Could not plot confusion matrix (maybe many classes).")
                    # chosen visualizations
                    for g in chosen_graphs:
                        if g == "Confusion Matrix":
                            # already shown
                            continue
                        # create dataframe for plotting: X_test to DataFrame
                        X_test_df = pd.DataFrame(X_test, columns=pipeline['feature_columns'])
                        plot_generic(X_test_df, g, title=f"{name} — {g}")
                    # download csv: features + actual + predicted (map codes to original labels)
                    X_test_df = pd.DataFrame(X_test, columns=pipeline['feature_columns'])
                    out = X_test_df.copy()
                    out["Actual_label"] = [label_mapping[v] for v in y_test]
                    out["Predicted_label"] = [label_mapping[v] if isinstance(v, (np.integer,int)) and v in label_mapping else str(v) for v in ypred]
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(f"Download results - {name}", data=csv, file_name=f"results_{name}.csv", mime="text/csv")

            else:
                st.warning("No model produced metrics. Check selections and logs.")

        else:
            # ---------------------------
            # Unsupervised flow
            # ---------------------------
            st.subheader("Unsupervised — preprocessing")
            X_proc, pipeline = preprocess_features(df, drop_cols=None)
            X_vals = X_proc.values
            st.success("Preprocessing done.")

            unsup_preds = {}  # model name -> array of 'Attack'/'Normal'
            # run each chosen model
            for m in chosen_models:
                st.info(f"Running {m} ...")
                try:
                    if m == "Isolation Forest":
                        iso = IsolationForest(contamination=contamination, random_state=42)
                        p = iso.fit_predict(X_vals)
                        lbls = np.where(p == -1, "Attack", "Normal")
                    elif m == "One-Class SVM":
                        oc = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
                        p = oc.fit_predict(X_vals)
                        lbls = np.where(p == -1, "Attack", "Normal")
                    elif m == "KMeans":
                        km = KMeans(n_clusters=2, random_state=42)
                        cl = km.fit_predict(X_vals)
                        counts = pd.Series(cl).value_counts()
                        small_cluster = counts.idxmin()
                        lbls = np.where(cl == small_cluster, "Attack", "Normal")
                    elif m == "DBSCAN":
                        db = DBSCAN(eps=0.5, min_samples=5)
                        cl = db.fit_predict(X_vals)
                        lbls = np.where(cl == -1, "Attack", "Normal")
                    elif m == "PCA-heuristic":
                        pca = PCA(n_components=min(5, X_vals.shape[1]))
                        comp = pca.fit_transform(X_vals)
                        dist = np.linalg.norm(comp - comp.mean(axis=0), axis=1)
                        thr = np.percentile(dist, 100 * (1 - contamination))
                        lbls = np.where(dist >= thr, "Attack", "Normal")
                    elif USE_TF and m == "Autoencoder (Keras)":
                        # small autoencoder
                        ae = Sequential([
                            Input(shape=(X_vals.shape[1],)),
                            Dense(max(16, X_vals.shape[1]//2), activation='relu'),
                            Dense(max(8, X_vals.shape[1]//4), activation='relu'),
                            Dense(max(16, X_vals.shape[1]//2), activation='relu'),
                            Dense(X_vals.shape[1], activation='linear'),
                        ])
                        ae.compile(optimizer=Adam(1e-3), loss='mse')
                        ae.fit(X_vals, X_vals, epochs=10, batch_size=128, verbose=0)
                        recon = ae.predict(X_vals)
                        rec_err = np.mean(np.square(recon - X_vals), axis=1)
                        thr = np.percentile(rec_err, 100 * (1 - contamination))
                        lbls = np.where(rec_err >= thr, "Attack", "Normal")
                    else:
                        st.warning(f"Unrecognized unsupervised model or TF not available: {m}")
                        continue
                    unsup_preds[m] = lbls
                    # show counts
                    s = pd.Series(lbls)
                    st.write(f"{m} — Attack: { (s=='Attack').sum() } | Normal: { (s=='Normal').sum() }  ( { (s=='Attack').mean():.2% } flagged )")
                    # show chosen graphs
                    # prepare df for plotting
                    for g in chosen_graphs:
                        if g in ("Pie", "Bar"):
                            plot_pie_bar_from_series(s, title=f"{m} — {g}")
                        else:
                            # use X_proc for other visual types
                            plot_generic(X_proc, g, title=f"{m} — {g}")
                    # download
                    out = df.copy()
                    out[f"Pred_{m}"] = lbls
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(f"Download predictions ({m})", data=csv, file_name=f"unsup_{m}.csv", mime="text/csv")
                except Exception as ex:
                    st.warning(f"{m} failed: {ex}")

            # consensus/hybrid across unsupervised
            if hybrid_checkbox:
                if len(unsup_preds) >= 2:
                    df_flags = pd.DataFrame(unsup_preds)
                    # convert Attack->1, Normal->0
                    flags_num = df_flags.applymap(lambda x: 1 if str(x).lower().startswith("attack") else 0)
                    cons_score = flags_num.mean(axis=1)
                    cons_pred = np.where(cons_score >= 0.5, "Attack", "Normal")
                    st.subheader("Consensus (majority) across unsupervised models")
                    s = pd.Series(cons_pred)
                    st.write(f"Consensus — Attack: { (s=='Attack').sum() } | Normal: { (s=='Normal').sum() }")
                    for g in chosen_graphs:
                        if g in ("Pie","Bar"):
                            plot_pie_bar_from_series(s, title=f"Consensus — {g}")
                        else:
                            plot_generic(X_proc, g, title=f"Consensus — {g}")
                    out = df.copy(); out["Consensus_Pred"] = cons_pred
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Consensus predictions", data=csv, file_name="unsup_consensus.csv", mime="text/csv")
                else:
                    st.warning("Consensus requested but fewer than 2 unsupervised model outputs available.")
    except Exception as main_ex:
        st.error(f"An error occurred while running models: {main_ex}")

# Help / notes
st.markdown("---")
with st.expander("Help & Notes"):
    st.markdown("""
    - **Label detection** uses *strict normalized name matching* only (e.g. 'label', 'Label', 'LABEL', 'Label ' etc.)
      — this prevents accidental selection of columns like 'syn_flag_count' or 'flow_bytes' as a label.
    - If your dataset truly has a label column but the app didn't detect it, rename the column to e.g. `label` or `class` and re-upload.
    - **Hybrid**: turn ON to select multiple models; OFF means select exactly one model.
    - **Visualization**: toggle 'Enable multiple visualizations?' to pick multiple graphs.
    - Deep models require TensorFlow; if not installed the deep options will not appear.
    - Autoencoder / deep training is CPU-bound and may be slow — for demo use small datasets or enable only classical models.
    """)
