# app.py
"""
CYBER SHIELD(complete)
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
from io import BytesIO

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
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input
    from tensorflow.keras.optimizers import Adam
except Exception:
    USE_TF = False

st.set_page_config(page_title="cybershield", layout="wide")
st.title("🔎CYBER SHIELD")
st.markdown("Upload a CSV/XLSX dataset. The app auto-detects labeled/unlabeled and runs selected models.")

# ---------------------------
# Helpers
# ---------------------------
def find_label_candidates(df, keywords=None, max_unique_pct=0.05, max_unique_count=50):
    if keywords is None:
        keywords = ["label","class","attack","target","category","type","y","outcome","result"]
    n = len(df)
    scores = {}
    for col in df.columns:
        norm = re.sub(r'[^0-9a-zA-Z]', ' ', str(col)).lower().strip()
        name_score = sum(1 for kw in keywords if kw in norm)
        try:
            uniq_vals = df[col].dropna().unique()
            nunq = len(uniq_vals)
        except Exception:
            nunq = df[col].nunique(dropna=True)
        uniq_ratio = nunq / n if n>0 else 0
        sample_vals = set([str(x).strip().lower() for x in df[col].dropna().unique()[:100]])
        binary_like_sets = [{"0","1"},{"1","0"},{"true","false"},{"yes","no"},{"y","n"},
                            {"attack","normal"},{"malicious","benign"}]
        is_binary = any(sample_vals.issubset(s) for s in binary_like_sets) or nunq<=2
        value_score = 2 if is_binary else (1 if (nunq<=max_unique_count or uniq_ratio<=max_unique_pct) else 0)
        score = name_score*2 + value_score
        scores[col] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def preprocess_df_for_model(df, drop_cols=None):
    if drop_cols is None: drop_cols=[]
    df_proc = df.copy()
    df_proc = df_proc.dropna(axis=1, how='all')
    for c in drop_cols:
        if c in df_proc.columns: df_proc = df_proc.drop(columns=[c])
    cat_cols = df_proc.select_dtypes(include=['object','category']).columns.tolist()
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)
    df_proc = df_proc.fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_proc.values)
    X_df = pd.DataFrame(X, columns=df_proc.columns)
    return X_df, {'scaler': scaler, 'feature_columns': df_proc.columns.tolist()}

# Simple small models for optional DL (only if TF installed)
def build_simple_cnn(input_len):
    model = Sequential([Input(shape=(input_len,1)),
                        Conv1D(32,3,activation='relu'),
                        MaxPooling1D(2),
                        Conv1D(64,3,activation='relu'),
                        MaxPooling1D(2),
                        Flatten(),
                        Dense(64,activation='relu'),
                        Dense(1,activation='sigmoid')])
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_autoencoder(n_features):
    model = Sequential([Dense(64, activation='relu', input_shape=(n_features,)),
                        Dense(max(8, n_features//4), activation='relu'),
                        Dense(64, activation='relu'),
                        Dense(n_features, activation='linear')])
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload dataset (CSV or XLSX)", type=['csv','xlsx'])
if not uploaded_file:
    st.info("Upload data to begin. See Help below.")
    st.markdown("---")
else:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.subheader("Dataset preview")
    st.dataframe(df.head(10))

    # label detection
    candidates = find_label_candidates(df)
    auto_label = candidates[0][0] if candidates and candidates[0][1]>0 else None
    if auto_label:
        st.success(f"Auto-detected label candidate: **{auto_label}** (score={candidates[0][1]})")
    else:
        st.info("No strong label column detected.")

    options = ["-- None / Unlabeled --"] + list(df.columns)
    default_idx = options.index(auto_label) if (auto_label and auto_label in options) else 0
    label_choice = st.selectbox("Select label column (or choose '-- None / Unlabeled --')", options, index=default_idx)
    if label_choice == "-- None / Unlabeled --":
        dataset_type = "Unlabeled"
        label_col = None
        st.warning("Proceeding as Unlabeled dataset.")
    else:
        dataset_type = "Labeled"
        label_col = label_choice
        st.info(f"Using **{label_col}** as label.")

    with st.expander("Show candidates and scores"):
        cand_df = pd.DataFrame(candidates, columns=['column','score'])
        st.dataframe(cand_df)

    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("Model Options")
    if dataset_type=="Labeled":
        sup_choices = ["Logistic Regression","Random Forest","SVM (RBF)","KNN","Voting (LR+RF+SVM) - Hybrid"]
        if USE_TF: sup_choices += ["1D-CNN (train)","Autoencoder-Finetune (not for classification)"]
        chosen_sup = st.sidebar.multiselect("Supervised models", sup_choices, default=["Logistic Regression"])
        train_deep = st.sidebar.checkbox("Train deep models (may be slow)", value=False)
    else:
        unsup_choices = ["Isolation Forest","One-Class SVM","KMeans Clustering"]
        if USE_TF: unsup_choices += ["Autoencoder (train)"]
        unsup_choices += ["Consensus (hybrid)"]
        chosen_unsup = st.sidebar.multiselect("Unsupervised models", unsup_choices, default=["Isolation Forest"])
        contamination = st.sidebar.slider("Estimated contamination (anomaly fraction)", 0.001, 0.5, 0.05, 0.001)

    st.sidebar.markdown("---")
    run = st.sidebar.button("Run Selected Models")

    # Run handling
    if run:
        st.info("Preprocessing...")
        if dataset_type=="Labeled":
            df_features = df.drop(columns=[label_col]).copy()
            y = df[label_col].copy()
            if y.dtype=='object' or y.dtype.name=='category':
                y, uniques = pd.factorize(y)
            else:
                y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        else:
            df_features = df.copy()
            y = None

        try:
            X_processed, pipeline_info = preprocess_df_for_model(df_features, drop_cols=None)
        except Exception as err:
            st.error(f"Preprocessing error: {err}")
            st.stop()

        st.success("Preprocessing complete.")
        st.markdown("---")

        # ---------------------------
        # Supervised flow
        # ---------------------------
        if dataset_type=="Labeled":
            X = X_processed.values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y))>1 else None)

            metrics_summary = {}
            preds_store = {}

            # Logistic
            if "Logistic Regression" in chosen_sup:
                st.info("Training Logistic Regression...")
                lr = LogisticRegression(max_iter=1000)
                lr.fit(X_train, y_train)
                yp = lr.predict(X_test)
                preds_store["Logistic Regression"] = yp
                metrics_summary["Logistic Regression"] = (accuracy_score(y_test,yp), precision_score(y_test,yp,average='weighted',zero_division=0),
                                                         recall_score(y_test,yp,average='weighted',zero_division=0), f1_score(y_test,yp,average='weighted',zero_division=0))

            # Random Forest
            if "Random Forest" in chosen_sup:
                st.info("Training Random Forest...")
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                yp = rf.predict(X_test)
                preds_store["Random Forest"] = yp
                metrics_summary["Random Forest"] = (accuracy_score(y_test,yp), precision_score(y_test,yp,average='weighted',zero_division=0),
                                                   recall_score(y_test,yp,average='weighted',zero_division=0), f1_score(y_test,yp,average='weighted',zero_division=0))
                # show top feature importances
                try:
                    fi = rf.feature_importances_
                    fi_df = pd.DataFrame({'feature': pipeline_info['feature_columns'], 'importance': fi}).sort_values('importance',ascending=False).head(10)
                    st.write("Top feature importances (Random Forest):")
                    st.table(fi_df)
                except Exception:
                    pass

            # SVM
            if "SVM (RBF)" in chosen_sup:
                st.info("Training SVM (RBF)...")
                svm = SVC(kernel='rbf', probability=True)
                svm.fit(X_train, y_train)
                yp = svm.predict(X_test)
                preds_store["SVM (RBF)"] = yp
                metrics_summary["SVM (RBF)"] = (accuracy_score(y_test,yp), precision_score(y_test,yp,average='weighted',zero_division=0),
                                                 recall_score(y_test,yp,average='weighted',zero_division=0), f1_score(y_test,yp,average='weighted',zero_division=0))

            # KNN
            if "KNN" in chosen_sup:
                st.info("Training KNN...")
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(X_train, y_train)
                yp = knn.predict(X_test)
                preds_store["KNN"] = yp
                metrics_summary["KNN"] = (accuracy_score(y_test,yp), precision_score(y_test,yp,average='weighted',zero_division=0),
                                          recall_score(y_test,yp,average='weighted',zero_division=0), f1_score(y_test,yp,average='weighted',zero_division=0))

            # Voting Hybrid
            if "Voting (LR+RF+SVM) - Hybrid" in chosen_sup:
                st.info("Training Voting Hybrid (LR+RF+SVM)...")
                estimators = []
                # fit base if not present
                if "Logistic Regression" in preds_store:
                    estimators.append(('lr', lr))
                else:
                    lr2 = LogisticRegression(max_iter=1000); lr2.fit(X_train,y_train); estimators.append(('lr', lr2))
                if "Random Forest" in preds_store:
                    estimators.append(('rf', rf))
                else:
                    rf2 = RandomForestClassifier(n_estimators=100, random_state=42); rf2.fit(X_train,y_train); estimators.append(('rf', rf2))
                if "SVM (RBF)" in preds_store:
                    estimators.append(('svm', svm))
                else:
                    svm2 = SVC(kernel='rbf', probability=True); svm2.fit(X_train,y_train); estimators.append(('svm', svm2))
                vote = VotingClassifier(estimators=estimators, voting='soft')
                vote.fit(X_train,y_train)
                yp = vote.predict(X_test)
                preds_store["Voting Hybrid"] = yp
                metrics_summary["Voting Hybrid"] = (accuracy_score(y_test,yp), precision_score(y_test,yp,average='weighted',zero_division=0),
                                                   recall_score(y_test,yp,average='weighted',zero_division=0), f1_score(y_test,yp,average='weighted',zero_division=0))

            # Optional simple DL training for binary only (if TF & enabled)
            if USE_TF and train_deep:
                st.info("Deep model training requested but kept minimal here (can be enabled).")

            # Show supervised metrics summary chart
            st.header("Supervised Models — Metrics Summary")
            if metrics_summary:
                metrics_df = pd.DataFrame.from_dict({m: {'Accuracy': v[0], 'Precision': v[1], 'Recall': v[2], 'F1': v[3]} for m,v in metrics_summary.items()}, orient='index')
                st.table(metrics_df.round(3))

                # bar chart of accuracy
                fig, ax = plt.subplots()
                metrics_df['Accuracy'].plot(kind='bar', ax=ax)
                ax.set_ylabel("Accuracy")
                st.pyplot(fig)

            # Show confusion matrices and downloads
            st.markdown("---")
            st.header("Supervised — Confusion Matrices & Download")
            for name, yp in preds_store.items():
                st.subheader(name)
                cm = confusion_matrix(y_test, yp)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
                ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
                st.pyplot(fig)
                # prepare download
                test_df = pd.DataFrame(X_test, columns=pipeline_info['feature_columns'])
                test_df['Actual'] = y_test
                test_df['Predicted'] = yp
                csv = test_df.to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {name} results", csv, file_name=f"{name.replace(' ','_')}_results.csv")
        # ---------------------------
        # Unsupervised flow
        # ---------------------------
        else:
            X_vals = X_processed.values
            unsup_results = {}
            percent_flagged = {}

            if "Isolation Forest" in chosen_unsup:
                st.info("Running Isolation Forest...")
                iso = IsolationForest(contamination=contamination, random_state=42)
                iso_pred = iso.fit_predict(X_vals)  # -1 anomaly
                labels = np.where(iso_pred==-1, "Attack", "Normal")
                df_iso = df.copy(); df_iso['Pred_IsolationForest'] = labels
                unsup_results['Isolation Forest'] = df_iso
                percent_flagged['Isolation Forest'] = (labels=="Attack").sum()/len(labels)

            if "One-Class SVM" in chosen_unsup:
                st.info("Running One-Class SVM...")
                try:
                    oc = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
                    oc_pred = oc.fit_predict(X_vals)
                    labels = np.where(oc_pred==-1, "Attack", "Normal")
                    df_oc = df.copy(); df_oc['Pred_OneClassSVM'] = labels
                    unsup_results['One-Class SVM'] = df_oc
                    percent_flagged['One-Class SVM'] = (labels=="Attack").sum()/len(labels)
                except Exception as ex:
                    st.error(f"One-Class SVM failed: {ex}")

            if "KMeans Clustering" in chosen_unsup:
                st.info("Running KMeans (k=3)...")
                try:
                    kmeans = KMeans(n_clusters=3, random_state=42)
                    clusters = kmeans.fit_predict(X_vals)
                    df_km = df.copy(); df_km['km_cluster'] = clusters
                    small = df_km['km_cluster'].value_counts().idxmin()
                    df_km['Pred_KMeans'] = np.where(df_km['km_cluster']==small, "Attack", "Normal")
                    unsup_results['KMeans'] = df_km
                    percent_flagged['KMeans'] = (df_km['Pred_KMeans']=="Attack').sum()/len(df_km)
                except Exception as ex:
                    st.error(f"KMeans failed: {ex}")

            if USE_TF and "Autoencoder (train)" in chosen_unsup:
                st.info("Training Autoencoder (this may take time)...")
                try:
                    ae = build_autoencoder(X_vals.shape[1])
                    ae.fit(X_vals, X_vals, epochs=20, batch_size=128, verbose=0)
                    recon = ae.predict(X_vals)
                    rec_err = np.mean(np.square(recon - X_vals), axis=1)
                    thr = np.percentile(rec_err, 100*(1-contamination))
                    labels = np.where(rec_err>=thr, "Attack", "Normal")
                    df_ae = df.copy(); df_ae['AE_recon_err'] = rec_err; df_ae['Pred_AE'] = labels
                    unsup_results['Autoencoder'] = df_ae
                    percent_flagged['Autoencoder'] = (labels=="Attack").sum()/len(labels)
                except Exception as ex:
                    st.error(f"Autoencoder failed: {ex}")

            # Consensus hybrid: majority voting across produced models
            if "Consensus (hybrid)" in chosen_unsup:
                st.info("Computing Consensus (majority) across unsupervised models...")
                if not unsup_results:
                    st.warning("No model outputs available for consensus.")
                else:
                    cons_df = df.copy()
                    flag_cols = []
                    for name, dfm in unsup_results.items():
                        # find prediction column
                        pcols = [c for c in dfm.columns if c.startswith('Pred_') or c.startswith('Pred_') or c.startswith('Prediction_')]
                        if not pcols:
                            # fallback standard names
                            pcols = [c for c in dfm.columns if c.startswith('Pred_') or c.endswith('_Pred')]
                        if pcols:
                            col = pcols[0]
                            cons_df[col] = dfm[col]
                            flag_cols.append(col)
                    if flag_cols:
                        flags = cons_df[flag_cols].applymap(lambda x: 1 if str(x).lower().startswith('attack') else 0)
                        cons_df['consensus_score'] = flags.sum(axis=1)/len(flag_cols)
                        cons_df['Consensus_Pred'] = np.where(cons_df['consensus_score']>=0.5, "Attack", "Normal")
                        unsup_results['Consensus'] = cons_df
                        percent_flagged['Consensus'] = (cons_df['Consensus_Pred']=="Attack').sum()/len(cons_df)

            # Visualizations: pie per model, and comparison bar of percent flagged
            st.header("Unsupervised Results & Visuals")
            for name, df_out in unsup_results.items():
                st.subheader(name)
                # find pred col
                predcols = [c for c in df_out.columns if c.startswith('Pred_') or c=='Consensus_Pred' or c=='Pred_AE']
                if predcols:
                    pcol = predcols[0]
                    vc = df_out[pcol].value_counts()
                    attacks = int(vc.get('Attack',0))
                    normals = int(vc.get('Normal',0))
                    fig, ax = plt.subplots()
                    ax.pie([attacks, normals], labels=['Attack','Normal'], autopct='%1.1f%%', colors=['red','green'])
                    st.pyplot(fig)
                # download
                csv = df_out.to_csv(index=False).encode('utf-8')
                st.download_button(f"Download {name} predictions", csv, file_name=f"{name.replace(' ','_')}_preds.csv")

            # Comparison bar chart
            if percent_flagged:
                st.subheader("Model comparison — % flagged as Attack")
                comp_df = pd.DataFrame.from_dict({k: {'percent': v} for k,v in percent_flagged.items()}, orient='index')
                comp_df = comp_df.sort_values('percent', ascending=False)
                st.table((comp_df*100).round(2))
                fig, ax = plt.subplots()
                comp_df['percent'].plot(kind='bar', ax=ax)
                ax.set_ylabel("Fraction flagged as Attack")
                st.pyplot(fig)

    # end run button

# Help
st.markdown("---")
with st.expander("Help & Quickstart"):
    st.write("""
    Quick steps:
    1. Upload CSV/XLSX file (rows = records, columns = features). If you have labels, ensure a label column exists.
    2. App auto-detects label column; override if needed.
    3. In the sidebar choose supervised/unsupervised models and press 'Run Selected Models'.
    4. For labeled data you'll get metrics and confusion matrices. For unlabeled you'll get flagged counts and charts.
    Notes:
    - Voting (hybrid) implemented for supervised; Consensus (hybrid) for unsupervised.
    - Deep models require TensorFlow and more compute — optional.
    """)

