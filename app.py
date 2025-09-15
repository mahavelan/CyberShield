# app.py
"""
CYBERSHIELD — Safe Deployable Web App
"""

import streamlit as st
import traceback

try:
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import re
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error

    # Classical ML models
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ---------------------------
    # Utilities
    # ---------------------------
    STRICT_LABEL_NAMES = {"label", "class", "target", "attack", "y", "output", "result"}

    def normalize_name(colname: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "", str(colname)).strip().lower()

    def find_label_column_strict(df: pd.DataFrame):
        for col in df.columns:
            if normalize_name(col) in STRICT_LABEL_NAMES:
                return col
        return None

    def detect_dataset_type(uploaded_file, df: pd.DataFrame):
        name = uploaded_file.name.lower()
        if name.endswith((".png", ".jpg", ".jpeg")):
            return "Image"
        text_like = sum([df[col].dtype == "object" for col in df.columns])
        if text_like > len(df.columns) / 2:
            return "Text"
        return "Numerical"

    def preprocess_numeric(df, label_col=None):
        if label_col:
            X = df.drop(columns=[label_col])
            y = df[label_col]
        else:
            X = df.copy()
            y = None
        X = X.fillna(X.median(numeric_only=True))
        X = pd.get_dummies(X, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y

    def compute_supervised_metrics(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "rmse": mean_squared_error(y_true, y_pred, squared=False),
        }

    def plot_confusion(y_true, y_pred, labels=None):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    def plot_generic(df, kind, title=""):
        fig, ax = plt.subplots()
        try:
            if kind == "Histogram":
                df.hist(ax=ax)
            elif kind == "Heatmap":
                sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
            elif kind == "Bar":
                df.sum().plot(kind="bar", ax=ax)
            elif kind == "Line":
                df.plot(ax=ax)
            elif kind == "Pie":
                df.iloc[:,0].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Plot failed: {e}")
        plt.close(fig)

    # ---------------------------
    # Streamlit UI
    # ---------------------------
    st.set_page_config(page_title="CyberShield", layout="wide")
    st.title("🛡 CyberShield — Multi-Dataset Intrusion Detection App")

    uploaded = st.file_uploader("Upload your dataset (CSV, XLSX, TXT, JPG, PNG)", type=["csv","xlsx","txt","jpg","jpeg","png"])
    if not uploaded:
        st.stop()

    if uploaded.name.endswith(("csv","xlsx","txt")):
        df = pd.read_csv(uploaded) if uploaded.name.endswith("csv") else pd.read_excel(uploaded)
        st.write("Dataset Preview", df.head())
    else:
        df = pd.DataFrame()  # placeholder for image/text

    label_col = find_label_column_strict(df) if not df.empty else None
    dtype = detect_dataset_type(uploaded, df if not df.empty else pd.DataFrame())
    st.info(f"Detected dataset type: **{dtype}**")

    # Sidebar
    st.sidebar.header("Options")
    if dtype == "Numerical":
        models = ["Logistic Regression","Random Forest","Decision Tree","KNN","SVM","Naive Bayes","Gradient Boosting","AdaBoost","Keras-MLP"]
    elif dtype == "Text":
        models = ["Naive Bayes (Text)","Logistic Regression (Text)","SVM (Text)","Random Forest (Text)"]
    elif dtype == "Image":
        models = ["CNN-2D","ResNet50","MobileNetV2"]

    hybrid = st.sidebar.checkbox("Enable Hybrid (choose multiple models)?")
    if hybrid:
        chosen = st.sidebar.multiselect("Choose 2+ models", models)
    else:
        chosen = [st.sidebar.selectbox("Choose ONE model", models)]

    multi_graph = st.sidebar.checkbox("Enable multiple visualizations?")
    graphs = st.sidebar.multiselect("Choose graphs", ["Confusion Matrix","Histogram","Heatmap","Bar","Line","Pie"]) if multi_graph else [st.sidebar.selectbox("Choose one graph", ["Confusion Matrix","Histogram","Heatmap","Bar","Line","Pie"])]

    train_deep = st.sidebar.checkbox("Train deep models (if selected)")
    run_btn = st.sidebar.button("Run")

    # ---------------------------
    # Run
    # ---------------------------
    if run_btn:
        try:
            if dtype == "Numerical" and not df.empty:
                X, y = preprocess_numeric(df, label_col)
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42) if label_col else (X,None,None,None)

                st.write("X_train shape:", X_train.shape)
                st.write("y_train shape:", y_train.shape if y_train is not None else None)
                st.write("Classes:", np.unique(y_train) if y_train is not None else None)

                # Lazy import TensorFlow only if deep models are selected
                USE_TF = False
                if train_deep and any("Keras" in m for m in chosen):
                    try:
                        import tensorflow as tf
                        from tensorflow.keras import Sequential
                        from tensorflow.keras.layers import Dense
                        USE_TF = True
                        tf.get_logger().setLevel("ERROR")
                    except Exception as e:
                        st.warning(f"TensorFlow failed to import: {e}")
                        USE_TF = False

                st.write("Models running...")
                for m in chosen:
                    try:
                        if m == "Logistic Regression":
                            clf = LogisticRegression(max_iter=1000).fit(X_train,y_train)
                            pred = clf.predict(X_test)
                        elif m == "Random Forest":
                            clf = RandomForestClassifier().fit(X_train,y_train)
                            pred = clf.predict(X_test)
                        elif m == "Decision Tree":
                            clf = DecisionTreeClassifier().fit(X_train,y_train)
                            pred = clf.predict(X_test)
                        elif m == "KNN":
                            clf = KNeighborsClassifier().fit(X_train,y_train)
                            pred = clf.predict(X_test)
                        elif m == "SVM":
                            clf = SVC().fit(X_train,y_train)
                            pred = clf.predict(X_test)
                        elif m == "Naive Bayes":
                            clf = GaussianNB().fit(X_train,y_train)
                            pred = clf.predict(X_test)
                        elif m == "Gradient Boosting":
                            clf = GradientBoostingClassifier().fit(X_train,y_train)
                            pred = clf.predict(X_test)
                        elif m == "AdaBoost":
                            clf = AdaBoostClassifier().fit(X_train,y_train)
                            pred = clf.predict(X_test)
                        elif USE_TF and m == "Keras-MLP":
                            # Handle binary vs multi-class
                            n_classes = len(np.unique(y_train))
                            if n_classes > 2:
                                activation = "softmax"
                                loss = "sparse_categorical_crossentropy"
                                output_units = n_classes
                            else:
                                activation = "sigmoid"
                                loss = "binary_crossentropy"
                                output_units = 1

                            model = Sequential([
                                Dense(128, activation="relu"),
                                Dense(64, activation="relu"),
                                Dense(output_units, activation=activation)
                            ])
                            model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
                            model.fit(X_train, y_train, epochs=3, verbose=0)
                            pred = model.predict(X_test)
                            if n_classes > 2:
                                pred = np.argmax(pred, axis=1)
                            else:
                                pred = (pred.ravel()>0.5).astype(int)
                        else:
                            st.warning(f"{m} skipped (unsupported or TF not available)")
                            continue

                        metrics = compute_supervised_metrics(y_test, pred)
                        st.write(f"### {m} Results", metrics)
                        if "Confusion Matrix" in graphs:
                            plot_confusion(y_test, pred)
                        for g in graphs:
                            if g != "Confusion Matrix":
                                plot_generic(pd.DataFrame(X_test), g, title=f"{m}-{g}")
                    except Exception as e:
                        st.warning(f"{m} failed: {traceback.format_exc()}")

            else:
                st.error("Text/Image dataset handling is placeholder in this version.")

        except Exception as e:
            st.error("Run block failed!")
            st.text(traceback.format_exc())

except Exception as e:
    st.error("An unexpected error occurred while running the app!")
    st.text(traceback.format_exc())
