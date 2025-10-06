# Required libraires for data cleaning

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)

# Helper utilities

st.set_page_config(page_title="Data Cleaning App", layout="wide")

def download_df_as_csv(df: pd.DataFrame, filename="data.csv"):
    b = BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

def small_hist(df, col, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 2.4))
    sns.histplot(df[col].dropna(), bins=20, kde=True, ax=ax)
    ax.set_title(col, fontsize=9)
    ax.tick_params(axis='x', labelrotation=20, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    return ax.get_figure()

def small_bar(df, col, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4,2.4))
    vc = df[col].value_counts().nlargest(8)
    sns.barplot(x=vc.values, y=vc.index, ax=ax)
    ax.set_title(col, fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    return ax.get_figure()

def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


# Session state init

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df" not in st.session_state:
    st.session_state.df = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []

# Main Tabs (Dashboard, Help, Assistant)

tab = st.sidebar.radio("📌 Navigation", ["Dashboard", "Help Info", "Assistant"])

# DASHBOARD TAB
# ======================================================================
if tab == "Dashboard":
    st.title("Data Cleaning & Quick-ML App")
    st.markdown("Follow sidebar to clean data step-by-step, then train a simple model. If you want compact graphs, toggle below.")

    
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error("Failed to read CSV: " + str(e))
                st.stop()
            st.session_state.df_raw = df.copy()
            if st.session_state.df is None:
                st.session_state.df = df.copy()
        else:
            if st.session_state.df_raw is None:
                st.info("Upload a CSV to start (try insurance.csv or any dataset).")
                st.stop()

        st.subheader("Data preview (first 200 rows)")
        st.dataframe(st.session_state.df.head(200), height=300)

    with col2:
        st.subheader("Quick info")
        if st.session_state.df is not None:
            dfcur = st.session_state.df
            st.write("Shape:", dfcur.shape)
            st.write("Columns:", len(dfcur.columns))
            st.write("Numeric cols:", len(dfcur.select_dtypes(include=np.number).columns))
            st.write("Categorical cols:", len(dfcur.select_dtypes(include='object').columns))

    
    # Sidebar: cleaning controls
    # ---------------------------
    st.sidebar.header("Cleaning & Preprocessing (step-by-step)")

    # 1) Missing value analysis + drop threshold
    st.sidebar.subheader("1. Missing values")
    df_for_null = st.session_state.df if st.session_state.df is not None else pd.DataFrame()
    null_summary = pd.DataFrame({
        "column": df_for_null.columns,
        "missing_count": df_for_null.isna().sum(),
        "missing_pct": (df_for_null.isna().mean()*100).round(2),
        "dtype": [str(t) for t in df_for_null.dtypes]
    }).sort_values("missing_pct", ascending=False)

    st.sidebar.dataframe(null_summary, height=220)

    drop_thresh = st.sidebar.slider("Drop columns with missing% >= ", 0.0, 100.0, 80.0)
    if st.sidebar.button("Drop selected (threshold)"):
        drop_cols = list(null_summary[null_summary["missing_pct"] >= drop_thresh]["column"])
        if drop_cols:
            st.session_state.df = st.session_state.df.drop(columns=drop_cols)
            st.success(f"Dropped columns: {drop_cols}")
            st.rerun()
        else:
            st.info("No columns meet threshold")

    # Per-column imputation
    st.sidebar.markdown("---")
    st.sidebar.subheader("Per-column imputation")
    cols_with_nulls = [c for c in st.session_state.df.columns if st.session_state.df[c].isna().sum() > 0]
    if cols_with_nulls:
        st.sidebar.write(f"{len(cols_with_nulls)} columns have nulls")
        impute_map = {}
        for c in cols_with_nulls:
            if pd.api.types.is_numeric_dtype(st.session_state.df[c]):
                strat = st.sidebar.selectbox(f"{c} (numeric)", ("leave", "mean", "median", "constant"), key=f"imp_num_{c}")
                if strat == "constant":
                    val = st.sidebar.number_input(f"Const value for {c}", key=f"const_num_{c}")
                    impute_map[c] = (strat, val)
                else:
                    impute_map[c] = (strat, None)
            else:
                strat = st.sidebar.selectbox(f"{c} (categorical)", ("leave", "most_frequent", "constant"), key=f"imp_cat_{c}")
                if strat == "constant":
                    val = st.sidebar.text_input(f"Const (text) for {c}", key=f"const_cat_{c}")
                    impute_map[c] = (strat, val)
                else:
                    impute_map[c] = (strat, None)

        if st.sidebar.button("Apply imputations"):
            for col, (strategy, val) in impute_map.items():
                if strategy == "leave":
                    continue
                if strategy in ("mean", "median", "most_frequent"):
                    imp = SimpleImputer(strategy=strategy if strategy != "most_frequent" else "most_frequent")
                else:
                    imp = SimpleImputer(strategy="constant", fill_value=val)
                st.session_state.df[[col]] = imp.fit_transform(st.session_state.df[[col]])
            st.success("Applied imputations")
            st.rerun()
    else:
        st.sidebar.write("No nulls detected")

    # 2) Encoding
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Encoding")
    cat_cols = [c for c in st.session_state.df.columns if st.session_state.df[c].dtype == "object" or str(st.session_state.df[c].dtype).startswith("category")]
    st.sidebar.write("Detected categorical columns:", len(cat_cols))
    encode_select = st.sidebar.multiselect("Choose categorical columns to encode", cat_cols)
    enc_type = st.sidebar.radio("Encoding type", ("label", "onehot"))
    if st.sidebar.button("Apply encoding"):
        if not encode_select:
            st.sidebar.info("Select columns to encode")
        else:
            if enc_type == "label":
                for c in encode_select:
                    le = LabelEncoder()
                    st.session_state.df[c] = le.fit_transform(st.session_state.df[c].astype(str))
            else:
                st.session_state.df = pd.get_dummies(st.session_state.df, columns=encode_select, drop_first=False)
            st.success("Encoding applied")
            st.rerun()

    # 3) Scaling
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Scaling")
    numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
    scale_select = st.sidebar.multiselect("Select numeric columns to scale", numeric_cols)
    scale_type = st.sidebar.selectbox("Scaler", ("none", "standard", "minmax"))
    if st.sidebar.button("Apply scaling"):
        if scale_type == "none" or not scale_select:
            st.sidebar.info("No scaling applied")
        else:
            if scale_type == "standard":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            st.session_state.df[scale_select] = scaler.fit_transform(st.session_state.df[scale_select])
            st.success(f"Applied {scale_type} scaler")
            st.rerun()

    # Quick download cleaned CSV
    st.sidebar.markdown("---")
    if st.sidebar.button("Download cleaned CSV"):
        b = download_df_as_csv(st.session_state.df, "cleaned.csv")
        st.sidebar.download_button("Download CSV", data=b, file_name="cleaned.csv", mime="text/csv")

    # ---------------------------
    # Compact EDA area
    # ---------------------------
    st.markdown("---")
    st.subheader("Compact EDA (optional)")
    show_eda = st.checkbox("Show small graphs (hist / bar / corr)", value=False)
    if show_eda:
        dfc = st.session_state.df
        num_cols = dfc.select_dtypes(include=np.number).columns.tolist()
        cat_cols = dfc.select_dtypes(include="object").columns.tolist()

        # numeric histograms (up to 4)
        if num_cols:
            st.write("Numeric distributions (small):")
            cols_to_plot = num_cols[:4]
            cols_count = len(cols_to_plot)
            fig, axs = plt.subplots(1, cols_count, figsize=(4*cols_count, 2.6))
            if cols_count == 1:
                axs = [axs]
            for ax, col in zip(axs, cols_to_plot):
                sns.histplot(dfc[col].dropna(), bins=20, kde=False, ax=ax, color="#2b8cbe")
                ax.set_title(col, fontsize=10)
                ax.tick_params(axis='x', labelrotation=20, labelsize=8)
            st.pyplot(fig)

        # categorical small bars (up to 4)
        if cat_cols:
            st.write("Categorical counts (small):")
            cols_to_plot = cat_cols[:4]
            fig, axs = plt.subplots(1, len(cols_to_plot), figsize=(4*len(cols_to_plot), 2.6))
            if len(cols_to_plot) == 1:
                axs = [axs]
            for ax, col in zip(axs, cols_to_plot):
                vc = dfc[col].value_counts().nlargest(6)
                sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="muted")
                ax.set_title(col, fontsize=10)
                ax.tick_params(axis='y', labelsize=8)
            st.pyplot(fig)

        # correlation (small)
        if len(num_cols) >= 2:
            st.write("Correlation (numeric):")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(dfc[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            st.pyplot(fig)

    
    # Train / Test split + modeling
    # ---------------------------
    st.markdown("---")
    st.subheader("Train & Evaluate")

    all_cols = st.session_state.df.columns.tolist()
    target_col = st.selectbox("Select target column (for modeling)", options=all_cols)

    if target_col:
        test_size = st.slider("Test size (fraction)", min_value=0.05, max_value=0.5, value=0.2)
        random_state = st.number_input("Random state (int)", value=42, step=1)

        X = st.session_state.df.drop(columns=[target_col])
        y = st.session_state.df[target_col]

        # Detect problem type
        problem_type = "regression"
        if not pd.api.types.is_numeric_dtype(y) or y.nunique() < 20:
            problem_type = "classification"
        st.write("Detected problem type:", problem_type)

        # If classification and target not numeric, label encode the target
        if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y):
            y_enc = LabelEncoder().fit_transform(y.astype(str))
        else:
            y_enc = y.values

        # choose features
        feature_choice = st.radio("Feature set", ("Numeric only (recommended)", "All columns (must encode categorical first)"))
        if feature_choice == "Numeric only (recommended)":
            X_use = X.select_dtypes(include=np.number).copy()
        else:
            X_use = X.copy()

        st.write("Features shape:", X_use.shape)

        # MODEL selection
        if problem_type == "regression":
            model_name = st.selectbox("Model", ("LinearRegression", "Ridge", "Lasso", "DecisionTreeRegressor"))
            alpha = 1.0
            max_depth = None
            if model_name in ("Ridge", "Lasso"):
                alpha = st.number_input("alpha (regularization)", min_value=0.0, value=1.0, step=0.1)
            if model_name == "DecisionTreeRegressor":
                md = st.number_input("max_depth (0 for None)", min_value=0, value=5, step=1)
                max_depth = None if md == 0 else int(md)
        else:
            model_name = st.selectbox("Model", ("LogisticRegression", "DecisionTreeClassifier"))
            C = 1.0
            max_depth = None
            if model_name == "LogisticRegression":
                C = st.number_input("C (inverse regularization)", min_value=0.01, value=1.0, step=0.01)
            if model_name == "DecisionTreeClassifier":
                md = st.number_input("max_depth (0 for None)", min_value=0, value=5, step=1)
                max_depth = None if md == 0 else int(md)

        if st.button("Train model"):
            if X_use.shape[1] == 0:
                st.error("No numeric features found. Encode categorical columns or choose 'All columns' after encoding.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_use, y_enc, test_size=test_size, random_state=int(random_state))
                try:
                    if problem_type == "regression":
                        if model_name == "LinearRegression":
                            model = LinearRegression()
                        elif model_name == "Ridge":
                            model = Ridge(alpha=alpha)
                        elif model_name == "Lasso":
                            model = Lasso(alpha=alpha)
                        else:
                            model = DecisionTreeRegressor(max_depth=max_depth)
                        model.fit(X_train, y_train)
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        train_mae = mean_absolute_error(y_train, y_pred_train)
                        train_mse = mean_squared_error(y_train, y_pred_train)
                        train_r2 = r2_score(y_train, y_pred_train)
                        test_mae = mean_absolute_error(y_test, y_pred_test)
                        test_mse = mean_squared_error(y_test, y_pred_test)
                        test_r2 = r2_score(y_test, y_pred_test)

                        st.success("Model trained (regression)")
                        st.write("Train MAE:", round(train_mae,4), " MSE:", round(train_mse,4), " R2:", round(train_r2,4))
                        st.write("Test  MAE:", round(test_mae,4), " MSE:", round(test_mse,4), " R2:", round(test_r2,4))

                        fig, ax = plt.subplots(figsize=(4.5,3))
                        ax.scatter(y_test, y_pred_test, alpha=0.6, s=18)
                        ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', linewidth=1)
                        ax.set_xlabel("Actual")
                        ax.set_ylabel("Predicted")
                        ax.set_title("Predicted vs Actual (test)")
                        st.pyplot(fig)

                    else:
                        if model_name == "LogisticRegression":
                            model = LogisticRegression(max_iter=2000, C=C)
                        else:
                            model = DecisionTreeClassifier(max_depth=max_depth)
                        model.fit(X_train, y_train)
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)

                        train_acc = accuracy_score(y_train, y_pred_train)
                        test_acc = accuracy_score(y_test, y_pred_test)
                        st.success("Model trained (classification)")
                        st.write("Train accuracy:", round(train_acc,4))
                        st.write("Test accuracy:", round(test_acc,4))

                        labels = np.unique(y_test)
                        cm = confusion_matrix(y_test, y_pred_test)
                        st.pyplot(plot_confusion(cm, labels))

                        st.text("Classification report (test):")
                        st.text(classification_report(y_test, y_pred_test))

                        if len(labels) == 2:
                            try:
                                y_prob = model.predict_proba(X_test)[:,1]
                                auc = roc_auc_score(y_test, y_prob)
                                st.write("ROC AUC (test):", round(auc,4))
                            except Exception:
                                st.info("ROC AUC not available for this model.")

                    out = X_test.copy()
                    out["actual"] = y_test
                    out["predicted"] = y_pred_test
                    st.session_state.predictions = out
                    b = download_df_as_csv(out, "predictions.csv")
                    st.download_button("Download predictions CSV", data=b, file_name="predictions.csv", mime="text/csv")

                except Exception as e:
                    st.error("Training failed: " + str(e))

    st.markdown("---")
    st.write("Done! 🎉 You can restart from uploading new dataset.")

# HELP INFO TAB
# ======================================================================
elif tab == "Help Info":
    st.title("📖 Help & Instructions")
    st.markdown("""
### How to use this app step-by-step:

1. **Upload a CSV file** from your computer.  
   - It should be a clean CSV (comma-separated).
   - The app will preview first 200 rows.

2. **Check missing values**  
   - Sidebar shows percentage missing for each column.  
   - Drop columns with too many missing values (slider).  
   - Fill missing values column by column using mean, median, most_frequent, or constant.

3. **Encode categorical data**  
   - Choose columns and encode as **Label** (numbers) or **One-Hot** (binary).  
   - This is needed for ML models.

4. **Scale numeric features**  
   - Use Standard (z-score) or MinMax scaling if needed.

5. **EDA (exploratory data analysis)**  
   - Optionally view compact histograms, bar charts, and correlation heatmaps.

6. **Train a model**  
   - Select the target column.  
   - App auto-detects problem type: regression or classification.  
   - Choose a model (Linear, Ridge, Lasso, Logistic, DecisionTree, etc.).  
   - Adjust hyperparameters.  
   - Train/test split is applied automatically.  

7. **View results**  
   - Regression → metrics: MAE, MSE, R², plus scatter plot.  
   - Classification → accuracy, confusion matrix, report, ROC AUC.  
   - Download predictions as CSV.

---

### Notes
- For categorical targets, the app automatically label-encodes them.  
- Try to encode your categorical inputs before training.  
- If you get errors: check data types and missing values.  

Good luck! 🚀
    """)

# ASSISTANT TAB
# ======================================================================
elif tab == "Assistant":
    st.title("🤖 Assistant")
    st.markdown("Ask me about **data cleaning, preprocessing, or ML steps** in this app.")

    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"🧑 **You:** {chat['content']}")
        else:
            st.markdown(f"🤖 **Assistant:** {chat['content']}")

    user_msg = st.text_input("Type your question here...")
    if st.button("Send"):
        if user_msg.strip():
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            # dummy answer (can replace with OpenAI API if wanted)
            answer = "This is a placeholder assistant. Example Q&A:\n- Missing values? → Use sidebar options.\n- Encoding? → Apply label/onehot encoding.\n- Scaling? → Use Standard or MinMax.\n- Training? → Choose target and model.\n\nFor advanced answers, connect OpenAI API here."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
