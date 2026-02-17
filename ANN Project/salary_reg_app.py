import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

st.set_page_config(page_title="Salary Regression", layout="wide")

st.title("Salary Regression — Train & Predict")

st.markdown("Upload a CSV with features and a numeric target (salary), or use the small sample dataset.")

uploaded = st.file_uploader("Upload CSV", type=["csv"]) 
use_sample = st.checkbox("Use sample dataset (YearsExperience -> Salary)")

if use_sample and uploaded is not None:
    st.warning("Using uploaded file takes precedence over sample unless you clear the upload.")

if uploaded is None and not use_sample:
    st.info("No data selected — either upload a CSV or check 'Use sample dataset'.")

# Load data
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded `{uploaded.name}` — shape: {df.shape}")
else:
    if use_sample:
        df = pd.DataFrame({
            "YearsExperience": [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5],
            "Salary": [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111]
        })
        st.success("Using built-in sample dataset")
    else:
        df = None

if df is not None:
    st.dataframe(df.head())

    # let user choose target and features
    cols = df.columns.tolist()
    target = st.selectbox("Select target column (numeric)", options=cols, index=len(cols)-1)
    features = st.multiselect("Select feature columns", options=[c for c in cols if c!=target], default=[c for c in cols if c!=target])

    if len(features) == 0:
        st.warning("Select at least one feature column")
    else:
        test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2)
        model_type = st.selectbox("Model", ["LinearRegression", "RandomForest"])

        if st.button("Train model"):
            X = df[features].copy()
            y = df[target].copy()

            # Basic preprocessing: numeric / categorical handling with get_dummies
            X = pd.get_dummies(X, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if model_type == "LinearRegression":
                model = LinearRegression()
                model.fit(X_train, y_train)
            else:
                model = RandomForestRegressor(n_estimators=200, random_state=42)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("Metrics on test set")
            st.write({"MSE": float(mse), "MAE": float(mae), "R2": float(r2)})

            # Show coefficients or feature importances
            st.subheader("Model details")
            if model_type == "LinearRegression":
                coefs = pd.Series(model.coef_, index=X_train.columns).sort_values(key=abs, ascending=False)
                st.write(coefs)
            else:
                imps = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                st.write(imps)

            # Save model and columns for prediction
            joblib.dump({"model": model, "columns": X_train.columns.tolist()}, "salary_reg_model.joblib")
            st.success("Model trained and saved to `salary_reg_model.joblib`")

        st.markdown("---")
        st.subheader("Make a prediction using the trained model file")

        model_file = st.file_uploader("Upload trained model (.joblib)", type=["joblib"], key="model_uploader")
        if model_file is None:
            try:
                loaded = joblib.load("salary_reg_model.joblib")
            except Exception:
                loaded = None
        else:
            loaded = joblib.load(model_file)

        if loaded is not None:
            model = loaded["model"]
            columns = loaded["columns"]

            st.write("Model ready — provide inputs for prediction")

            # Build input widgets for each feature column used during training
            input_vals = {}
            for col in columns:
                # try to infer type from training dataframe if possible
                if col in df.columns:
                    # if original column exists, use same dtype
                    if pd.api.types.is_numeric_dtype(df[col]):
                        input_vals[col] = st.number_input(col, value=float(df[col].median()))
                    else:
                        opts = df[col].dropna().unique().tolist()
                        input_vals[col] = st.selectbox(col, opts)
                else:
                    # for dummy columns (like Gender_Male or Geography_Spain), use 0/1
                    input_vals[col] = st.selectbox(col, options=[0,1], index=0)

            if st.button("Predict"):
                x_input = pd.DataFrame([input_vals])
                # ensure all columns present
                x_input = x_input.reindex(columns=columns, fill_value=0)
                pred = model.predict(x_input)[0]
                st.success(f"Predicted {target}: {pred:.2f}")
        else:
            st.info("No trained model found — train one above or upload a `.joblib` file.")

else:
    st.stop()

st.markdown("---")
st.write("Run: `streamlit run salary_reg_app.py`")
