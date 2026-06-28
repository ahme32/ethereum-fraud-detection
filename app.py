"""
Ethereum Fraud Detection - Streamlit App
==========================================
ML model يكتشف حسابات Ethereum المحتالة بناءً على سلوك المعاملات.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========== Page Configuration ==========
st.set_page_config(
    page_title="Ethereum Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Load Artifacts (مرة واحدة بس) ==========
@st.cache_resource
def load_artifacts():
    """تحميل الموديل والـ scaler والـ features list."""
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('selected_features.pkl')
    return model, scaler, features

model, scaler, FEATURES = load_artifacts()

# ========== Helper Functions ==========
def compute_engineered_features(raw_inputs):
    """
    حساب الـ features المهندَسة من المدخلات الخام.
    لازم نطبق نفس الحسابات اللي عملناها في الـ training.
    """
    d = dict(raw_inputs)

    # الـ features المهندسة
    d['sent_received_ratio'] = d['sent_tnx'] / (d['received_tnx'] + 1)
    d['active_days'] = d['time_diff_first_last_mins'] / (60 * 24)
    d['avg_tnx_per_day'] = d['total_transactions'] / (d['active_days'] + 1)
    d['received_density'] = d['received_tnx'] / (d['unique_received_from_addresses'] + 1)
    d['erc20_activity_score'] = d['has_erc20_sent_token'] + d['has_erc20_rec_token']
    d['is_active'] = int(d['time_diff_first_last_mins'] > 0)
    d['sent_density'] = d['sent_tnx'] / (d['unique_sent_to_addresses'] + 1)

    return d


def predict_fraud(input_dict):
    """
    التنبؤ بـ fraud من الـ inputs.
    Returns: (prediction, probability)
    """
    # نحول لـ DataFrame بترتيب الـ FEATURES
    df = pd.DataFrame([{f: input_dict.get(f, 0) for f in FEATURES}])

    # Scaling
    X_scaled = scaler.transform(df)

    # Prediction
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0, 1]

    return int(pred), float(proba)


# ========== UI ==========
st.title("🛡️ Ethereum Fraud Detection System")
st.markdown("##### كشف حسابات Ethereum المحتالة باستخدام Machine Learning")
st.markdown("---")

# Sidebar - معلومات الموديل
with st.sidebar:
    st.header("📊 Model Info")
    st.markdown("""
    **Algorithm:** XGBoost (Tuned)

    **Performance on Test Set:**
    - Accuracy: **98.88%**
    - Precision: **98.36%**
    - Recall: **96.56%**
    - F1-Score: **97.45%**
    - ROC-AUC: **99.77%**

    **Training:**
    - 9,841 Ethereum accounts
    - 20 selected features
    - SMOTE for imbalance
    - 5-Fold Cross-Validation

    ---
    🔗 [GitHub Repo](#)
    📧 [Contact](#)
    """)

    st.markdown("---")
    st.info("💡 **ملاحظة:** الموديل ده ML demo، مش financial advice.")

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Upload (CSV)", "ℹ️ About"])

# ========== TAB 1: Single Prediction ==========
with tab1:
    st.subheader("أدخل بيانات الحساب")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📤 Transaction Volume**")
        sent_tnx = st.number_input("Sent Transactions", min_value=0, value=50, step=1)
        received_tnx = st.number_input("Received Transactions", min_value=0, value=40, step=1)
        total_transactions = st.number_input("Total Transactions", min_value=0, value=100, step=1)
        unique_received_from_addresses = st.number_input("Unique Received From", min_value=0, value=20, step=1)
        unique_sent_to_addresses = st.number_input("Unique Sent To", min_value=0, value=20, step=1)

    with col2:
        st.markdown("**💰 Ether Values**")
        total_ether_received = st.number_input("Total Ether Received", min_value=0.0, value=5.0, step=0.1, format="%.4f")
        max_value_received = st.number_input("Max Value Received", min_value=0.0, value=2.0, step=0.1, format="%.4f")
        avg_val_received = st.number_input("Avg Value Received", min_value=0.0, value=0.5, step=0.01, format="%.4f")
        time_diff_first_last_mins = st.number_input("Time Diff First-Last (mins)", min_value=0, value=10000, step=100)

    with col3:
        st.markdown("**🪙 ERC20 Activity**")
        has_erc20_sent_token = st.selectbox("Has ERC20 Sent Tokens?", [0, 1], index=0)
        has_erc20_rec_token = st.selectbox("Has ERC20 Received Tokens?", [0, 1], index=1)
        total_erc20_tnxs = st.number_input("Total ERC20 Transactions", min_value=0, value=10, step=1)
        erc20_total_ether_received = st.number_input("ERC20 Total Ether Received", min_value=0.0, value=1.0, step=0.1, format="%.4f")

    # Advanced (collapsed)
    with st.expander("🔧 Advanced Features (Optional)"):
        col4, col5 = st.columns(2)
        with col4:
            avg_min_between_received_tnx = st.number_input("Avg Min Between Received Tnx", min_value=0.0, value=100.0)
            erc20_min_val_rec = st.number_input("ERC20 Min Value Received", min_value=0.0, value=0.0, format="%.4f")
            erc20_max_val_rec = st.number_input("ERC20 Max Value Received", min_value=0.0, value=0.5, format="%.4f")
        with col5:
            erc20_uniq_rec_addr = st.number_input("ERC20 Unique Rec Addr", min_value=0, value=3)
            erc20_uniq_rec_token_name = st.number_input("ERC20 Unique Rec Token Name", min_value=0, value=2)

    # Predict button
    st.markdown("---")
    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):

        raw_inputs = {
            'sent_tnx': sent_tnx,
            'received_tnx': received_tnx,
            'total_transactions': total_transactions,
            'unique_received_from_addresses': unique_received_from_addresses,
            'unique_sent_to_addresses': unique_sent_to_addresses,
            'total_ether_received': total_ether_received,
            'max_value_received': max_value_received,
            'avg_val_received': avg_val_received,
            'time_diff_first_last_mins': time_diff_first_last_mins,
            'has_erc20_sent_token': has_erc20_sent_token,
            'has_erc20_rec_token': has_erc20_rec_token,
            'total_erc20_tnxs': total_erc20_tnxs,
            'erc20_total_ether_received': erc20_total_ether_received,
            'avg_min_between_received_tnx': avg_min_between_received_tnx,
            'erc20_min_val_rec': erc20_min_val_rec,
            'erc20_max_val_rec': erc20_max_val_rec,
            'erc20_uniq_rec_addr': erc20_uniq_rec_addr,
            'erc20_uniq_rec_token_name': erc20_uniq_rec_token_name,
        }

        # حساب الـ features المهندسة
        full_inputs = compute_engineered_features(raw_inputs)

        # التنبؤ
        pred, proba = predict_fraud(full_inputs)

        # عرض النتيجة
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if pred == 1:
                st.error("### 🚨 FRAUD DETECTED")
            else:
                st.success("### ✅ NORMAL ACCOUNT")

        with col_b:
            st.metric("Fraud Probability", f"{proba*100:.2f}%")

        with col_c:
            confidence = max(proba, 1-proba) * 100
            st.metric("Confidence", f"{confidence:.2f}%")

        # Progress bar
        st.progress(proba)

        # Interpretation
        if proba > 0.8:
            st.error("🔴 **High Risk:** الحساب ده على الأرجح محتال. ننصح بإجراء تحقيق إضافي.")
        elif proba > 0.5:
            st.warning("🟡 **Medium Risk:** ممكن يكون محتال. راقب الحساب.")
        elif proba > 0.2:
            st.info("🔵 **Low Risk:** أغلب الظن طبيعي، لكن خلي عينك عليه.")
        else:
            st.success("🟢 **Safe:** الحساب يبدو طبيعي تماماً.")


# ========== TAB 2: Batch Upload ==========
with tab2:
    st.subheader("رفع ملف CSV لفحص عدة حسابات")

    st.info(f"""
    📋 **متطلبات الملف:**
    - CSV format
    - يجب أن يحتوي على الأعمدة الـ 20 المختارة:
    {', '.join(FEATURES[:10])}...
    """)

    uploaded_file = st.file_uploader("اختر ملف CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded: {df_upload.shape[0]} rows, {df_upload.shape[1]} columns")

            # Compute engineered features for each row
            df_expanded = pd.DataFrame([
                compute_engineered_features(row)
                for row in df_upload.to_dict('records')
            ])

            # تأكد إن الأعمدة المطلوبة موجودة
            missing = [f for f in FEATURES if f not in df_expanded.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                # Predict
                X = df_expanded[FEATURES]
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]

                result_df = df_upload.copy()
                result_df['Prediction'] = ['Fraud' if p == 1 else 'Normal' for p in predictions]
                result_df['Fraud_Probability'] = probabilities.round(4)

                # Summary
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Accounts", len(result_df))
                col2.metric("Fraud Detected", int(predictions.sum()))
                col3.metric("Fraud Rate", f"{predictions.mean()*100:.2f}%")

                st.dataframe(result_df, use_container_width=True)

                # Download
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Predictions",
                    csv,
                    "fraud_predictions.csv",
                    "text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ========== TAB 3: About ==========
with tab3:
    st.subheader("ℹ️ About this Project")

    st.markdown("""
    ### 🎯 Problem Statement
    في عالم العملات الرقمية، فيه محتالين بيستخدموا حسابات وهمية للاحتيال على الناس.
    المشروع ده بيستخدم Machine Learning لكشف حسابات Ethereum المحتالة بناءً على
    سلوك المعاملات.

    ### 📊 Dataset
    - **Source:** [Kaggle - Ethereum Fraud Detection](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
    - **Size:** 9,841 accounts × 51 features
    - **Target:** Binary (FLAG: 0 = Normal, 1 = Fraud)
    - **Imbalance:** 22% fraud

    ### 🔧 Pipeline
    1. **Data Cleaning:** Missing values, constant columns, type fixes
    2. **EDA:** 8+ types of visualizations
    3. **Feature Engineering:** 7 new engineered features
    4. **Feature Selection:** 3 methods (Filter, Wrapper, Embedded)
    5. **Modeling:** 5 algorithms compared (LR, DT, RF, XGB, KNN)
    6. **Tuning:** GridSearchCV on XGBoost
    7. **Validation:** 5-Fold Stratified CV

    ### 🏆 Best Model
    **XGBoost** with SMOTE oversampling:
    - F1-Score: **97.45%**
    - Recall: **96.56%** (نقدر نمسك 96 من كل 100 محتال)
    - Precision: **98.36%** (false alarms < 1%)

    ### 🛠️ Tech Stack
    `Python` `Pandas` `Scikit-learn` `XGBoost` `Imbalanced-learn` `Streamlit`

    ### 📚 Course
    Final Project - Epsilon AI Data Science Track
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Made with ❤️ using Streamlit | Epsilon AI Final Project"
    "</p>",
    unsafe_allow_html=True
)
