import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ================= LOAD MODEL =================
model = joblib.load("final_model.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Variant Dashboard", layout="wide")

st.title("🧬 Variant Analysis & Prediction System")

# ================= SIDEBAR =================
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Predict"])

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df.columns = df.columns.str.strip()   # FIX for column issues
    return df

df = load_data()

# ================= DASHBOARD =================
if page == "Dashboard":

    st.header("📊 Data Dashboard")

    # ---- 1. Label Distribution ----
    st.subheader("Variant Classification Distribution")

    if "label" in df.columns:
        st.bar_chart(df["label"].value_counts())
    else:
        st.error("⚠️ 'label' column not found")

    # ---- 2. Allele Frequency ----
    st.subheader("Allele Frequency Distribution")

    if "Allele.Frequency" in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df["Allele.Frequency"], bins=30, ax=ax)
        st.pyplot(fig)

    # ---- 3. Correlation Heatmap ----
    st.subheader("Feature Correlation")

    numeric_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ---- 4. Feature Importance ----
    st.subheader("Model Feature Importance")

    try:
        importances = model.feature_importances_

        features = ["Allele.Frequency", "cadd", "revel_max",
                    "sift_max", "polyphen_max", "PPI_score"]

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

    except:
        st.warning("Feature importance not available")

# ================= PREDICTION =================
elif page == "Predict":

    st.header("🔍 Variant Prediction")

    st.markdown("Enter variant features:")

    allele_freq = st.number_input("Allele Frequency", value=0.001)
    cadd = st.number_input("CADD Score", value=10.0)
    revel = st.number_input("REVEL Score", value=0.5)
    sift = st.number_input("SIFT Score", value=0.05)
    polyphen = st.number_input("PolyPhen Score", value=0.5)
    ppi = st.number_input("PPI Score", value=100.0)

    if st.button("Predict"):

        input_data = np.array([[allele_freq, cadd, revel, sift, polyphen, ppi]])

        # Probability-based prediction
        proba = model.predict_proba(input_data)[0][1]

        if proba > 0.5:
            st.error(f"🔴 Pathogenic (Confidence: {proba:.2f})")
        else:
            st.success(f"🟢 Benign (Confidence: {1-proba:.2f})")

# ================= FOOTER =================
st.markdown("---")
st.caption("Developed for Bioinformatics Project – AP-4 Variant Analysis")
