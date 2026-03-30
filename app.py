import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ================= CONFIG =================
st.set_page_config(page_title="Variant Dashboard", layout="wide")

# ================= LOAD =================
model = joblib.load("final_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("final_dataset.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ================= HEADER =================
st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>
🧬 Variant Intelligence Dashboard
</h1>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
page = st.sidebar.radio("Navigation", ["📊 Dashboard", "🔍 Prediction"])

# ================= DASHBOARD =================
if page == "📊 Dashboard":

    st.markdown("## 📊 Data Insights")

    # ---- TABS ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution", "Feature Analysis", "Correlation", "Model Insights"
    ])

    # ================= TAB 1 =================
    with tab1:
        col1, col2 = st.columns(2)

        # Label distribution
        with col1:
            if "label" in df.columns:
                fig = px.pie(
                    df,
                    names="label",
                    title="Variant Classification",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)

        # Allele frequency histogram
        with col2:
            fig = px.histogram(
                df,
                x="Allele.Frequency",
                nbins=40,
                title="Allele Frequency Distribution",
                color_discrete_sequence=["#00C9A7"]
            )
            st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 2 =================
    with tab2:

        col1, col2 = st.columns(2)

        # CADD vs REVEL
        with col1:
            fig = px.scatter(
                df,
                x="cadd",
                y="revel_max",
                color="label",
                title="CADD vs REVEL",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)

        # SIFT vs PolyPhen
        with col2:
            fig = px.scatter(
                df,
                x="sift_max",
                y="polyphen_max",
                color="label",
                title="SIFT vs PolyPhen",
                color_continuous_scale="plasma"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 3 =================
    with tab3:

        numeric_df = df.select_dtypes(include=np.number)

        fig = px.imshow(
            numeric_df.corr(),
            text_auto=True,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ================= TAB 4 =================
    with tab4:

        try:
            importances = model.feature_importances_

            features = [
                "Allele.Frequency", "cadd", "revel_max",
                "sift_max", "polyphen_max", "PPI_score"
            ]

            importance_df = pd.DataFrame({
                "Feature": features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=True)

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation='h',
                color="Importance",
                color_continuous_scale="viridis",
                title="Feature Importance"
            )

            st.plotly_chart(fig, use_container_width=True)

        except:
            st.warning("Model does not support feature importance")

# ================= PREDICTION =================
elif page == "🔍 Prediction":

    st.markdown("## 🔍 Variant Prediction")

    col1, col2 = st.columns(2)

    with col1:
        allele_freq = st.number_input("Allele Frequency", value=0.001)
        cadd = st.number_input("CADD Score", value=10.0)
        revel = st.number_input("REVEL Score", value=0.5)

    with col2:
        sift = st.number_input("SIFT Score", value=0.05)
        polyphen = st.number_input("PolyPhen Score", value=0.5)
        ppi = st.number_input("PPI Score", value=100.0)

    if st.button("Predict"):

        input_data = np.array([[allele_freq, cadd, revel, sift, polyphen, ppi]])

        proba = model.predict_proba(input_data)[0][1]

        if proba > 0.5:
            st.error(f"🔴 Pathogenic (Confidence: {proba:.2f})")
        else:
            st.success(f"🟢 Benign (Confidence: {1-proba:.2f})")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<center>✨ Advanced Bioinformatics Dashboard | AP-4 Variant Analysis</center>",
    unsafe_allow_html=True
)
