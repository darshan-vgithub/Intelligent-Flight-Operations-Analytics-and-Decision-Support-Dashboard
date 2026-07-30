# ============================================================
# Intelligent Flight Operations Analytics Dashboard
# Dissertation Project
#
# Title:
# Intelligent Flight Operations Analytics and Decision Support
# Dashboard with Hybrid AI-Based Risk Assessment and
# Information Overload Mitigation
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Flight Operations Analytics Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# Dashboard Title
# ------------------------------------------------------------

st.title("✈️ Intelligent Flight Operations Analytics Dashboard")

st.markdown("""
### Hybrid AI-Based Risk Assessment and Information Overload Mitigation

This dashboard supports operational decision-making for an Operations Control Centre (OCC).

The system integrates:

- Gaussian Mixture Model (Behaviour Analysis)
- ELM Autoencoder (Anomaly Detection)
- Hybrid AI Risk Assessment
- Smart Alert Prioritisation
- Explainable AI (SHAP)

to identify high-risk flights, prioritise operational alerts, and reduce information overload.
""")

st.divider()

# ------------------------------------------------------------
# Project Paths
# ------------------------------------------------------------

# Folder containing this Python file
CURRENT_DIR = Path(__file__).resolve().parent

# Project root folder
PROJECT_DIR = CURRENT_DIR.parent

# data/final folder
DATA_DIR = PROJECT_DIR / "data" / "final"

# ------------------------------------------------------------
# Show paths for debugging
# ------------------------------------------------------------


# ------------------------------------------------------------
# Load Datasets
# ------------------------------------------------------------

@st.cache_data
def load_data():

    occ_file = DATA_DIR / "occ_decision_support_dataset.csv"
    hybrid_file = DATA_DIR / "hybrid_ai_risk_assessment.csv"
    summary_file = DATA_DIR / "dashboard_summary.csv"


    occ_df = pd.read_csv(occ_file)
    hybrid_df = pd.read_csv(hybrid_file)
    summary_df = pd.read_csv(summary_file)

    return occ_df, hybrid_df, summary_df


occ_df, hybrid_df, summary_df = load_data()

st.success("✅ Datasets loaded successfully!")

# ============================================================
# Executive KPI Dashboard
# ============================================================

st.header("📊 Executive Overview")

# ------------------------------------------------------------
# Calculate KPIs
# ------------------------------------------------------------

total_flights = len(occ_df)

average_risk = occ_df["Hybrid_Flight_Risk_Score"].mean()

high_risk = occ_df[
    occ_df["Hybrid_Risk_Category"].isin(["High", "Critical"])
].shape[0]

critical_alerts = occ_df[
    occ_df["OCC_Alert_Level"] == "Critical"
].shape[0]

# ------------------------------------------------------------
# KPI Cards
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "✈️ Total Flights",
        f"{total_flights:,}"
    )

with col2:
    st.metric(
        "⚠️ Average Risk Score",
        f"{average_risk:.2f}"
    )

with col3:
    st.metric(
        "🚨 High Risk Flights",
        f"{high_risk:,}"
    )

with col4:
    st.metric(
        "🔴 Critical Alerts",
        f"{critical_alerts:,}"
    )

st.divider()

# ============================================================
# Sidebar Filters
# ============================================================

st.sidebar.header("✈ Flight Filters")

selected_phase = st.sidebar.multiselect(
    "Flight Phase",
    sorted(occ_df["flight_phase"].dropna().unique()),
    default=sorted(occ_df["flight_phase"].dropna().unique())
)

selected_risk = st.sidebar.multiselect(
    "Hybrid Risk Category",
    sorted(occ_df["Hybrid_Risk_Category"].dropna().unique()),
    default=sorted(occ_df["Hybrid_Risk_Category"].dropna().unique())
)

selected_alert = st.sidebar.multiselect(
    "OCC Alert Level",
    sorted(occ_df["OCC_Alert_Level"].dropna().unique()),
    default=sorted(occ_df["OCC_Alert_Level"].dropna().unique())
)

selected_country = st.sidebar.multiselect(
    "Origin Country",
    sorted(occ_df["origin_country"].dropna().unique()),
    default=sorted(occ_df["origin_country"].dropna().unique())
)

filtered_df = occ_df[
    (occ_df["flight_phase"].isin(selected_phase)) &
    (occ_df["Hybrid_Risk_Category"].isin(selected_risk)) &
    (occ_df["OCC_Alert_Level"].isin(selected_alert)) &
    (occ_df["origin_country"].isin(selected_country))
]