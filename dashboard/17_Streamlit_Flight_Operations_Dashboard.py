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
    options=sorted(occ_df["origin_country"].dropna().unique()),
    default=[]
)

if len(selected_country) == 0:
    selected_country = occ_df["origin_country"].dropna().unique()

filtered_df = occ_df[
    (occ_df["flight_phase"].isin(selected_phase)) &
    (occ_df["Hybrid_Risk_Category"].isin(selected_risk)) &
    (occ_df["OCC_Alert_Level"].isin(selected_alert)) &
    (occ_df["origin_country"].isin(selected_country))
]

# ============================================================
# Risk Analysis Dashboard
# ============================================================

st.header("📈 Flight Risk Analysis")

col1, col2 = st.columns(2)

# ------------------------------------------------------------
# Hybrid Risk Score Distribution
# ------------------------------------------------------------

with col1:

    fig = px.histogram(
        filtered_df,
        x="Hybrid_Flight_Risk_Score",
        nbins=30,
        title="Hybrid Flight Risk Score Distribution"
    )

    fig.update_layout(
        xaxis_title="Hybrid Flight Risk Score",
        yaxis_title="Number of Flights"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Hybrid Risk Category
# ------------------------------------------------------------

with col2:

    risk_counts = (
        filtered_df["Hybrid_Risk_Category"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = ["Risk Category", "Flights"]

    fig = px.pie(
        risk_counts,
        names="Risk Category",
        values="Flights",
        title="Hybrid Risk Category Distribution",
        hole=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

col3, col4 = st.columns(2)

# ------------------------------------------------------------
# OCC Alert Levels
# ------------------------------------------------------------

with col3:

    alert_counts = (
        filtered_df["OCC_Alert_Level"]
        .value_counts()
        .reset_index()
    )

    alert_counts.columns = ["Alert Level", "Flights"]

    fig = px.bar(
        alert_counts,
        x="Alert Level",
        y="Flights",
        title="OCC Alert Levels"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Flight Phase
# ------------------------------------------------------------

with col4:

    phase_counts = (
        filtered_df["flight_phase"]
        .value_counts()
        .reset_index()
    )

    phase_counts.columns = ["Flight Phase", "Flights"]

    fig = px.bar(
        phase_counts,
        x="Flight Phase",
        y="Flights",
        title="Flight Phase Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# Information Overload Mitigation
# ============================================================

st.header("🛡️ Information Overload Mitigation")

# ------------------------------------------------------------
# Calculate Metrics
# ------------------------------------------------------------

total_flights = len(filtered_df)

routine_flights = filtered_df[
    filtered_df["OCC_Alert_Level"] == "Routine"
].shape[0]

medium_alerts = filtered_df[
    filtered_df["OCC_Alert_Level"] == "Medium"
].shape[0]

high_alerts = filtered_df[
    filtered_df["OCC_Alert_Level"] == "High"
].shape[0]

critical_alerts = filtered_df[
    filtered_df["OCC_Alert_Level"] == "Critical"
].shape[0]

priority_flights = high_alerts + critical_alerts

workload_reduction = (
    (routine_flights / total_flights) * 100
    if total_flights > 0 else 0
)

# ------------------------------------------------------------
# KPI Cards
# ------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric("Flights Analysed", f"{total_flights:,}")

with c2:
    st.metric("Routine Flights", f"{routine_flights:,}")

with c3:
    st.metric("High + Critical", f"{priority_flights:,}")

with c4:
    st.metric("Critical Alerts", f"{critical_alerts:,}")

with c5:
    st.metric(
        "Workload Reduction",
        f"{workload_reduction:.2f}%"
    )

st.divider()

# ============================================================
# Flights Requiring Immediate Attention
# ============================================================

st.subheader("🚨 Flights Requiring Immediate Attention")

priority_df = filtered_df[
    filtered_df["OCC_Alert_Level"].isin(["High", "Critical"])
].copy()

priority_df = priority_df.sort_values(
    by="Flight_Priority_Rank"
)

columns = [
    "Flight_Priority_Rank",
    "callsign",
    "origin_country",
    "Hybrid_Flight_Risk_Score",
    "Hybrid_Risk_Category",
    "OCC_Alert_Level",
    "OCC_Recommendation"
]

st.dataframe(
    priority_df[columns],
    use_container_width=True,
    hide_index=True
)

st.divider()