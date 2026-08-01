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

ASSETS_DIR = CURRENT_DIR / "assets"

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
# Operational Flight Map
# ============================================================

st.header("🗺 Operational Flight Map")

st.markdown("""
This map displays the geographical locations of all flights using their
latest ADS-B positions. Flights are coloured according to their Hybrid
Risk Category, enabling OCC operators to quickly identify high-risk
aircraft and their current locations.
""")

fig = px.scatter_mapbox(
    filtered_df,
    lat="latitude",
    lon="longitude",
    color="Hybrid_Risk_Category",
    hover_name="callsign",
    hover_data={
        "origin_country": True,
        "Hybrid_Flight_Risk_Score": ":.2f",
        "OCC_Alert_Level": True,
        "velocity": ":.1f",
        "baro_altitude": ":.0f",
        "latitude": False,
        "longitude": False,
    },
    zoom=1.5,
    height=650,
    color_discrete_map={
        "Low": "green",
        "Medium": "orange",
        "High": "red",
        "Critical": "darkred"
    }
)

fig.update_layout(
    mapbox_style="open-street-map",
    margin=dict(l=0, r=0, t=0, b=0)
)

st.plotly_chart(fig, use_container_width=True)

st.info("""
**Operational Flight Map**

• Displays the current positions of all monitored flights.

• Marker colours represent the Hybrid AI Risk Category.

• Hover over a flight to view operational details including
  callsign, origin country, alert level, altitude and speed.

This provides OCC personnel with geographical situational awareness,
complementing the AI-driven risk assessment and decision support.
""")

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

# ============================================================
# Decision Support Dashboard
# ============================================================

st.header("🎯 Decision Support Dashboard")

st.markdown("""
This section converts the Hybrid AI Risk Assessment into actionable
operational recommendations for the Operations Control Centre (OCC).
""")

# ------------------------------------------------------------
# Recommendation Summary
# ------------------------------------------------------------

recommendation_counts = (
    filtered_df["OCC_Recommendation"]
    .value_counts()
    .reset_index()
)

recommendation_counts.columns = [
    "Recommendation",
    "Flights"
]

col1, col2 = st.columns([1, 2])

# ------------------------------------------------------------
# Recommendation Table
# ------------------------------------------------------------

with col1:

    st.subheader("Recommendation Summary")

    st.dataframe(
        recommendation_counts,
        hide_index=True,
        use_container_width=True
    )

# ------------------------------------------------------------
# Recommendation Chart
# ------------------------------------------------------------

with col2:

    fig = px.bar(
        recommendation_counts,
        x="Recommendation",
        y="Flights",
        color="Recommendation",
        title="Operational Recommendations"
    )

    fig.update_layout(
        xaxis_title="Recommendation",
        yaxis_title="Number of Flights",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.divider()

# ============================================================
# Operational Action Summary
# ============================================================

st.subheader("🚦 Operational Action Summary")

# ------------------------------------------------------------
# Recommendation Mapping
# ------------------------------------------------------------

recommendation_mapping = {
    "Routine monitoring": "🟢 Routine",
    "Continue enhanced monitoring": "🟡 Enhanced",
    "Notify OCC supervisor and increase monitoring": "🟠 Supervisor",
    "Immediate operational intervention required": "🔴 Immediate"
}

recommendation_counts = (
    filtered_df["OCC_Recommendation"]
    .value_counts()
    .rename_axis("Recommendation")
    .reset_index(name="Flights")
)

recommendation_counts["Dashboard_Label"] = (
    recommendation_counts["Recommendation"]
    .map(recommendation_mapping)
)

summary = dict(
    zip(
        recommendation_counts["Dashboard_Label"],
        recommendation_counts["Flights"]
    )
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("🟢 Routine", summary.get("🟢 Routine", 0))
col2.metric("🟡 Enhanced", summary.get("🟡 Enhanced", 0))
col3.metric("🟠 Supervisor", summary.get("🟠 Supervisor", 0))
col4.metric("🔴 Immediate", summary.get("🔴 Immediate", 0))

st.divider()

# ============================================================
# Flight Explorer
# ============================================================

st.header("🔍 Flight Explorer")

st.markdown("""
Search and inspect individual flights to view their operational status,
Hybrid AI Risk Assessment, alert level and recommended OCC action.
""")

# ------------------------------------------------------------
# Flight Selection
# ------------------------------------------------------------

flight_list = (
    filtered_df["callsign"]
    .dropna()
    .sort_values()
    .unique()
)

selected_callsign = st.selectbox(
    "Select Flight Callsign",
    flight_list
)

# ------------------------------------------------------------
# Selected Flight
# ------------------------------------------------------------

selected_flight = filtered_df[
    filtered_df["callsign"] == selected_callsign
].iloc[0]


# ------------------------------------------------------------
# Flight Information
# ------------------------------------------------------------

st.subheader("✈ Flight Information")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Callsign",
        selected_flight["callsign"]
    )

    st.metric(
        "ICAO24",
        selected_flight["icao24"]
    )

    st.metric(
        "Origin Country",
        selected_flight["origin_country"]
    )

with col2:

    st.metric(
        "Flight Phase",
        selected_flight["flight_phase"]
    )

    st.metric(
        "Risk Category",
        selected_flight["Hybrid_Risk_Category"]
    )

    st.metric(
        "Alert Level",
        selected_flight["OCC_Alert_Level"]
    )

with col3:

    st.metric(
        "Risk Score",
        round(selected_flight["Hybrid_Flight_Risk_Score"],2)
    )

    st.metric(
        "Priority Rank",
        int(selected_flight["Flight_Priority_Rank"])
    )

    st.metric(
        "Recommendation",
        selected_flight["OCC_Recommendation"]
    )

st.divider()


# ============================================================
# AI Assessment
# ============================================================

st.subheader("🧠 Hybrid AI Assessment")

c1, c2 = st.columns(2)

with c1:

    st.metric(
        "Behaviour Intelligence",
        round(selected_flight["Behavioural_Intelligence_Score"],2)
    )

    st.metric(
        "Operational Intelligence",
        round(selected_flight["Operational_Intelligence_Score"],2)
    )

with c2:

    st.metric(
        "GMM Result",
        selected_flight["GMM_Anomaly_Label"]
    )

    st.metric(
        "ELM Result",
        selected_flight["ELM_Anomaly_Label"]
    )

st.divider()




# ============================================================
# Explainable AI (SHAP)
# ============================================================

st.header("🧠 Explainable AI (SHAP)")

st.markdown("""
This section explains **why** the Hybrid AI model classified flights with
different operational risk levels. The SHAP (SHapley Additive exPlanations)
analysis improves model transparency by highlighting the contribution of
each feature to the Hybrid Flight Risk Score.
""")

st.divider()

st.subheader("📊 Global Feature Importance")

st.image(
    ASSETS_DIR / "shap_summary.png",
    caption="SHAP Summary Plot",
    use_container_width=True
)

st.info("""
The SHAP Summary Plot shows the global importance of features used by the
Hybrid AI Risk Assessment model.

Higher SHAP values indicate a greater contribution towards increasing
the Hybrid Flight Risk Score.

Key observations:

• Operational Complexity Index has the strongest influence.

• GMM Anomaly Score is one of the primary contributors.

• Flight Operational Score significantly affects risk.

• Vertical Rate contributes to abnormal flight behaviour.

• ELM Anomaly Score supports anomaly detection.
""")

st.divider()

st.subheader("📈 Feature Dependence")

st.image(
    ASSETS_DIR / "shap_dependence.png",
    caption="SHAP Dependence Plot",
    use_container_width=True
)

st.info("""
The dependence plot illustrates how changes in an operational feature
influence the Hybrid Flight Risk Score.

It highlights non-linear relationships between operational conditions
and predicted flight risk.
""")

st.divider()

st.subheader("💧 Local Flight Explanation")

st.image(
    ASSETS_DIR / "shap_waterfall.png",
    caption="SHAP Waterfall Plot",
    use_container_width=True
)

st.info("""
The waterfall plot explains an individual prediction by showing how each
feature increases or decreases the Hybrid Flight Risk Score.

This provides transparency for operational decision-making and helps
OCC personnel understand why a specific flight received its risk level.
""")

st.divider()

st.subheader("📌 Explainable AI Findings")

st.success("""
The Explainable AI analysis demonstrates that the Hybrid Flight Risk Score
is not generated by a black-box model.

The model primarily considers:

• Operational Complexity Index

• GMM Anomaly Score

• Flight Operational Score

• Vertical Rate

• ELM Anomaly Score

This confirms that the final operational risk assessment combines both
behavioural anomaly detection and operational intelligence, providing
transparent and trustworthy decision support for Operations Control
Centre (OCC) personnel.
""")

st.divider()

