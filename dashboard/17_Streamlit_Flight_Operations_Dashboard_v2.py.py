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

# ============================================================
# PROFESSIONAL DASHBOARD THEME
# ============================================================

st.markdown("""
<style>

/* ===============================
Google Font
=============================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* ===============================
Main Background
=============================== */

.stApp{
    background:#F8FAFC;
}

/* ===============================
Content Width
=============================== */

.block-container{
    padding-top:2rem;
    padding-bottom:3rem;
    padding-left:2.5rem;
    padding-right:2.5rem;
}

/* ===============================
Sidebar
=============================== */

[data-testid="stSidebar"]{

    background:#FFFFFF;

    border-right:1px solid #E5E7EB;

}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{

    color:#1E3A8A;

    font-weight:700;

}

/* ===============================
Titles
=============================== */

h1{

    color:#111827;

    font-size:44px;

    font-weight:800;

}

h2{

    color:#111827;

    font-weight:700;

}

h3{

    color:#374151;

    font-weight:600;

}

/* ===============================
Paragraphs
=============================== */

p{

    color:#4B5563;

    font-size:16px;

}

/* ===============================
Metric Cards
=============================== */

[data-testid="metric-container"]{

    background:white;

    border-radius:14px;

    border:1px solid #E5E7EB;

    padding:18px;

    box-shadow:0px 4px 10px rgba(0,0,0,0.05);

}

/* metric label */

[data-testid="stMetricLabel"]{

    font-size:15px;

    font-weight:600;

}

/* metric value */

[data-testid="stMetricValue"]{

    font-size:34px;

    font-weight:700;

    color:#111827;

}

/* ===============================
Dataframes
=============================== */

[data-testid="stDataFrame"]{

    border-radius:12px;

    border:1px solid #E5E7EB;

}

/* ===============================
Buttons
=============================== */

.stButton>button{

    border-radius:8px;

    background:#1E3A8A;

    color:white;

    border:none;

    font-weight:600;

}

.stButton>button:hover{

    background:#163172;

    color:white;

}

/* ===============================
Alerts
=============================== */

.stAlert{

    border-radius:12px;

}

/* ===============================
Divider
=============================== */

hr{

    margin-top:30px;

    margin-bottom:30px;

}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Dashboard Title
# ------------------------------------------------------------


st.markdown("""
#✈️ Intelligent Flight Operations Analytics Dashboard
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

st.caption(
    "🟢 System Status: Operational | All aviation datasets successfully loaded."
)

# ============================================================
# DASHBOARD COLOURS
# ============================================================

PRIMARY = "#1E3A8A"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#DC2626"
BACKGROUND = "#F8FAFC"
CARD = "#FFFFFF"

PLOT_TEMPLATE = "plotly_white"

# ============================================================
# PROFESSIONAL PLOTLY STYLE
# ============================================================

def style_plot(fig):

    fig.update_layout(

        template=PLOT_TEMPLATE,

        paper_bgcolor="white",

        plot_bgcolor="white",

        font=dict(

            family="Inter",

            size=14,

            color="#374151"

        ),

        title_font=dict(

            size=20,

            color="#111827"

        ),

        margin=dict(

            l=20,

            r=20,

            t=60,

            b=20

        ),

        legend_title="",

    )

    return fig
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
# Framework Evaluation Dashboard
# Research Contribution 1
# ============================================================

st.header("📊 Framework Evaluation")

st.markdown("""
This section demonstrates the contribution of the proposed
**Information Overload Mitigation Framework** by comparing
operational decision-making before and after intelligent
information filtering and alert prioritisation.
""")

# ------------------------------------------------------------
# Framework KPIs
# ------------------------------------------------------------

total_alerts = len(occ_df)

priority_alerts = occ_df[
    occ_df["OCC_Alert_Level"].isin(["High", "Critical"])
].shape[0]

routine_alerts = occ_df[
    occ_df["OCC_Alert_Level"] == "Routine"
].shape[0]

workload_reduction = (
    (total_alerts - priority_alerts)
    / total_alerts
) * 100
# ------------------------------------------------------------
# KPI Cards
# ------------------------------------------------------------

k1, k2, k3 = st.columns(3)

with k1:
    st.metric(
        "📩 Alerts Without Framework",
        f"{total_alerts:,}"
    )

with k2:
    st.metric(
        "🎯 Priority Alerts After Framework",
        f"{priority_alerts:,}"
    )

with k3:
    st.metric(
        "📉 Monitoring Workload Reduction",
        f"{workload_reduction:.2f}%"
    )

st.divider()

# ------------------------------------------------------------
# Before vs After Comparison
# ------------------------------------------------------------

comparison_df = pd.DataFrame({

    "Framework":[
        "Without Framework",
        "With Framework"
    ],

    "Alerts":[
        total_alerts,
        priority_alerts
    ]

})

fig = px.bar(
    comparison_df,
    x="Framework",
    y="Alerts",
    color="Framework",
    text="Alerts",
    color_discrete_sequence=[
        "#D62728",
        "#2CA02C"
    ],
    title="Alert Volume Before and After Information Overload Mitigation"
)
fig = style_plot(fig)

fig.update_traces(textposition="outside")

fig.update_layout(
    showlegend=False,
    xaxis_title="",
    yaxis_title="Number of Alerts"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------------------------------------------------
# Framework Comparison Table
# ------------------------------------------------------------

st.subheader("📋 Framework Comparison")

comparison_table = pd.DataFrame({

    "Evaluation Metric":[
        "Alerts Presented",
        "Information Filtering",
        "Priority Ranking",
        "Decision Support",
        "Operator Focus"
    ],

    "Without Framework":[
        f"{total_alerts:,}",
        "✖",
        "✖",
        "✖",
        "All Flights"
    ],

    "With Framework":[
        f"{priority_alerts:,}",
        "✔",
        "✔",
        "✔",
        "High & Critical Flights"
    ]

})

st.dataframe(
    comparison_table,
    hide_index=True,
    use_container_width=True
)

st.divider()

# ------------------------------------------------------------
# Research Findings
# ------------------------------------------------------------

st.success(f"""

### 🔬 Research Findings

The proposed **Information Overload Mitigation Framework**
reduced the number of alerts requiring operator attention
from **{total_alerts:,} flights** to **{priority_alerts:,}
priority flights**.

This corresponds to a monitoring workload reduction of
**{workload_reduction:.2f}%**.

The framework therefore enables Operations Control Centre
(OCC) personnel to:

- Focus on operationally significant flights.
- Reduce information overload.
- Improve alert prioritisation.
- Support faster operational decision-making.
- Allocate attention to High and Critical risk flights.

""")

st.divider()

# ============================================================
# AI Model Comparison Dashboard
# Research Contribution 2
# ============================================================

st.header("🤖 AI Model Comparison")

st.markdown("""
This section evaluates the contribution of the proposed
**Hybrid AI Risk Assessment Engine** by comparing the outputs
of the individual machine learning models with the integrated
Hybrid AI framework.
""")

st.divider()

# ------------------------------------------------------------
# Model Statistics
# ------------------------------------------------------------

gmm_anomalies = occ_df[
    occ_df["GMM_Anomaly_Label"] == "Anomaly"
].shape[0]

gmm_normal = occ_df[
    occ_df["GMM_Anomaly_Label"] == "Normal"
].shape[0]

elm_anomalies = occ_df[
    occ_df["ELM_Anomaly_Label"] == "Anomaly"
].shape[0]

elm_normal = occ_df[
    occ_df["ELM_Anomaly_Label"] == "Normal"
].shape[0]

hybrid_high = occ_df[
    occ_df["Hybrid_Risk_Category"] == "High"
].shape[0]

hybrid_critical = occ_df[
    occ_df["Hybrid_Risk_Category"] == "Critical"
].shape[0]

# ------------------------------------------------------------
# KPI Cards
# ------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "🛰 GMM Anomalies",
        f"{gmm_anomalies:,}"
    )

with c2:
    st.metric(
        "⚡ ELM Anomalies",
        f"{elm_anomalies:,}"
    )

with c3:
    st.metric(
        "⚠ Hybrid High Risk",
        f"{hybrid_high:,}"
    )

with c4:
    st.metric(
        "🚨 Hybrid Critical",
        f"{hybrid_critical:,}"
    )

st.divider()

# ------------------------------------------------------------
# Model Comparison Chart
# ------------------------------------------------------------

comparison_df = pd.DataFrame({

    "Model":[
        "GMM",
        "ELM",
        "Hybrid High Risk",
        "Hybrid Critical"
    ],

    "Flights":[
        gmm_anomalies,
        elm_anomalies,
        hybrid_high,
        hybrid_critical
    ]

})

fig = px.bar(

    comparison_df,

    x="Model",

    y="Flights",

    text="Flights",

    color="Model",

    color_discrete_sequence=[
        "#3498db",
        "#9b59b6",
        "#f39c12",
        "#e74c3c"
    ]

)

fig.update_traces(textposition="outside")

fig.update_layout(

    title="Comparison of Individual AI Models and Hybrid AI",

    xaxis_title="",

    yaxis_title="Number of Flights",

    showlegend=False

)

st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("📋 Model Comparison")

comparison_table = pd.DataFrame({

    "Model":[

        "Gaussian Mixture Model (GMM)",

        "ELM Autoencoder",

        "Hybrid AI Risk Assessment"

    ],

    "Primary Purpose":[

        "Behaviour Learning",

        "Anomaly Detection",

        "Integrated Risk Assessment"

    ],

    "Strength":[

        "Learns normal flight behaviour",

        "Detects abnormal flight patterns",

        "Combines ML and operational intelligence"

    ],

    "Limitation":[

        "No operational context",

        "No behavioural understanding",

        "More computationally complex"

    ]

})

st.dataframe(

    comparison_table,

    hide_index=True,

    use_container_width=True

)

st.divider()

st.success(f"""

## 🔬 Research Findings

The proposed **Hybrid AI Risk Assessment Engine** integrates:

• Gaussian Mixture Model (Behaviour Analysis)

• ELM Autoencoder (Anomaly Detection)

• Operational Intelligence

Unlike the individual machine learning models, the Hybrid AI
framework incorporates both behavioural anomalies and
operational context to generate a comprehensive flight
risk assessment.

Key observations:

• GMM detected **{gmm_anomalies:,}** behavioural anomalies.

• ELM detected **{elm_anomalies:,}** anomalous flights.

• The Hybrid AI framework classified
**{hybrid_high + hybrid_critical:,}**
flights as operationally significant.

This demonstrates that integrating multiple intelligence
sources provides more meaningful decision support than using
either model independently.

""")

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

# ============================================================
# Operational Scenario Evaluation
# Research Contribution 3
# ============================================================

st.header("🚦 Operational Scenario Evaluation")

st.markdown("""
This section evaluates the proposed Hybrid AI Decision Support Framework
under different operational scenarios typically encountered within an
Airline Operations Control Centre (OCC).

The evaluation demonstrates how the framework adapts its operational
recommendations according to the current operational context.
""")

st.divider()

# ------------------------------------------------------------
# Scenario Selection
# ------------------------------------------------------------

scenario = st.selectbox(

    "Select Operational Scenario",

    [

        "Normal Operations",

        "Heavy Traffic",

        "High Risk Operations",

        "Critical Operations"

    ]

)

# ------------------------------------------------------------
# Create Scenario Dataset
# ------------------------------------------------------------

scenario_df = filtered_df.copy()

if scenario == "Normal Operations":

    scenario_df = filtered_df[
        filtered_df["Hybrid_Risk_Category"] == "Low"
    ]

elif scenario == "Heavy Traffic":

    threshold = filtered_df[
        "Operational_Complexity_Index"
    ].quantile(0.75)

    scenario_df = filtered_df[
        filtered_df["Operational_Complexity_Index"] >= threshold
    ]

elif scenario == "High Risk Operations":

    scenario_df = filtered_df[
        filtered_df["Hybrid_Risk_Category"] == "High"
    ]

elif scenario == "Critical Operations":

    scenario_df = filtered_df[
        filtered_df["Hybrid_Risk_Category"] == "Critical"
    ]

# ------------------------------------------------------------
# Scenario KPIs
# ------------------------------------------------------------

total = len(scenario_df)

avg_risk = scenario_df[
    "Hybrid_Flight_Risk_Score"
].mean()

critical = scenario_df[
    scenario_df["OCC_Alert_Level"]=="Critical"
].shape[0]

high = scenario_df[
    scenario_df["OCC_Alert_Level"]=="High"
].shape[0]

c1,c2,c3,c4 = st.columns(4)

with c1:

    st.metric(
        "Flights",
        f"{total:,}"
    )

with c2:

    st.metric(
        "Average Risk",
        f"{avg_risk:.2f}"
    )

with c3:

    st.metric(
        "High Alerts",
        f"{high:,}"
    )

with c4:

    st.metric(
        "Critical Alerts",
        f"{critical:,}"
    )

st.divider()

# ------------------------------------------------------------
# Risk Distribution
# ------------------------------------------------------------

risk_summary = (

    scenario_df["Hybrid_Risk_Category"]

    .value_counts()

    .reset_index()

)

risk_summary.columns=[

    "Risk Category",

    "Flights"

]

fig = px.bar(

    risk_summary,

    x="Risk Category",

    y="Flights",

    color="Risk Category",

    text="Flights",

    title=f"Risk Distribution - {scenario}"

)

fig.update_traces(textposition="outside")

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ------------------------------------------------------------
# OCC Recommendations
# ------------------------------------------------------------

recommendations = (

    scenario_df["OCC_Recommendation"]

    .value_counts()

    .reset_index()

)

recommendations.columns=[

    "Recommendation",

    "Flights"

]

fig = px.pie(

    recommendations,

    names="Recommendation",

    values="Flights",

    hole=0.45,

    title=f"OCC Recommendations - {scenario}"

)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.divider()

# ------------------------------------------------------------
# Flights in Current Scenario
# ------------------------------------------------------------

st.subheader("✈ Flights in Selected Scenario")

display_columns=[

    "callsign",

    "origin_country",

    "flight_phase",

    "Hybrid_Flight_Risk_Score",

    "Hybrid_Risk_Category",

    "OCC_Alert_Level",

    "OCC_Recommendation"

]

st.dataframe(

    scenario_df[display_columns]

    .sort_values(

        "Hybrid_Flight_Risk_Score",

        ascending=False

    ),

    hide_index=True,

    use_container_width=True

)

st.divider()

# ------------------------------------------------------------
# Operational Interpretation
# ------------------------------------------------------------

if scenario=="Normal Operations":

    st.success("""

### ✅ Operational Interpretation

• Flights are operating within expected operational limits.

• Routine monitoring is sufficient.

• No significant operational intervention is required.

• OCC operators can focus on maintaining normal operations.

""")

elif scenario=="Heavy Traffic":

    st.warning("""

### 🚦 Operational Interpretation

• Increased operational complexity detected.

• Enhanced monitoring is recommended.

• Operators should monitor congestion-related risks.

• Resource allocation should be reviewed.

""")

elif scenario=="High Risk Operations":

    st.warning("""

### ⚠ Operational Interpretation

• Multiple high-risk flights have been identified.

• Continuous monitoring is recommended.

• OCC supervisors should prepare mitigation strategies.

• Airlines should monitor operational disruptions.

""")

elif scenario=="Critical Operations":

    st.error("""

### 🚨 Operational Interpretation

• Critical operational conditions detected.

• Immediate OCC intervention is required.

• Notify supervisors immediately.

• Coordinate with ATC and airline dispatch.

• Prepare contingency procedures.

""")

    # ============================================================
# Intelligent OCC Decision Support
# Research Contribution 4
# ============================================================

st.header("🧠 Intelligent OCC Decision Support")

st.markdown("""
This section demonstrates how the proposed Decision Support Framework
adapts its recommendations according to the operational situation,
Hybrid AI Risk Assessment and alert severity.

Unlike a conventional alerting system, the proposed framework provides
context-aware operational actions for OCC personnel.
""")

st.divider()

# ------------------------------------------------------------
# Select Flight
# ------------------------------------------------------------

scenario_df["callsign"] = (
    scenario_df["callsign"]
    .fillna("")
    .astype(str)
    .str.strip()
)

flight_options = (
    scenario_df.loc[
        scenario_df["callsign"] != "",
        "callsign"
    ]
    .unique()
)

selected_flight = st.selectbox(
    "Select Flight for Decision Support",
    flight_options,
    key="decision_support_flight"
)

flight = scenario_df[
    scenario_df["callsign"] == selected_flight
].iloc[0]

risk = flight["Hybrid_Risk_Category"]
alert = flight["OCC_Alert_Level"]
phase = flight["flight_phase"]
complexity = flight["Operational_Complexity_Index"]

# ------------------------------------------------------------
# Flight Summary
# ------------------------------------------------------------

st.subheader("✈ Flight Summary")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Callsign", flight["callsign"])

with c2:
    st.metric("Risk Category", risk)

with c3:
    st.metric("Alert Level", alert)

with c4:
    st.metric(
        "Risk Score",
        f"{flight['Hybrid_Flight_Risk_Score']:.2f}"
    )

st.divider()

# ------------------------------------------------------------
# Why was this flight prioritised?
# ------------------------------------------------------------

st.subheader("🔍 Why was this flight prioritised?")

reasons = []

if flight["GMM_Anomaly_Label"] == "Anomaly":
    reasons.append("✅ Behavioural anomaly detected by Gaussian Mixture Model")

if flight["ELM_Anomaly_Label"] == "Anomaly":
    reasons.append("✅ Structural anomaly detected by ELM Autoencoder")

if complexity >= scenario_df["Operational_Complexity_Index"].median():
    reasons.append("✅ High operational complexity")

if phase in ["Approach", "Landing", "Departure"]:
    reasons.append("✅ Critical phase of flight")

if len(reasons) == 0:
    reasons.append("✅ Routine operational behaviour")

for item in reasons:
    st.write(item)

st.divider()

# ------------------------------------------------------------
# Adaptive OCC Recommendations
# ------------------------------------------------------------

st.subheader("🚦 Recommended OCC Actions")

if risk == "Critical":

    st.error("""
### Immediate Operational Intervention

1. Notify OCC Supervisor immediately.

2. Contact Flight Crew.

3. Coordinate with Air Traffic Control.

4. Prepare alternate airport if required.

5. Increase monitoring frequency.

6. Activate contingency procedures.

Priority Level: **Immediate**
""")

elif risk == "High":

    st.warning("""
### Enhanced Operational Monitoring

1. Notify OCC Supervisor.

2. Monitor aircraft continuously.

3. Review airport congestion.

4. Assess operational delays.

5. Prepare mitigation strategies.

Priority Level: **High**
""")

elif risk == "Medium":

    st.info("""
### Preventive Monitoring

1. Continue enhanced monitoring.

2. Observe operational conditions.

3. Review airport status.

4. Monitor flight progress.

Priority Level: **Medium**
""")

else:

    st.success("""
### Routine Operations

1. Continue routine monitoring.

2. No immediate intervention required.

3. Maintain standard OCC procedures.

Priority Level: **Routine**
""")

st.divider()

# ------------------------------------------------------------
# Decision Timeline
# ------------------------------------------------------------

st.subheader("📈 Decision Support Workflow")

workflow = pd.DataFrame({

    "Stage":[

        "ADS-B Data",

        "Feature Engineering",

        "GMM Behaviour Analysis",

        "ELM Autoencoder",

        "Operational Intelligence",

        "Hybrid AI Risk Assessment",

        "Decision Support",

        "OCC Recommendation"

    ],

    "Status":[

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Completed",

        "Generated"

    ]

})

st.dataframe(
    workflow,
    hide_index=True,
    use_container_width=True
)

st.divider()

# ------------------------------------------------------------
# Research Contribution
# ------------------------------------------------------------

st.success(f"""
## 🔬 Research Contribution

For the selected flight (**{flight['callsign']}**), the proposed
Decision Support Framework combined:

- Behavioural intelligence from the GMM.
- Structural anomaly detection from the ELM Autoencoder.
- Operational intelligence from engineered aviation features.
- Rule-based decision support logic.

The framework generated a **{risk}** operational risk assessment
and recommended an OCC response appropriate to the operational situation.

This demonstrates how Hybrid AI can move beyond anomaly detection
to provide actionable operational decision support for airline
Operations Control Centres.
""")

