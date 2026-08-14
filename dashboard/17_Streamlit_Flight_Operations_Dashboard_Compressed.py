import streamlit as st
import pandas as pd
import plotly.express as px



st.set_page_config(
    page_title="Intelligent Flight Operations Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)



st.markdown("""
<style>
.block-container{
    padding-top:1.9rem;
    padding-left:0.8rem;
    padding-right:0.8rem;
    padding-bottom:0rem;
    max-width:100%;
}
div[data-testid="stMetric"]{
    background:#1b1f2b;
    border:1px solid #2d3748;
    border-radius:8px;
    padding:2px 6px !important;
    min-height:50px;
}
div[data-testid="stMetricLabel"]{ font-size:10px; }
div[data-testid="stMetricValue"]{ font-size:15px; }
[data-testid="stSidebar"]{ background:#0B2545; }
[data-testid="stSidebar"] *{ color:white; }
section[data-testid="stSidebar"]{ width:220px !important; }

h1{
    color:#003366;
    font-weight:700;
    font-size:22px !important;
    margin:0 !important;
}
h3{
    margin-top:0px !important;
    margin-bottom:0px !important;
    font-size:14px !important;
}
p{ margin-bottom:0px; }


div[data-testid="stVerticalBlock"] > div{
    padding-top:0rem;
    padding-bottom:0rem;
    gap:0.2rem;
}
div[data-testid="stHorizontalBlock"]{
    gap:0.6rem;
}
div[data-testid="stPlotlyChart"]{
    margin-top:-10px;
    margin-bottom:-10px;
}
div[data-testid="stDataFrame"]{
    margin-top:-10px;
}
hr{ margin:4px 0 !important; }
.stButton button{
    padding:1px 8px;
    font-size:12px;
    min-height:26px;
}
[data-testid="stMarkdownContainer"] p{
    font-size:12px;
}
</style>
""", unsafe_allow_html=True)


from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
DATA_DIR = PROJECT_DIR / "data" / "final"
ASSETS_DIR = CURRENT_DIR / "assets"


@st.cache_data
def load_data():
    occ_df = pd.read_csv(DATA_DIR / "occ_decision_support_dataset.csv")
    hybrid_df = pd.read_csv(DATA_DIR / "hybrid_ai_risk_assessment.csv")
    summary_df = pd.read_csv(DATA_DIR / "dashboard_summary.csv")
    return occ_df, hybrid_df, summary_df


occ_df, hybrid_df, summary_df = load_data()



for key, default in [
    ("show_map", True),
    ("show_alerts", True),
    ("show_risk", True),
    ("show_distribution", True),
    ("show_priority", True),
    ("show_status", True),
    ("show_status_graph", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default



st.sidebar.image(
    "https://img.icons8.com/fluency/96/airplane-take-off.png",
    width=80
)
st.sidebar.title("Flight Operations")

country = st.sidebar.multiselect(
    "Origin Country",
    sorted(occ_df["origin_country"].dropna().unique()),
    default=sorted(occ_df["origin_country"].dropna().unique())
)

phase = st.sidebar.multiselect(
    "Flight Phase",
    sorted(occ_df["flight_phase"].dropna().unique()),
    default=sorted(occ_df["flight_phase"].dropna().unique())
)

filtered_df = occ_df[
    (occ_df["origin_country"].isin(country)) &
    (occ_df["flight_phase"].isin(phase))
]


total_flights = len(filtered_df)
priority = filtered_df[
    filtered_df["OCC_Alert_Level"].isin(["High", "Critical"])
].shape[0]
critical = filtered_df[
    filtered_df["Hybrid_Risk_Category"] == "Critical"
].shape[0]
average_risk = filtered_df["Hybrid_Flight_Risk_Score"].mean()
workload = (priority / total_flights) * 100 if total_flights else 0

scenario_total = len(occ_df)
scenario_priority = occ_df[
    occ_df["OCC_Alert_Level"].isin(["High", "Critical"])
].shape[0]
scenario_critical = occ_df[
    occ_df["Hybrid_Risk_Category"] == "Critical"
].shape[0]
scenario_focus = (
    scenario_priority / scenario_total * 100 if scenario_total > 0 else 0
)




@st.dialog("Information Overload Mitigation", width="large")
def overload_dialog():
    st.caption(
        "Progressive prioritisation of operational information "
        "for focused OCC operator attention."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Information", f"{scenario_total:,}")
    with m2:
        st.metric("Priority Flights", f"{scenario_priority:,}")
    with m3:
        st.metric("Critical Flights", f"{scenario_critical:,}")
    with m4:
        st.metric("Operator Focus", f"{scenario_focus:.1f}%")

    st.markdown("#### How the Information is Prioritised")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("**1. Information Load**")
        st.markdown(f"**{scenario_total:,}** observations")
        st.caption("Complete operational information")
    with p2:
        st.markdown("**2. Hybrid AI Assessment**")
        st.markdown("**GMM + ELM + Context**")
        st.caption("Behavioural and operational risk assessment")
    with p3:
        st.markdown("**3. Prioritisation**")
        st.markdown(f"**{scenario_priority:,}** priority flights")
        st.caption("Greater operator attention")
    with p4:
        st.markdown("**4. Critical Escalation**")
        st.markdown(f"**{scenario_critical:,}** critical flights")
        st.caption("Highest-risk operational cases")

    st.markdown("#### Progressive Information Prioritisation")
    funnel_df = pd.DataFrame({
        "Stage": [
            "Full Operational Information",
            "Priority Information",
            "Critical Information"
        ],
        "Flights": [scenario_total, scenario_priority, scenario_critical]
    })
    funnel_fig = px.funnel(funnel_df, y="Stage", x="Flights")
    funnel_fig.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=5, b=5),
        showlegend=False,
        xaxis_title="Flight Observations",
        yaxis_title=""
    )
    st.plotly_chart(
        funnel_fig, use_container_width=True,
        config={"displayModeBar": False}
    )

    st.info(
        f"""
**Scenario Interpretation:** The OCC initially has
**{scenario_total:,} observations** available. The hybrid framework
uses **GMM, ELM Autoencoder and operational context** to assess and
prioritise the information. **{scenario_priority:,} flights** are
identified for greater operator attention, with **{scenario_critical:,}**
classified as Critical. The priority subset represents approximately
**{scenario_focus:.1f}%** of the total operational information.

This demonstrates **information prioritisation**, not a direct
measurement of human cognitive workload.
"""
    )


h_left, h_right = st.columns([5, 1])

with h_left:
    st.markdown("""
<h1>✈ Intelligent Flight Operations Analytics Dashboard</h1>
<p style="margin-top:2px;margin-bottom:4px;font-size:12px;color:#A0A0A0;">
Hybrid AI Risk Assessment • Information Overload Mitigation • OCC Decision Support
</p>
""", unsafe_allow_html=True)

with h_right:
    st.write("")
    if st.button("▶ Simulate Overload", use_container_width=True):
        overload_dialog()



k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Flights", f"{total_flights:,}")
k2.metric("Priority", priority)
k3.metric("Critical", critical)
k4.metric("Avg Risk", f"{average_risk:.1f}")
k5.metric("Operator Focus", f"{workload:.1f}%")
k6.metric("Status", "Operational")

st.markdown("<hr>", unsafe_allow_html=True)


left, right = st.columns([2, 1])

with left:
    c1, c2 = st.columns([9, 1])
    with c1:
        st.subheader("Live Flight Operations")
    with c2:
        if st.button("👁" if st.session_state.show_map else "🙈", key="map_btn"):
            st.session_state.show_map = not st.session_state.show_map

    if st.session_state.show_map:
        fig = px.scatter_map(
            filtered_df,
            lat="latitude",
            lon="longitude",
            hover_name="callsign",
            hover_data=[
                "origin_country", "flight_phase",
                "Hybrid_Risk_Category", "Hybrid_Flight_Risk_Score"
            ],
            color="Hybrid_Risk_Category",
            zoom=1,
            height=160,
            color_discrete_map={
                "Low": "green", "Medium": "gold",
                "High": "orange", "Critical": "red"
            }
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            map_style="carto-darkmatter"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    c1, c2 = st.columns([9, 1])
    with c1:
        st.subheader("Live OCC Alerts")
    with c2:
        if st.button("👁" if st.session_state.show_alerts else "🙈", key="alerts_btn"):
            st.session_state.show_alerts = not st.session_state.show_alerts

    if st.session_state.show_alerts:
        crit_df = filtered_df[filtered_df["Hybrid_Risk_Category"] == "Critical"]
        high_df = filtered_df[filtered_df["Hybrid_Risk_Category"] == "High"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Critical", len(crit_df))
        c2.metric("High", len(high_df))
        c3.metric("Priority", priority)

        st.markdown("###### Top Alerts")
        latest = filtered_df.sort_values("Flight_Priority_Rank").head(3)
        st.dataframe(
            latest[["callsign", "Hybrid_Risk_Category", "OCC_Alert_Level"]],
            hide_index=True,
            use_container_width=True,
            height=60
        )



left, right = st.columns(2)

with left:
    c1, c2 = st.columns([9, 1])
    with c1:
        st.subheader("Hybrid Risk Distribution")
    with c2:
        if st.button("👁" if st.session_state.show_risk else "🙈", key="risk_btn"):
            st.session_state.show_risk = not st.session_state.show_risk

    if st.session_state.show_risk:
        risk_counts = filtered_df["Hybrid_Risk_Category"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Flights"]
        fig = px.pie(
            risk_counts, names="Risk", values="Flights", hole=0.65, color="Risk",
            color_discrete_map={
                "Low": "green", "Medium": "gold",
                "High": "orange", "Critical": "red"
            }
        )
        fig.update_layout(height=130, margin=dict(l=0, r=0, t=10, b=0), showlegend=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    c1, c2 = st.columns([9, 1])
    with c1:
        st.subheader("OCC Alert Distribution")
    with c2:
        if st.button("👁" if st.session_state.show_distribution else "🙈", key="distribution_btn"):
            st.session_state.show_distribution = not st.session_state.show_distribution

    if st.session_state.show_distribution:
        alert_counts = filtered_df["OCC_Alert_Level"].value_counts().reset_index()
        alert_counts.columns = ["Alert", "Flights"]
        fig = px.bar(
            alert_counts, x="Alert", y="Flights", color="Alert", text="Flights",
            color_discrete_map={
                "Routine": "green", "Medium": "gold",
                "High": "orange", "Critical": "red"
            }
        )
        fig.update_layout(height=130, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})



left, right = st.columns([1.9, 1])

with left:
    c1, c2 = st.columns([9, 1])
    with c1:
        st.subheader("Top Priority Flights")
    with c2:
        if st.button("👁" if st.session_state.show_priority else "🙈", key="priority_btn"):
            st.session_state.show_priority = not st.session_state.show_priority

    if st.session_state.show_priority:
        priority_df = filtered_df.sort_values("Flight_Priority_Rank").head(4)
        st.dataframe(
            priority_df[[
                "callsign", "origin_country", "flight_phase",
                "Hybrid_Risk_Category", "Hybrid_Flight_Risk_Score", "OCC_Alert_Level"
            ]],
            hide_index=True,
            use_container_width=True,
            height=130
        )

with right:
    h1, h2, h3 = st.columns([8, 1, 1])
    with h1:
        st.subheader("Flight Risk Analytics" if st.session_state.show_status_graph else "Aircraft Status")
    with h2:
        if st.button("📊", key="status_graph_btn"):
            st.session_state.show_status_graph = not st.session_state.show_status_graph
    with h3:
        if st.button("👁" if st.session_state.show_status else "🙈", key="status_btn"):
            st.session_state.show_status = not st.session_state.show_status

    if st.session_state.show_status:
        callsigns = sorted(filtered_df["callsign"].dropna().astype(str).unique())
        selected = st.selectbox("", callsigns, label_visibility="collapsed")
        flight = filtered_df[filtered_df["callsign"] == selected].iloc[0]

        if not st.session_state.show_status_graph:
            risk = flight["Hybrid_Risk_Category"]
            st.markdown(f"""
###### {risk} Risk — {flight["callsign"]}
| | | | |
|---|---|---|---|
| Country | {flight["origin_country"]} | Phase | {flight["flight_phase"]} |
| Risk | {flight["Hybrid_Flight_Risk_Score"]:.1f} | Alert | {flight["OCC_Alert_Level"]} |
| Complexity | {flight["Operational_Complexity_Index"]:.1f} | Flight Score | {flight["Flight_Operational_Score"]:.1f} |
| Behaviour | {flight["Behavioural_Intelligence_Score"]:.1f} | Intelligence | {flight["Operational_Intelligence_Score"]:.1f} |

**Recommendation:** {flight["OCC_Recommendation"]}
""")
        else:
            chart_df = pd.DataFrame({
                "Metric": ["Risk", "Complexity", "Flight", "Behaviour", "Intelligence"],
                "Score": [
                    flight["Hybrid_Flight_Risk_Score"],
                    flight["Operational_Complexity_Index"],
                    flight["Flight_Operational_Score"],
                    flight["Behavioural_Intelligence_Score"],
                    flight["Operational_Intelligence_Score"]
                ]
            })
            fig = px.bar(chart_df, x="Metric", y="Score", color="Metric", text="Score")
            fig.update_layout(
                height=150,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
                yaxis_title="",
                xaxis_title=""
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption(flight["OCC_Recommendation"])