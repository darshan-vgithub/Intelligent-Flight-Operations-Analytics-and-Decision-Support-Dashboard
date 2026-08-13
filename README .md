
# Intelligent Flight Operations Analytics and Decision Support Dashboard with Hybrid AI-Based Risk Assessment and Information Overload Mitigation

An MSc dissertation project developing an intelligent aviationOperations Control Centre (OCC) decision-support prototype. The systemanalyses flight-operation information, detects abnormal behaviour,assesses operational risk, prioritises alerts, and presents importantinformation through an interactive Streamlit dashboard.

Core idea: use AI to transform a large operational informationenvironment into a risk-prioritised decision-support view so an OCCoperator can focus attention on the most operationally significantcases.


## 1. Objectives
Analyse flight-operation data.

Detect abnormal aircraft behaviour.

Combine behavioural anomaly information with operational context.

Produce a hybrid flight-risk assessment.

Prioritise operational alerts.

Demonstrate information-overload mitigation through progressiveprioritisation.

Present risk and recommendations through an OCC dashboard.

Provide a foundation for future human-centred evaluation.
## 2. System Pipeline
Raw Aviation Data
        ↓
Cleaning & Quality Assessment
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Cross-Dataset Integration
        ↓
Information Overload Mitigation
        ↓
GMM Anomaly Detection
        +
ELM Autoencoder
        +
Operational Context
        ↓
Hybrid Flight Risk Assessment
        ↓
Risk Category
Low / Medium / High / Critical
        ↓
Alert Prioritisation
        ↓
OCC Decision Support Dashboard


## 3. Main Contribution
The project proposes a hybrid aviation decision-support frameworkcombining:

GMM-based behavioural anomaly information

ELM Autoencoder anomaly information

Operational context

Hybrid flight-risk scoring

Risk categorisation

Smart alert prioritisation

Interactive OCC visualisation

The system is designed as decision support, not as anair-traffic-control or certified flight-tracking system.
## 4. Information Overload Mitigation
The dashboard contains an interactive overload scenario demonstratingprogressive information prioritisation.

Current dataset results:

Measure                            Current result
Total operational observations          7,516Priority flights                          500Critical flights                           66Operator-focused subset                  6.7%

Calculation:

500 / 7,516 × 100 ≈ 6.7%

The dashboard visualises:

7,516
Full Operational Information
        ↓
Hybrid AI Risk Assessment
        ↓
500
Priority Information
        ↓
66
Critical Information
        ↓
6.7%
Focused Operator Attention

Important interpretation

The 6.7% value is an information-prioritisation metric. It does notmean that human cognitive workload has been reduced by 93.3%. Directcognitive-workload reduction would require controlled user evaluation.
## 5. Machine Learning
Gaussian Mixture Model

The GMM models normal flight behaviour and identifies observations thatdeviate from learned behavioural patterns.

Current GMM features include:

Latitude

Longitude

Velocity

Barometric altitude

Geometric altitude

Vertical rate

True track

Distance to airport

Operational Complexity Index

On-ground status

The current model-selection process selected 18 GMM components.

Generated GMM outputs include:

GMM_Cluster

GMM_Log_Likelihood

GMM_Anomaly_Score

GMM_Anomaly_Label

ELM Autoencoder

The ELM Autoencoder provides a second behavioural anomaly signal usingreconstruction-based anomaly detection.

Hybrid Risk Assessment

The hybrid engine combines:

GMM anomaly information
        +
ELM anomaly information
        +
Operational context
        ↓
Hybrid Flight Risk Score
        ↓
Risk Category
        ↓
Alert / Priority
## 6. Data Sources
The project uses multiple aviation-related datasets.

OpenSky ADS-B

Primary source for flight-operation observations, behavioural analysis,feature engineering and anomaly detection.

Airport Data

Provides airport and operational context, including airport-relatedfeatures and distance-to-airport information.

Airline Delay Data

Provides operational delay context.

Weather Data

Provides additional operational-context information for the widerrisk-assessment framework.

Aircraft Metadata

Provides aircraft context such as:

Aircraft age

Engine configuration

Seat-capacity category
## 7. OpenSky Data Processing
The initial OpenSky dataset contained:

7,561 rows

16 columns

After cleaning:

7,516 rows

Feature engineering produced operational variables including:

flight_phase

speed_category

altitude_band

vertical_movement

heading_direction

operational_risk

Additional cross-dataset features were developed for the OCCdecision-support framework.
## 8. Dashboard
The project uses Streamlit for the interactive dashboard.

Main dashboard features

Executive KPIs

Origin Country filtering

Flight Phase filtering

Live flight-operation map

Live OCC alerts

Critical / High / Priority alert counts

Top alerts

Hybrid risk distribution

OCC alert distribution

Top priority flights

Aircraft status

Flight risk analytics

Operational recommendations

Information Overload Mitigation scenario

Progressive information-prioritisation funnel

Overload demonstration

The dashboard provides:

▶ Simulate Overload

The scenario demonstrates:

Initial operational information load.

Hybrid AI risk assessment.

Risk-based information prioritisation.

Critical alert escalation.

Progressive prioritisation from the complete dataset to a focusedsubset.
## 9. Technology Stack
Programming and Data

Python

Pandas

NumPy

Machine Learning

Scikit-learn

Custom ELM Autoencoder implementation

Visualisation

Plotly

Matplotlib

Seaborn

Dashboard

Streamlit

Development

Jupyter Notebook

Visual Studio Code

Explainability

SHAP is part of the wider project explainability work.
## 11. Installation
Create or activate a Python environment, then install the maindependencies:

pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn shap

If additional packages are required by a specific notebook, installthose separately.
## 12. Running the Dashboard
The dashboard expects the final datasets under:

data/final/

The main files are:

occ_decision_support_dataset.csv
hybrid_ai_risk_assessment.csv
dashboard_summary.csv

Run the dashboard from the project root:

streamlit run dashboard/17_Streamlit_Flight_Operations_Dashboard.py

Use the current dashboard filename if it has been renamed.
## 13. Dashboard Interaction
Filters

The sidebar provides:

Origin Country

Flight Phase

Executive KPIs

The dashboard summarises:

Flights

Priority

Critical

Average Risk

Operator Focus

Status

Operational Map

Displays flight observations geographically with risk informationavailable through colour and hover details.

OCC Alerts

Displays:

Critical alerts

High alerts

Priority information

Top alerts

Risk and Alert Distributions

Provides visual summaries of hybrid risk categories and OCC alertlevels.

Top Priority Flights

Surfaces high-priority flights so the operator can focus on the mostimportant cases.

Aircraft Status

Allows selection of an aircraft and inspection of risk, alert,operational and recommendation information
## 14. Evaluation
The current Information Overload Mitigation evaluation is primarily atthe system and information-prioritisation level.

Current evidence:

Initial information:       7,516
Priority information:        500
Critical information:         66
Focused subset:              6.7%

This demonstrates that the prototype can progressively prioritise alarge operational information set.

It does not directly establish human cognitive-workload reduction.

Future human evaluation

A controlled evaluation could compare operators using:

The prioritised decision-support interface

An unprioritised information interface

Potential measures include:

Task completion time

Detection accuracy

Response time

Decision accuracy

Missed critical events

Perceived workload

User satisfaction

Trust in recommendations
## 15. Limitations
The current overload evaluation measures information prioritisationrather than direct cognitive workload.

The 6.7% value is a system-level information metric.

Human factors have not yet been fully validated through a controlleduser study.

The project combines datasets with different structures and levelsof completeness.

Weather and other contextual features depend on data availabilityand alignment.

The dashboard is an academic decision-support prototype, not acertified aviation operational system.

Further model validation is required on unseen and morerepresentative operational data.
## 16. Future Work
Potential extensions include:

Controlled OCC user evaluation

Human cognitive-workload measurement

More extensive SHAP explanations

Temporal flight-trajectory modelling

Real-time ADS-B streaming

Adaptive alert thresholds

Alert grouping and suppression

More detailed airport congestion modelling

Additional weather-risk features

Aircraft-specific risk modelling

Model calibration and validation

Human-in-the-loop decision-support evaluation
## 17. Research Areas
This project combines:

Aviation Analytics

Machine Learning

Anomaly Detection

Risk Assessment

Explainable AI

Decision Support Systems

Information Overload Mitigation

Interactive Visual Analytics

Human-Centred Decision Support
## 18. Academic Disclaimer
This project is an academic research prototype developed for MScdissertation work.

It is intended for research, experimentation and demonstration. It isnot a certified aviation operational system and should not replacequalified aviation procedures, certified systems or professionaljudgement.