
# ============================================================
# PALO ALTO NETWORKS
# EMPLOYEE ATTRITION RISK DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Risk Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    # CSV must be in the same GitHub repository as app.py
    file_path = Path(__file__).parent / "Palo Alto Networks.csv"

    if not file_path.exists():
        st.error(
            "Palo Alto Networks.csv was not found. "
            "Please upload the CSV file to the same GitHub "
            "repository as app.py."
        )
        st.stop()

    data = pd.read_csv(file_path)

    return data


df = load_data()


# ============================================================
# BASIC DATA VALIDATION
# ============================================================

required_columns = [
    "Attrition",
    "Department",
    "JobRole",
    "MonthlyIncome",
    "TotalWorkingYears",
    "YearsSinceLastPromotion",
    "JobInvolvement",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "OverTime"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The following required columns are missing from "
        "Palo Alto Networks.csv:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["IncomeToExperienceRatio"] = (
    df["MonthlyIncome"] /
    (df["TotalWorkingYears"] + 1)
)

df["PromotionDelay"] = np.where(
    df["YearsSinceLastPromotion"] >= 5,
    1,
    0
)

df["EngagementScore"] = (
    df["JobInvolvement"]
    + df["JobSatisfaction"]
    + df["EnvironmentSatisfaction"]
    + df["RelationshipSatisfaction"]
    + df["WorkLifeBalance"]
) / 5

df["WorkloadStress"] = np.where(
    (df["OverTime"] == "Yes") &
    (df["WorkLifeBalance"] <= 2),
    1,
    0
)


# ============================================================
# EMPLOYEE ID
# ============================================================

df["EmployeeID"] = range(1, len(df) + 1)


# ============================================================
# PREPARE TARGET
# ============================================================

# Convert Attrition into binary 0/1 if necessary

if df["Attrition"].dtype == "object":

    attrition_values = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Attrition"] = attrition_values.map({
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0
    })

else:

    df["Attrition"] = pd.to_numeric(
        df["Attrition"],
        errors="coerce"
    )


# Remove rows where target could not be converted
df = df.dropna(
    subset=["Attrition"]
).copy()

df["Attrition"] = df["Attrition"].astype(int)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[
        "Attrition",
        "EmployeeID"
    ]
)

y = df["Attrition"]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = (
    X.select_dtypes(
        include=["object", "category"]
    )
    .columns
    .tolist()
)

numerical_features = (
    X.select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# CREATE MODELS
# ============================================================

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


gradient_boosting_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )
        )
    ]
)


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def train_models(
    X_train,
    y_train,
    X_test,
    y_test
):

    models = {

        "Logistic Regression":
            Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),

                    (
                        "classifier",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1000,
                            random_state=42
                        )
                    )
                ]
            ),

        "Random Forest":
            Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),

                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=300,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1
                        )
                    )
                ]
            ),

        "Gradient Boosting":
            Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),

                    (
                        "classifier",
                        GradientBoostingClassifier(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=42
                        )
                    )
                ]
            )
    }


    results = []


    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]


        results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_test,
                predictions
            ),

            "Precision": precision_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "F1-Score": f1_score(
                y_test,
                predictions,
                zero_division=0
            ),

            "ROC-AUC": roc_auc_score(
                y_test,
                probabilities
            )
        })


    results_df = pd.DataFrame(
        results
    )


    # Select model with highest ROC-AUC
    best_model_name = results_df.loc[
        results_df["ROC-AUC"].idxmax(),
        "Model"
    ]


    best_model = models[
        best_model_name
    ]


    return (
        models,
        results_df,
        best_model_name,
        best_model
    )


(
    models,
    model_results,
    best_model_name,
    best_model
) = train_models(
    X_train,
    y_train,
    X_test,
    y_test
)


# ============================================================
# GENERATE ATTRITION PROBABILITY
# ============================================================

X_all = df.drop(
    columns=[
        "Attrition",
        "EmployeeID"
    ]
)

df["Attrition_Probability"] = (
    best_model
    .predict_proba(X_all)[:, 1]
    * 100
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Palo Alto Networks Employee Attrition Risk Dashboard"
)

st.write(
    "Machine Learning-Based Employee Attrition "
    "Prediction and Risk Scoring System"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🎛️ Dashboard Filters"
)


# ------------------------------------------------------------
# DEPARTMENT
# ------------------------------------------------------------

department_options = [
    "All"
] + sorted(
    df["Department"]
    .dropna()
    .unique()
    .tolist()
)

selected_department = st.sidebar.selectbox(
    "🏢 Department",
    department_options
)


# ------------------------------------------------------------
# JOB ROLE
# ------------------------------------------------------------

role_options = [
    "All"
] + sorted(
    df["JobRole"]
    .dropna()
    .unique()
    .tolist()
)

selected_role = st.sidebar.selectbox(
    "💼 Job Role",
    role_options
)


# ------------------------------------------------------------
# RISK THRESHOLD
# ------------------------------------------------------------

risk_threshold = st.sidebar.slider(
    "⚠️ High-Risk Threshold (%)",
    min_value=30,
    max_value=90,
    value=60,
    step=5
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df.copy()


if selected_department != "All":

    filtered_df = filtered_df[
        filtered_df["Department"]
        == selected_department
    ]


if selected_role != "All":

    filtered_df = filtered_df[
        filtered_df["JobRole"]
        == selected_role
    ]


# ============================================================
# EMPLOYEE ID SELECTOR
# ============================================================

employee_ids = (
    filtered_df["EmployeeID"]
    .sort_values()
    .tolist()
)


if len(employee_ids) > 0:

    selected_employee = st.sidebar.selectbox(
        "👤 Employee ID",
        employee_ids
    )

else:

    st.sidebar.warning(
        "No employees match the selected filters."
    )

    selected_employee = None


# ------------------------------------------------------------
# SIDEBAR SUMMARY
# ------------------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader(
    "📌 Current Selection"
)

st.sidebar.write(
    f"**Department:** {selected_department}"
)

st.sidebar.write(
    f"**Job Role:** {selected_role}"
)

st.sidebar.write(
    f"**Risk Threshold:** {risk_threshold}%"
)

if selected_employee is not None:

    st.sidebar.write(
        f"**Employee ID:** {selected_employee}"
    )


# ============================================================
# RISK CATEGORY FUNCTION
# ============================================================

def get_risk_category(
    probability,
    threshold
):

    if probability < 30:

        return "Low Risk"

    elif probability < threshold:

        return "Medium Risk"

    else:

        return "High Risk"


# ============================================================
# OVERALL ATTRITION RISK DASHBOARD
# ============================================================

st.header(
    "1. Attrition Risk Dashboard"
)


col1, col2, col3, col4 = st.columns(4)


total_employees = len(
    filtered_df
)


high_risk = (
    filtered_df[
        "Attrition_Probability"
    ] >= risk_threshold
).sum()


medium_risk = (
    (
        filtered_df[
            "Attrition_Probability"
        ] >= 30
    )
    &
    (
        filtered_df[
            "Attrition_Probability"
        ] < risk_threshold
    )
).sum()


low_risk = (
    filtered_df[
        "Attrition_Probability"
    ] < 30
).sum()


col1.metric(
    "Total Employees",
    int(total_employees)
)

col2.metric(
    "High Risk",
    int(high_risk)
)

col3.metric(
    "Medium Risk",
    int(medium_risk)
)

col4.metric(
    "Low Risk",
    int(low_risk)
)


# ============================================================
# OVERALL RISK DISTRIBUTION
# ============================================================

st.subheader(
    "Overall Risk Distribution"
)


risk_distribution = pd.DataFrame({

    "Risk Category": [
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ],

    "Employees": [
        low_risk,
        medium_risk,
        high_risk
    ]
})


colors = [
    "green",
    "orange",
    "red"
]


fig, ax = plt.subplots(
    figsize=(8, 5)
)


bars = ax.bar(
    risk_distribution[
        "Risk Category"
    ],
    risk_distribution[
        "Employees"
    ],
    color=colors
)


ax.set_title(
    "Overall Risk Distribution",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Risk Category"
)

ax.set_ylabel(
    "Number of Employees"
)


ax.bar_label(
    bars,
    labels=[
        str(int(value))
        for value in
        risk_distribution[
            "Employees"
        ]
    ],
    padding=5,
    fontsize=12,
    fontweight="bold"
)


ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)


max_value = max(
    risk_distribution[
        "Employees"
    ]
)

ax.set_ylim(
    0,
    max_value * 1.15
)


plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# HIGH-RISK EMPLOYEES
# ============================================================

st.subheader(
    "High-Risk Employees"
)


high_risk_table = (
    filtered_df[
        filtered_df[
            "Attrition_Probability"
        ] >= risk_threshold
    ]
    [
        [
            "EmployeeID",
            "Age",
            "Department",
            "JobRole",
            "MonthlyIncome",
            "OverTime",
            "JobSatisfaction",
            "WorkLifeBalance",
            "Attrition_Probability"
        ]
    ]
    .copy()
)


high_risk_table[
    "Risk_Category"
] = high_risk_table[
    "Attrition_Probability"
].apply(
    lambda x:
    get_risk_category(
        x,
        risk_threshold
    )
)


high_risk_table[
    "Attrition_Probability"
] = (
    high_risk_table[
        "Attrition_Probability"
    ].round(2)
)


high_risk_table = (
    high_risk_table
    .sort_values(
        "Attrition_Probability",
        ascending=False
    )
)


st.dataframe(
    high_risk_table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# INDIVIDUAL EMPLOYEE RISK PROFILE
# ============================================================

st.header(
    "2. Individual Employee Risk Profile"
)


if selected_employee is not None:

    employee = df[
        df["EmployeeID"]
        == selected_employee
    ].iloc[0]


    employee_risk_category = (
        get_risk_category(
            employee[
                "Attrition_Probability"
            ],
            risk_threshold
        )
    )


    profile_col1, profile_col2, profile_col3 = (
        st.columns(3)
    )


    profile_col1.metric(
        "Employee ID",
        int(
            employee["EmployeeID"]
        )
    )


    profile_col2.metric(
        "Attrition Probability",
        f"{employee['Attrition_Probability']:.2f}%"
    )


    profile_col3.metric(
        "Risk Category",
        employee_risk_category
    )


    st.subheader(
        "Employee Details"
    )


    details = pd.DataFrame({

        "Attribute": [

            "Age",
            "Department",
            "Job Role",
            "Monthly Income",
            "Job Satisfaction",
            "Environment Satisfaction",
            "Relationship Satisfaction",
            "Work-Life Balance",
            "Overtime",
            "Years at Company",
            "Years in Current Role",
            "Years Since Last Promotion"
        ],


        "Value": [

            employee["Age"],
            employee["Department"],
            employee["JobRole"],
            employee["MonthlyIncome"],
            employee["JobSatisfaction"],
            employee["EnvironmentSatisfaction"],
            employee["RelationshipSatisfaction"],
            employee["WorkLifeBalance"],
            employee["OverTime"],
            employee["YearsAtCompany"],
            employee["YearsInCurrentRole"],
            employee["YearsSinceLastPromotion"]
        ]
    })


    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# AGGREGATED RISK BY DEPARTMENT & JOB ROLE
# ============================================================

st.header(
    "3. Aggregated Risk by Department & Job Role"
)


aggregated_risk = (
    filtered_df
    .groupby(
        [
            "Department",
            "JobRole"
        ]
    )[
        "Attrition_Probability"
    ]
    .mean()
    .reset_index()
)


aggregated_risk[
    "Attrition_Probability"
] = (
    aggregated_risk[
        "Attrition_Probability"
    ].round(2)
)


aggregated_risk = (
    aggregated_risk
    .sort_values(
        "Attrition_Probability",
        ascending=False
    )
)


st.subheader(
    "Average Attrition Probability"
)


st.dataframe(
    aggregated_risk,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# BAR CHART
# ------------------------------------------------------------

aggregated_risk[
    "Department & Role"
] = (
    aggregated_risk[
        "Department"
    ]
    + " - "
    + aggregated_risk[
        "JobRole"
    ]
)


chart_data = (
    aggregated_risk
    .sort_values(
        "Attrition_Probability",
        ascending=True
    )
)


fig, ax = plt.subplots(
    figsize=(12, 8)
)


bars = ax.barh(
    chart_data[
        "Department & Role"
    ],
    chart_data[
        "Attrition_Probability"
    ]
)


ax.set_title(
    "Average Attrition Probability by Department & Job Role",
    fontsize=16,
    fontweight="bold"
)


ax.set_xlabel(
    "Average Attrition Probability (%)"
)


ax.set_ylabel(
    "Department & Job Role"
)


ax.bar_label(
    bars,
    labels=[
        f"{value:.1f}%"
        for value in
        chart_data[
            "Attrition_Probability"
        ]
    ],
    padding=4,
    fontsize=9
)


ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)


plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header(
    "4. Model Performance"
)


display_results = (
    model_results.copy()
)


for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]:

    display_results[column] = (
        display_results[column]
        * 100
    ).round(2)


st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)


st.success(
    f"Selected Model: {best_model_name}"
)


# ============================================================
# MODEL EXPLAINABILITY
# ============================================================

st.header(
    "5. Model Explainability"
)


st.subheader(
    f"Top 15 Features Based on Selected Model: "
    f"{best_model_name}"
)


# ------------------------------------------------------------
# GET SELECTED MODEL COMPONENTS
# ------------------------------------------------------------

selected_preprocessor = (
    best_model
    .named_steps["preprocessor"]
)

selected_classifier = (
    best_model
    .named_steps["classifier"]
)


feature_names = (
    selected_preprocessor
    .get_feature_names_out()
)


# ------------------------------------------------------------
# GET FEATURE IMPORTANCE
# ------------------------------------------------------------

if best_model_name == "Logistic Regression":

    importance_values = (
        selected_classifier
        .coef_[0]
    )

else:

    importance_values = (
        selected_classifier
        .feature_importances_
    )


# ------------------------------------------------------------
# FEATURE IMPORTANCE DATAFRAME
# ------------------------------------------------------------

feature_importance = pd.DataFrame({

    "Feature":
        feature_names,

    "Importance":
        importance_values
})


feature_importance[
    "Absolute Importance"
] = (
    feature_importance[
        "Importance"
    ].abs()
)


# ------------------------------------------------------------
# TOP 15
# ------------------------------------------------------------

top_features = (
    feature_importance
    .sort_values(
        "Absolute Importance",
        ascending=False
    )
    .head(15)
    .sort_values(
        "Absolute Importance",
        ascending=True
    )
)


# ------------------------------------------------------------
# FEATURE IMPORTANCE TABLE
# ------------------------------------------------------------

display_features = (
    top_features[
        [
            "Feature",
            "Importance"
        ]
    ]
    .copy()
)


display_features[
    "Importance"
] = (
    display_features[
        "Importance"
    ].round(3)
)


st.dataframe(
    display_features,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# FEATURE IMPORTANCE BAR CHART
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 7)
)


bars = ax.barh(
    top_features[
        "Feature"
    ],
    top_features[
        "Absolute Importance"
    ]
)


ax.set_title(
    f"Top 15 Features Based on {best_model_name}",
    fontsize=16,
    fontweight="bold"
)


ax.set_xlabel(
    "Feature Importance"
)


ax.set_ylabel(
    "Features"
)


ax.bar_label(
    bars,
    fmt="%.3f",
    padding=4,
    fontsize=9
)


ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3
)


plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


st.info(
    "Higher importance indicates that the feature "
    "has a stronger influence on the model's "
    "attrition prediction."
)


# ============================================================
# WHAT-IF SCENARIO EXPLORATION
# ============================================================

st.subheader(
    "What-If Scenario Exploration"
)


st.write(
    "Modify selected employee attributes to see "
    "how the predicted attrition probability changes."
)


if selected_employee is not None:

    # --------------------------------------------------------
    # ORIGINAL EMPLOYEE
    # --------------------------------------------------------

    employee = df[
        df["EmployeeID"]
        == selected_employee
    ].iloc[0]


    # --------------------------------------------------------
    # WHAT-IF INPUTS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        what_if_overtime = st.selectbox(
            "Overtime",
            ["No", "Yes"],
            index=(
                0
                if employee["OverTime"] == "No"
                else 1
            ),
            key="what_if_overtime"
        )


    with col2:

        what_if_job_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=int(
                employee[
                    "JobSatisfaction"
                ]
            ),
            key="what_if_job_satisfaction"
        )


    with col3:

        what_if_worklife = st.slider(
            "Work-Life Balance",
            min_value=1,
            max_value=4,
            value=int(
                employee[
                    "WorkLifeBalance"
                ]
            ),
            key="what_if_worklife"
        )


    # --------------------------------------------------------
    # CREATE SCENARIO COPY
    # --------------------------------------------------------

    what_if_employee = employee.copy()


    what_if_employee[
        "OverTime"
    ] = what_if_overtime


    what_if_employee[
        "JobSatisfaction"
    ] = what_if_job_satisfaction


    what_if_employee[
        "WorkLifeBalance"
    ] = what_if_worklife


    # --------------------------------------------------------
    # RECALCULATE ENGINEERED FEATURES
    # --------------------------------------------------------

    what_if_employee[
        "IncomeToExperienceRatio"
    ] = (
        what_if_employee[
            "MonthlyIncome"
        ]
        /
        (
            what_if_employee[
                "TotalWorkingYears"
            ] + 1
        )
    )


    what_if_employee[
        "PromotionDelay"
    ] = np.where(
        what_if_employee[
            "YearsSinceLastPromotion"
        ] >= 5,
        1,
        0
    )


    what_if_employee[
        "EngagementScore"
    ] = (
        what_if_employee[
            "JobInvolvement"
        ]
        + what_if_employee[
            "JobSatisfaction"
        ]
        + what_if_employee[
            "EnvironmentSatisfaction"
        ]
        + what_if_employee[
            "RelationshipSatisfaction"
        ]
        + what_if_employee[
            "WorkLifeBalance"
        ]
    ) / 5


    what_if_employee[
        "WorkloadStress"
    ] = np.where(

        (
            what_if_employee[
                "OverTime"
            ] == "Yes"
        )
        &
        (
            what_if_employee[
                "WorkLifeBalance"
            ] <= 2
        ),

        1,

        0
    )


    # --------------------------------------------------------
    # PREPARE WHAT-IF INPUT
    # --------------------------------------------------------

    what_if_input = pd.DataFrame(
        [what_if_employee]
    )


    what_if_input = (
        what_if_input
        .drop(
            columns=[
                "Attrition",
                "EmployeeID",
                "Attrition_Probability",
                "Risk_Category"
            ],
            errors="ignore"
        )
    )


    # --------------------------------------------------------
    # ORIGINAL RISK
    # --------------------------------------------------------

    original_probability = (
        employee[
            "Attrition_Probability"
        ]
    )


    # --------------------------------------------------------
    # WHAT-IF RISK
    # --------------------------------------------------------

    what_if_probability = (
        best_model
        .predict_proba(
            what_if_input
        )[0][1]
        * 100
    )


    # --------------------------------------------------------
    # DISPLAY COMPARISON
    # --------------------------------------------------------

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    result_col1.metric(
        "Original Risk",
        f"{original_probability:.2f}%"
    )


    result_col2.metric(
        "What-If Risk",
        f"{what_if_probability:.2f}%"
    )


    risk_change = (
        what_if_probability
        - original_probability
    )


    result_col3.metric(
        "Risk Change",
        f"{risk_change:+.2f}%"
    )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    if what_if_probability < original_probability:

        st.success(
            f"The what-if scenario reduces the "
            f"predicted attrition risk by "
            f"{abs(risk_change):.2f} percentage points."
        )

    elif what_if_probability > original_probability:

        st.warning(
            f"The what-if scenario increases the "
            f"predicted attrition risk by "
            f"{abs(risk_change):.2f} percentage points."
        )

    else:

        st.info(
            "The what-if scenario does not change "
            "the predicted attrition risk."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Employee Attrition Prediction and Risk Scoring System | "
    "Palo Alto Networks"
)
