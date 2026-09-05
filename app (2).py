# ============================================================
# PALO ALTO NETWORKS - EMPLOYEE ATTRITION RISK DASHBOARD
# Best Tuned Model: Logistic Regression
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Palo Alto Networks Employee Attrition Risk Dashboard",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    file_path = Path(__file__).parent / "Palo Alto Networks.csv"

    if not file_path.exists():
        st.error(
            "Palo Alto Networks.csv was not found. "
            "Upload it to the same GitHub repository as app.py."
        )
        st.stop()

    return pd.read_csv(file_path)


df = load_data()

# ============================================================
# REQUIRED COLUMNS
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
    "OverTime",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error("Missing required columns:")
    st.write(missing)
    st.stop()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["IncomeToExperienceRatio"] = (
    df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
)

df["PromotionDelay"] = np.where(
    df["YearsSinceLastPromotion"] >= 5, 1, 0
)

df["EngagementScore"] = (
    df["JobInvolvement"]
    + df["JobSatisfaction"]
    + df["EnvironmentSatisfaction"]
    + df["RelationshipSatisfaction"]
    + df["WorkLifeBalance"]
) / 5

df["WorkloadStress"] = np.where(
    (df["OverTime"] == "Yes") & (df["WorkLifeBalance"] <= 2),
    1,
    0,
)

df["EmployeeID"] = range(1, len(df) + 1)

# ============================================================
# TARGET
# ============================================================

if df["Attrition"].dtype == "object":
    df["Attrition"] = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0, "1": 1, "0": 0})
    )
else:
    df["Attrition"] = pd.to_numeric(
        df["Attrition"], errors="coerce"
    )

df = df.dropna(subset=["Attrition"]).copy()
df["Attrition"] = df["Attrition"].astype(int)

# ============================================================
# FEATURES / TARGET
# ============================================================

X = df.drop(columns=["Attrition", "EmployeeID"])
y = df["Attrition"]

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
            ),
            categorical_features,
        ),
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
    stratify=y,
)

# ============================================================
# TUNED BEST MODEL
# ============================================================
# Your notebook result shows:
# Logistic Regression = 0.8128 Test ROC-AUC
# Random Forest       = 0.7989
# Gradient Boosting   = 0.7929
#
# The exact tuned hyperparameters were not present in app (1).py.
# Therefore this dashboard performs Logistic Regression tuning
# inside Streamlit instead of guessing the parameters.

@st.cache_resource
def train_tuned_model(X_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"],
        "classifier__solver": ["liblinear", "lbfgs"],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid


best_model, tuning_grid = train_tuned_model(
    X_train, y_train
)

best_model_name = "Tuned Logistic Regression"

# ============================================================
# TEST PREDICTIONS
# ============================================================

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(
        y_test, y_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_pred, zero_division=0
    ),
    "F1-Score": f1_score(
        y_test, y_pred, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, y_prob),
}

# ============================================================
# ALL EMPLOYEE PREDICTIONS
# ============================================================

X_all = df.drop(columns=["Attrition", "EmployeeID"])

df["Attrition_Probability"] = (
    best_model.predict_proba(X_all)[:, 1] * 100
)

df["Predicted_Attrition"] = best_model.predict(X_all)

df["Prediction_Outcome"] = np.where(
    df["Predicted_Attrition"] == 1,
    "Attrition",
    "No Attrition",
)

# ============================================================
# RISK FUNCTION
# ============================================================

def get_risk_category(probability, threshold):
    if probability < 30:
        return "Low Risk"
    elif probability < threshold:
        return "Medium Risk"
    return "High Risk"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛️ Dashboard Filters")

department_options = ["All"] + sorted(
    df["Department"].dropna().unique().tolist()
)

selected_department = st.sidebar.selectbox(
    "🏢 Department",
    department_options,
)

role_options = ["All"] + sorted(
    df["JobRole"].dropna().unique().tolist()
)

selected_role = st.sidebar.selectbox(
    "💼 Job Role",
    role_options,
)

risk_threshold = st.sidebar.slider(
    "⚠️ High-Risk Threshold (%)",
    min_value=30,
    max_value=90,
    value=60,
    step=5,
)

# Risk category is recalculated from the selected threshold.
df["Risk_Category"] = df["Attrition_Probability"].apply(
    lambda x: get_risk_category(x, risk_threshold)
)

filtered_df = df.copy()

if selected_department != "All":
    filtered_df = filtered_df[
        filtered_df["Department"] == selected_department
    ]

if selected_role != "All":
    filtered_df = filtered_df[
        filtered_df["JobRole"] == selected_role
    ]

employee_ids = sorted(filtered_df["EmployeeID"].tolist())

selected_employee = None

if employee_ids:
    selected_employee = st.sidebar.selectbox(
        "👤 Employee ID",
        employee_ids,
    )

# ============================================================
# TITLE
# ============================================================

st.title("📊 Palo Alto Networks Employee Attrition Risk Dashboard")

st.write(
    "Machine Learning-Based Employee Attrition Prediction "
    "and Risk Scoring System"
)

st.info(
    f"Best Tuned Model: {best_model_name} | "
    f"Test ROC-AUC obtained in this app: "
    f"{test_metrics['ROC-AUC']:.4f}"
)

# ============================================================
# 1. ATTRITION RISK DASHBOARD
# ============================================================

st.header("1. Attrition Risk Dashboard")

low_count = (
    filtered_df["Risk_Category"] == "Low Risk"
).sum()

medium_count = (
    filtered_df["Risk_Category"] == "Medium Risk"
).sum()

high_count = (
    filtered_df["Risk_Category"] == "High Risk"
).sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Employees", len(filtered_df))
c2.metric("High Risk", int(high_count))
c3.metric("Medium Risk", int(medium_count))
c4.metric("Low Risk", int(low_count))

risk_dist = pd.DataFrame({
    "Risk Category": [
        "Low Risk",
        "Medium Risk",
        "High Risk",
    ],
    "Employees": [
        low_count,
        medium_count,
        high_count,
    ],
})

st.subheader("Risk Category Distribution")

st.bar_chart(
    risk_dist.set_index("Risk Category")
)

# ============================================================
# 2. PREDICTION OUTCOME
# ============================================================

st.header("2. Prediction Outcome")

outcome_counts = (
    filtered_df["Prediction_Outcome"]
    .value_counts()
    .rename_axis("Prediction")
    .reset_index(name="Employees")
)

st.dataframe(
    outcome_counts,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 3. HIGH-RISK EMPLOYEES
# ============================================================

st.header("3. High-Risk Employees")

high_risk_table = filtered_df[
    filtered_df["Attrition_Probability"] >= risk_threshold
].copy()

high_risk_table["Risk_Percentage"] = (
    high_risk_table["Attrition_Probability"].round(2)
)

display_high_risk = high_risk_table[
    [
        "EmployeeID",
        "Department",
        "JobRole",
        "Attrition_Probability",
        "Risk_Category",
        "Risk_Percentage",
    ]
].sort_values(
    "Attrition_Probability",
    ascending=False,
).head(15)

st.dataframe(
    display_high_risk,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 4. INDIVIDUAL EMPLOYEE RISK PROFILE
# ============================================================

st.header("4. Individual Employee Risk Profile")

if selected_employee is not None:

    employee = df[
        df["EmployeeID"] == selected_employee
    ].iloc[0]

    employee_probability = float(
        employee["Attrition_Probability"]
    )

    employee_category = get_risk_category(
        employee_probability,
        risk_threshold,
    )

    p1, p2, p3 = st.columns(3)

    p1.metric(
        "Employee ID",
        int(selected_employee),
    )

    p2.metric(
        "Attrition Probability",
        f"{employee_probability:.2f}%",
    )

    p3.metric(
        "Risk Category",
        employee_category,
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
            "Years Since Last Promotion",
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
            employee["YearsSinceLastPromotion"],
        ],
    })

    st.subheader("Employee Details")

    st.dataframe(
        details,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# 5. DEPARTMENT-LEVEL RISK
# ============================================================

st.header("5. Department-Level Risk")

department_risk = (
    df.groupby("Department")
    .agg(
        Employee_Count=("Attrition_Probability", "count"),
        Average_Risk=("Attrition_Probability", "mean"),
        High_Risk_Count=(
            "Risk_Category",
            lambda x: (x == "High Risk").sum(),
        ),
    )
    .reset_index()
)

department_risk["High_Risk_Percentage"] = (
    department_risk["High_Risk_Count"]
    / department_risk["Employee_Count"]
    * 100
)

department_risk = department_risk.sort_values(
    "Average_Risk",
    ascending=False,
)

st.dataframe(
    department_risk.round(2),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 6. JOB ROLE-LEVEL RISK
# ============================================================

st.header("6. Job Role-Level Risk")

role_risk = (
    df.groupby("JobRole")
    .agg(
        Employee_Count=("Attrition_Probability", "count"),
        Average_Risk=("Attrition_Probability", "mean"),
        High_Risk_Count=(
            "Risk_Category",
            lambda x: (x == "High Risk").sum(),
        ),
    )
    .reset_index()
)

role_risk["High_Risk_Percentage"] = (
    role_risk["High_Risk_Count"]
    / role_risk["Employee_Count"]
    * 100
)

role_risk = role_risk.sort_values(
    "Average_Risk",
    ascending=False,
)

st.dataframe(
    role_risk.round(2),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 7. DEPARTMENT + JOB ROLE AGGREGATED RISK
# ============================================================

st.header("7. Aggregated Risk by Department & Job Role")

aggregated_risk = (
    df.groupby(["Department", "JobRole"])
    .agg(
        Employee_Count=("Attrition_Probability", "count"),
        Average_Risk=("Attrition_Probability", "mean"),
        High_Risk_Count=(
            "Risk_Category",
            lambda x: (x == "High Risk").sum(),
        ),
    )
    .reset_index()
)

aggregated_risk["High_Risk_Percentage"] = (
    aggregated_risk["High_Risk_Count"]
    / aggregated_risk["Employee_Count"]
    * 100
)

aggregated_risk = aggregated_risk.sort_values(
    "Average_Risk",
    ascending=False,
)

st.dataframe(
    aggregated_risk.round(2),
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# 8. TUNED MODEL PERFORMANCE
# ============================================================

st.header("8. Tuned Model Performance")

performance = pd.DataFrame({
    "Model": ["Tuned Logistic Regression"],
    "Accuracy": [test_metrics["Accuracy"]],
    "Precision": [test_metrics["Precision"]],
    "Recall": [test_metrics["Recall"]],
    "F1-Score": [test_metrics["F1-Score"]],
    "ROC-AUC": [test_metrics["ROC-AUC"]],
})

performance_display = performance.copy()

for col in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
]:
    performance_display[col] = (
        performance_display[col] * 100
    ).round(2)

st.dataframe(
    performance_display,
    use_container_width=True,
    hide_index=True,
)

st.success(
    f"Best tuned model: {best_model_name}"
)

st.write("Best parameters found by cross-validation:")

st.json(tuning_grid.best_params_)

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

st.header("9. Confusion Matrix - Best Tuned Model")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm)

ax.set_title(
    "Confusion Matrix - Tuned Logistic Regression"
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels([
    "No Attrition",
    "Attrition",
])

ax.set_yticklabels([
    "No Attrition",
    "Attrition",
])

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14,
        )

fig.colorbar(im, ax=ax)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True,
)

plt.close(fig)

# ============================================================
# 10. ROC CURVE
# ============================================================

st.header("10. ROC Curve - Best Tuned Model")

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob,
)

auc_value = roc_auc_score(
    y_test,
    y_prob,
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    fpr,
    tpr,
    label=(
        "Tuned Logistic Regression "
        f"(AUC = {auc_value:.4f})"
    ),
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Best Tuned Model")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True,
)

plt.close(fig)

# ============================================================
# 11. ACTUAL VS PREDICTED ATTRITION
# ============================================================

st.header("11. Actual vs Predicted Attrition")

actual_vs_predicted = pd.DataFrame({
    "Actual Attrition": y_test.values,
    "Predicted Attrition": y_pred,
})

actual_vs_predicted["Actual"] = (
    actual_vs_predicted["Actual Attrition"]
    .map({
        0: "No Attrition",
        1: "Attrition",
    })
)

actual_vs_predicted["Predicted"] = (
    actual_vs_predicted["Predicted Attrition"]
    .map({
        0: "No Attrition",
        1: "Attrition",
    })
)

comparison_counts = pd.crosstab(
    actual_vs_predicted["Actual"],
    actual_vs_predicted["Predicted"],
)

st.dataframe(
    comparison_counts,
    use_container_width=True,
)

# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

st.header("12. Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Attrition",
        "Attrition",
    ],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True,
)

# ============================================================
# 13. FEATURE EXPLAINABILITY
# ============================================================

st.header("13. Feature Explainability")

selected_preprocessor = (
    best_model.named_steps["preprocessor"]
)

selected_classifier = (
    best_model.named_steps["classifier"]
)

feature_names = (
    selected_preprocessor
    .get_feature_names_out()
)

coefficients = selected_classifier.coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
})

feature_importance["Absolute_Coefficient"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    "Absolute_Coefficient",
    ascending=False,
)

top_15 = feature_importance.head(15).copy()

top_15["Feature"] = (
    top_15["Feature"]
    .str.replace(
        "numerical__",
        "",
        regex=False,
    )
    .str.replace(
        "categorical__",
        "",
        regex=False,
    )
)

st.subheader("Top 15 Features")

st.dataframe(
    top_15[
        [
            "Feature",
            "Coefficient",
            "Absolute_Coefficient",
        ]
    ].round(4),
    use_container_width=True,
    hide_index=True,
)

plot_features = top_15.sort_values(
    "Absolute_Coefficient",
    ascending=True,
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

bars = ax.barh(
    plot_features["Feature"],
    plot_features["Absolute_Coefficient"],
)

ax.set_title(
    "Top 15 Features - Tuned Logistic Regression"
)

ax.set_xlabel("Absolute Coefficient")
ax.set_ylabel("Feature")

ax.bar_label(
    bars,
    fmt="%.3f",
    padding=3,
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.3,
)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True,
)

plt.close(fig)

st.info(
    "For Logistic Regression, the coefficient sign indicates "
    "the direction of association, while absolute coefficient "
    "is used to rank feature influence."
)

# ============================================================
# 14. WHAT-IF SCENARIO
# ============================================================

st.header("14. What-If Scenario Exploration")

st.write(
    "Change Job Satisfaction, Overtime, and Work-Life Balance "
    "for the selected employee and compare current risk with "
    "the what-if risk."
)

if selected_employee is not None:

    employee = df[
        df["EmployeeID"] == selected_employee
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        what_if_overtime = st.selectbox(
            "Overtime",
            ["No", "Yes"],
            index=(
                1
                if str(employee["OverTime"]) == "Yes"
                else 0
            ),
            key="what_if_overtime",
        )

    with col2:
        what_if_job_satisfaction = st.slider(
            "Job Satisfaction",
            min_value=1,
            max_value=4,
            value=int(
                employee["JobSatisfaction"]
            ),
            key="what_if_job_satisfaction",
        )

    with col3:
        what_if_worklife = st.slider(
            "Work-Life Balance",
            min_value=1,
            max_value=4,
            value=int(
                employee["WorkLifeBalance"]
            ),
            key="what_if_worklife",
        )

    what_if_employee = employee.copy()

    what_if_employee["OverTime"] = (
        what_if_overtime
    )

    what_if_employee["JobSatisfaction"] = (
        what_if_job_satisfaction
    )

    what_if_employee["WorkLifeBalance"] = (
        what_if_worklife
    )

    # Recalculate engineered features affected
    # by the what-if controls.
    what_if_employee[
        "IncomeToExperienceRatio"
    ] = (
        what_if_employee["MonthlyIncome"]
        / (
            what_if_employee["TotalWorkingYears"]
            + 1
        )
    )

    what_if_employee["PromotionDelay"] = int(
        what_if_employee[
            "YearsSinceLastPromotion"
        ] >= 5
    )

    what_if_employee["EngagementScore"] = (
        what_if_employee["JobInvolvement"]
        + what_if_employee["JobSatisfaction"]
        + what_if_employee["EnvironmentSatisfaction"]
        + what_if_employee["RelationshipSatisfaction"]
        + what_if_employee["WorkLifeBalance"]
    ) / 5

    what_if_employee["WorkloadStress"] = int(
        (
            what_if_employee["OverTime"]
            == "Yes"
        )
        and (
            what_if_employee["WorkLifeBalance"]
            <= 2
        )
    )

    what_if_input = pd.DataFrame(
        [what_if_employee]
    ).drop(
        columns=[
            "Attrition",
            "EmployeeID",
            "Attrition_Probability",
            "Predicted_Attrition",
            "Prediction_Outcome",
            "Risk_Category",
        ],
        errors="ignore",
    )

    original_probability = float(
        employee["Attrition_Probability"]
    )

    what_if_probability = float(
        best_model.predict_proba(
            what_if_input
        )[0, 1] * 100
    )

    risk_change = (
        what_if_probability
        - original_probability
    )

    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )

    result_col1.metric(
        "Original Risk",
        f"{original_probability:.2f}%",
    )

    result_col2.metric(
        "What-If Risk",
        f"{what_if_probability:.2f}%",
    )

    result_col3.metric(
        "Risk Change",
        f"{risk_change:+.2f}%",
    )

    original_category = get_risk_category(
        original_probability,
        risk_threshold,
    )

    what_if_category = get_risk_category(
        what_if_probability,
        risk_threshold,
    )

    cat1, cat2 = st.columns(2)

    cat1.write(
        f"**Original Risk Category:** "
        f"{original_category}"
    )

    cat2.write(
        f"**What-If Risk Category:** "
        f"{what_if_category}"
    )

    comparison = pd.DataFrame({
        "Scenario": [
            "Current",
            "What-If",
        ],
        "Attrition Probability (%)": [
            original_probability,
            what_if_probability,
        ],
    })

    st.subheader(
        "Current vs What-If Comparison"
    )

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        hide_index=True,
    )

    if risk_change < 0:
        st.success(
            "The what-if scenario reduces predicted "
            f"attrition risk by {abs(risk_change):.2f} "
            "percentage points."
        )

    elif risk_change > 0:
        st.warning(
            "The what-if scenario increases predicted "
            f"attrition risk by {abs(risk_change):.2f} "
            "percentage points."
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
