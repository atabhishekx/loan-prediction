import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

st.set_page_config(page_title="Loan Approval Prediction", layout="wide")
st.title("Loan Approval Prediction")
st.caption("Decision Tree vs Random Forest on the Kaggle Loan Prediction Problem Dataset (614 applicants)")

DATA_FILE = "loan_data.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.drop("Loan_ID", axis=1)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


@st.cache_resource
def train_models(df):
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(exclude="number").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    return encoders, X, y_test, X_test, dt, rf


df = load_data()
encoders, X, y_test, X_test, dt_model, rf_model = train_models(df)

dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

# ---------------------------------------------------------
# Sidebar: interactive applicant form (prediction updates live)
# ---------------------------------------------------------
st.sidebar.header("Applicant Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.sidebar.slider("Applicant Income (per month)", 0, 20000, 5000, 100)
coapplicant_income = st.sidebar.slider("Coapplicant Income (per month)", 0, 10000, 0, 100)
loan_amount = st.sidebar.slider("Loan Amount (in thousands)", 0, 500, 150, 5)
term_options = [int(t) for t in sorted(df["Loan_Amount_Term"].unique())]
loan_term = st.sidebar.selectbox(
    "Loan Amount Term (months)", term_options, index=term_options.index(360)
)
credit_history = st.sidebar.selectbox("Credit History", ["Yes (1) - no delays", "No (0) - delays"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
model_choice = st.sidebar.radio("Model to use", ["Random Forest", "Decision Tree"])

input_data = {
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": float(loan_term),
    "Credit_History": 1.0 if credit_history.startswith("Yes") else 0.0,
    "Property_Area": property_area,
}

row = {}
for col in X.columns:
    val = input_data[col]
    if col in encoders:
        val = encoders[col].transform([val])[0]
    row[col] = val
sample = pd.DataFrame([row], columns=X.columns)

model = rf_model if model_choice == "Random Forest" else dt_model
pred = int(model.predict(sample)[0])
proba = model.predict_proba(sample)[0]

DISPLAY_NAMES = {
    "Gender": "Gender",
    "Married": "Married",
    "Dependents": "Dependents",
    "Education": "Education",
    "Self_Employed": "Self Employed",
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Coapplicant Income",
    "LoanAmount": "Loan Amount",
    "Loan_Amount_Term": "Loan Term",
    "Credit_History": "Credit History",
    "Property_Area": "Property Area",
}

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab_predict, tab_results, tab_visuals, tab_data = st.tabs(
    ["Prediction", "Results", "Visualisations", "Data"]
)

with tab_predict:
    st.header("Prediction for this applicant")
    if pred == 1:
        st.success(f"Loan APPROVED  (approval probability: {proba[1] * 100:.1f}%)")
    else:
        st.error(f"Loan REJECTED  (rejection probability: {proba[0] * 100:.1f}%)")

    st.subheader("Prediction Summary")
    st.dataframe(pd.DataFrame({
        "Metric": ["Model", "Approval Probability", "Rejection Probability", "Prediction"],
        "Value": [
            model_choice,
            f"{proba[1] * 100:.1f}%",
            f"{proba[0] * 100:.1f}%",
            "APPROVED" if pred == 1 else "REJECTED",
        ],
    }), hide_index=True, width="stretch")

    st.subheader("Key factors considered")
    st.caption(f"Feature importance from the {model_choice} model - higher bar = more influence on this prediction")
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_imp = importance.iloc[0]
    lines = []
    for feature, imp in importance.head(5).items():
        name = DISPLAY_NAMES.get(feature, feature)
        bar_len = max(1, round(imp / top_imp * 30))
        bar = "█" * bar_len
        if imp >= 0.5 * top_imp:
            level = "High"
        elif imp >= 0.2 * top_imp:
            level = "Medium"
        else:
            level = "Lower"
        lines.append(f"{name:<18} {bar:<32} {imp * 100:5.1f}%  {level}")
    st.code("\n".join(lines))

    st.write(f"Model used: **{model_choice}**")
    st.dataframe(sample.T.rename(columns={0: "Encoded value"}))

with tab_results:
    st.header("Model performance (80/20 train-test split)")
    col1, col2 = st.columns(2)
    col1.metric("Decision Tree accuracy", f"{dt_acc * 100:.2f}%")
    col2.metric("Random Forest accuracy", f"{rf_acc * 100:.2f}%")

    with st.expander("Decision Tree - classification report", expanded=True):
        st.text(classification_report(y_test, dt_pred, target_names=["Rejected", "Approved"]))
    with st.expander("Random Forest - classification report", expanded=True):
        st.text(classification_report(y_test, rf_pred, target_names=["Rejected", "Approved"]))

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Decision Tree", "Random Forest"], [dt_acc * 100, rf_acc * 100],
                  color=["#4C9BE8", "#2ECC71"])
    ax.bar_label(bars, fmt="%.1f%%")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

with tab_visuals:
    st.header("Visualisations")
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(dt_model, feature_names=X.columns, class_names=["Rejected", "Approved"],
              filled=True, rounded=True, fontsize=9)
    ax.set_title("Decision Tree (depth = 4)")
    st.pyplot(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, dt_pred, display_labels=["Rejected", "Approved"], ax=axes[0], colorbar=False)
    axes[0].set_title("Decision Tree")
    ConfusionMatrixDisplay.from_predictions(
        y_test, rf_pred, display_labels=["Rejected", "Approved"], ax=axes[1], colorbar=False)
    axes[1].set_title("Random Forest")
    fig.suptitle("Confusion Matrices")
    fig.tight_layout()
    st.pyplot(fig)

    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.plot(kind="barh", color="#F39C12", ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    st.pyplot(fig)
    st.write("Top 3 important features:")
    st.dataframe(importance.sort_values(ascending=False).head(3))

with tab_data:
    st.header("Dataset")
    st.write(f"Shape: **{df.shape[0]} rows x {df.shape[1]} columns** (after cleaning)")
    st.dataframe(df.head(10))
    with st.expander("Missing values before cleaning"):
        raw = pd.read_csv(DATA_FILE).drop("Loan_ID", axis=1)
        st.dataframe(raw.isnull().sum().rename("Missing values"))
