import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# ------------------------------------
# Load Dataset
# ------------------------------------
df = pd.read_csv("electricity_dataset_150.csv")

st.set_page_config(
    page_title="Electricity Consumption Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit default UI
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Electricity Consumption & Billing Prediction Dashboard")

# ------------------------------------
# Train ML Model
# ------------------------------------
features = [
    "Number_of_Appliances",
    "AC_Usage_Hours",
    "Family_Members",
    "Average_Temperature",
    "Daily_Usage_Duration",
]

X = df[features]
y = df["Monthly_Consumption_kWh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------
# Sidebar (User Inputs)
# ------------------------------------
st.sidebar.header("🔧 Input Controls")

appliances = st.sidebar.slider("Number of Appliances", 1, 20, 6)
ac_hours = st.sidebar.slider("AC Usage Hours", 0, 24, 5)
family = st.sidebar.slider("Family Members", 1, 12, 4)
temperature = st.sidebar.slider("Average Temperature (°C)", 15, 45, 30)
daily_usage = st.sidebar.slider("Daily Usage Duration (hrs)", 1, 24, 10)

input_data = np.array([[appliances, ac_hours, family, temperature, daily_usage]])

predicted_kwh = model.predict(input_data)[0]
predicted_inr = predicted_kwh * 9
predicted_aed = predicted_inr * 0.044

# ------------------------------------
# Prediction Output
# ------------------------------------
st.subheader("📊 Prediction Output")

col1, col2, col3 = st.columns(3)

col1.metric("Predicted Consumption (kWh)", f"{predicted_kwh:.2f}")
col2.metric("Estimated Bill (INR)", f"₹ {predicted_inr:.0f}")
col3.metric("Estimated Bill (AED)", f"AED {predicted_aed:.2f}")

st.divider()

# ------------------------------------
# Dataset Viewer
# ------------------------------------
st.subheader("📁 Dataset Preview")
st.dataframe(df.head(15))

# ------------------------------------
# Correlation Heatmap
# ------------------------------------
st.subheader("📌 Feature Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.corr(), annot=True, cmap="viridis", ax=ax)
st.pyplot(fig)

# ------------------------------------
# Actual vs Predicted Chart
# ------------------------------------
st.subheader("📈 Actual vs Predicted (Test Data)")

y_pred = model.predict(X_test)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(y_test.values, label="Actual", linewidth=2)
ax2.plot(y_pred, label="Predicted", linewidth=2)
ax2.set_xlabel("Samples")
ax2.set_ylabel("Consumption (kWh)")
ax2.legend()
st.pyplot(fig2)

# ------------------------------------
# Scatter Plot
# ------------------------------------
st.subheader("📉 Scatter Plot: Monthly Bill vs Consumption")

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.scatter(df["Monthly_Consumption_kWh"], df["Monthly_Bill_INR"])
ax3.set_xlabel("Monthly Consumption (kWh)")
ax3.set_ylabel("Monthly Bill (INR)")
st.pyplot(fig3)

# ------------------------------------
# Footer
# ------------------------------------
st.markdown(
    "<h6 style='text-align:center; opacity:0.6;'>Created by Alan Chacolamannil Alex • Machine Learning Project</h6>",
    unsafe_allow_html=True
)
