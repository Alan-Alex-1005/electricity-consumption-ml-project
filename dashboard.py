import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Electricity Usage Dashboard", layout="wide")

st.title("⚡ Electricity Usage Analytics Dashboard")
st.write("Explore trends and patterns in electricity consumption and billing.")

# =======================
# Load Dataset
# =======================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # =======================
    # Summary Stats
    # =======================
    st.subheader("📊 Dataset Summary")
    st.write(df.describe())

    # =======================
    # Filter Section
    # =======================
    st.sidebar.header("Filters")

    appliances_filter = st.sidebar.slider(
        "Number of Appliances", 
        int(df["Number_of_Appliances"].min()), 
        int(df["Number_of_Appliances"].max()),
        int(df["Number_of_Appliances"].mean())
    )

    family_filter = st.sidebar.slider(
        "Family Members", 
        int(df["Family_Members"].min()), 
        int(df["Family_Members"].max()),
        int(df["Family_Members"].mean())
    )

    filtered_df = df[
        (df["Number_of_Appliances"] >= appliances_filter) &
        (df["Family_Members"] >= family_filter)
    ]

    st.subheader("📌 Filtered Dataset")
    st.dataframe(filtered_df)

    # =======================
    # Line Chart: Consumption vs Bill
    # =======================
    st.subheader("📈 Monthly Consumption vs Bill")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df["Monthly_Consumption_kWh"], label="Consumption (kWh)")
    ax.plot(df["Monthly_Bill_INR"], label="Bill (INR)")
    ax.set_xlabel("Records")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    # =======================
    # Heatmap
    # =======================
    st.subheader("🔥 Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="Blues", ax=ax2)
    st.pyplot(fig2)

    # =======================
    # Relationship plot
    # =======================
    st.subheader("📉 AC Usage vs Monthly Consumption")

    fig3, ax3 = plt.subplots(figsize=(8,4))
    sns.scatterplot(data=df, x="AC_Usage_Hours", y="Monthly_Consumption_kWh", ax=ax3)
    st.pyplot(fig3)

else:
    st.info("Please upload your CSV file to generate the dashboard.")
