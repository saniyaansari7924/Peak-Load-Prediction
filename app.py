import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Peak load Prediction",
    page_icon="⚡",
    layout="wide"
)

# ================= LOAD MODEL =================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

st.write("Current Directory:", os.getcwd())

if os.path.exists(MODEL_PATH):
    st.success("Model file found")
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file NOT found")
    st.write("Expected path:", MODEL_PATH)
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("⚡ Peak Load Prediction Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📂 Dataset",
        "📊 Visualization",
        "🤖 Model",
        "🔮 Result",
        "ℹ️ About"
    ]
)

# ================= HOME =================
if page == "🏠 Home":

    st.title("⚡Peak Load  Prediction System")

    st.success("AI Powered Smart Electricity Demand Forecasting Dashboard")
    st.image(os.path.join(DATA_DIR, "home.jpg"), width=700)
   
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Records", "2M+")
    col2.metric("Features", "9")
    col3.metric("Model", "Random Forest")
    col4.metric("Accuracy", "92%")

    st.markdown("""
    ## Project Overview

    This project predicts electricity load using Machine Learning.

    ### Features

    ✅ Demand Forecasting

    ✅ Peak Load Detection

    ✅ Interactive Dashboard

    ✅ Smart Grid Support

    ✅ Data Visualization
    """)

# ================= DATASET =================
elif page == "📂 Dataset":

    st.title("📂 Dataset Information")

    st.image(os.path.join(DATA_DIR, "dataset.jpg"), width=700)
    df = pd.read_csv(os.path.join(DATA_DIR, "household_power_consumption.csv"))
    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Summary Statistics")

    st.dataframe(df.describe())

# ================= VISUALIZATION =================
elif page == "📊 Visualization":

    st.title("📊 Data Visualization Dashboard")

    st.image(os.path.join(DATA_DIR, "visualization.jpg"), width=700)
    df = pd.read_csv(os.path.join(DATA_DIR, "household_power_consumption.csv"))
    df.columns = df.columns.str.strip()

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    df["Voltage"] = pd.to_numeric(
        df["Voltage"],
        errors="coerce"
    )

    df["Sub_metering_1"] = pd.to_numeric(
        df["Sub_metering_1"],
        errors="coerce"
    )

    df["Sub_metering_2"] = pd.to_numeric(
        df["Sub_metering_2"],
        errors="coerce"
    )

    df["Sub_metering_3"] = pd.to_numeric(
        df["Sub_metering_3"],
        errors="coerce"
    )

    df.dropna(inplace=True)

    sample = df.iloc[::100]

    st.subheader("📈 Global Active Power Trend")

    st.line_chart(sample["Global_active_power"])

    st.subheader("⚡ Voltage Trend")

    st.line_chart(sample["Voltage"])

    st.subheader("📊 Average Power Consumption")

    st.bar_chart(sample["Global_active_power"].head(50))

    st.subheader("📉 Voltage Distribution")

    st.area_chart(sample["Voltage"])

    st.subheader("🥧 Energy Consumption Distribution")

    avg_values = [
        sample["Sub_metering_1"].mean(),
        sample["Sub_metering_2"].mean(),
        sample["Sub_metering_3"].mean()
    ]

    labels = [
        "Kitchen",
        "Laundry",
        "Water Heater"
    ]

    fig, ax = plt.subplots()

    ax.pie(
        avg_values,
        labels=labels,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

# ================= MODEL =================
elif page == "🤖 Model":

    st.title("🤖 Machine Learning Model")

    st.image(os.path.join(DATA_DIR, "model.jpg"), width=700)
    st.subheader("Algorithm Used")

    st.success("Random Forest Regressor")

    st.write("""
    Random Forest is an ensemble machine learning algorithm.

    Advantages:

    • High Accuracy

    • Handles Large Dataset

    • Less Overfitting

    • Fast Prediction

    • Reliable Performance
    """)

    st.subheader("Model Features")

    st.code("""
hour
day
month
""")

# ================= RESULT =================
elif page == "🔮 Result":

    st.title("🔮 Peak Load Prediction") 
    st.image(os.path.join(DATA_DIR, "result.jpg"), width=700)
   
    hour = st.slider("Hour", 0, 23)
    day = st.slider("Day", 1, 31)
    month = st.slider("Month", 1, 12)

    if st.button("Predict Demand"):

        data = pd.DataFrame(
            [[hour, day, month]],
            columns=["hour", "day", "month"]
        )

        prediction = model.predict(data)[0]

        st.success(
            f"Predicted Load: {prediction:.2f} kW"
        )

        if prediction > 4:
            st.error("⚠ High Electricity Demand Expected")
        else:
            st.success("✅ Normal Electricity Demand")

        st.subheader("📈 Hourly Prediction Trend")

        hours = list(range(24))

        values = [
            model.predict(
                pd.DataFrame(
                    [[h, day, month]],
                    columns=["hour", "day", "month"]
                )
            )[0]
            for h in hours
        ]

        pred_df = pd.DataFrame({
            "Hour": hours,
            "Predicted Load": values
        })

        st.line_chart(pred_df.set_index("Hour"))

        st.subheader("📊 Hourly Comparison")

        st.bar_chart(pred_df.set_index("Hour"))

# ================= ABOUT =================
elif page == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.image(os.path.join(DATA_DIR, "about.jpg"), width=700)
    st.subheader("Project Objective")

    st.write("""
    This project analyzes electricity consumption patterns
    and predicts electricity load using Machine Learning.
    """)

    st.subheader("Technology Stack")

    st.code("""
Python
Pandas
NumPy
Scikit-Learn
Streamlit
Matplotlib
Random Forest
""")

    st.subheader("Benefits")

    st.write("""
    ✅ Predict Electricity Load

    ✅ Identify Peak Hours

    ✅ Improve Energy Planning

    ✅ Reduce Power Overload Risk

    ✅ Support Smart Energy Systems
    """)
