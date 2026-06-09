Peak Load Prediction System
📌 Overview

This project focuses on analyzing electricity consumption data to identify peak load hours and predict future peak demand.

In this project, we study how electricity usage changes during different times of the day. The goal is to identify peak load hours when the demand is highest.

By analyzing historical data, we discover usage patterns and predict future peak hours. This helps in better energy planning, reduces the risk of power overload, and improves the efficiency of power distribution systems.

🎯 Objectives
Identify peak electricity load hours
Predict future peak demand
Analyze electricity usage patterns
Improve load management and energy planning

🛠️ Tech Stack
Python
Pandas, NumPy
Matplotlib
Scikit-Learn
Streamlit
MySQL
Jupyter Notebook
HTML/CSS

📂 Project Structure
electricity-demand-prediction/
│
├── data/                # Dataset
├── src/                 # Data preprocessing & model training
├── model/               # Trained ML model
├── app/                 # Streamlit app
├── notebooks/           # EDA and analysis
├── database/            # MySQL config
├── requirements.txt
└── README.md
⚙️ Features
🔮 Peak load prediction using Machine Learning
📈 Global Active Power trend visualization
⚡ Voltage trend analysis
📊 Hourly peak load detection
🌐 Interactive dashboard using Streamlit
🧠 Machine Learning Model
Model Used: Random Forest Regressor
Input Features:
Hour
Day
Month
Output:
Predicted electricity load

👉 Peak hours are identified based on highest predicted or actual load values

📊 Visualization
Global Active Power trend (full dataset)
Voltage trend (full dataset)
Hourly average consumption (bar chart)
Prediction-based line & bar charts


🚀 How to Run Project
1. Clone Repository
git clone https://github.com/saniyaansari7924/Peak-Load-Prediction.git
cd electricity-demand-prediction

2. Install Requirements
pip install -r requirements.txt

3. Train Model
cd src
python train_model.py
4. Run Streamlit App
cd ../app
streamlit run app.py
