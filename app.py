import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


# Load trained model

model = joblib.load("bike_demand_model.pkl")
x_test = joblib.load("x_test.pkl")
y_test = joblib.load("y_test.pkl")

y_test_pred = model.predict(x_test)


# Page Config

st.set_page_config(
    page_title="Bike Sharing Demand Prediction",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 Bike Sharing Demand Prediction")
st.write("Predict bike rental demand based on time, weather, and seasonal conditions")

# Sidebar Inputs

st.sidebar.header("Input Features")

# ---- Time & Calendar Details ----
st.sidebar.subheader("Time & Calendar Details")

hr = st.sidebar.slider("Hour (0–23)", 0, 23, 12)
day = st.sidebar.slider("Day of Month", 1, 31, 15)

year = st.sidebar.number_input("Year", min_value=2010, max_value=2030, value=2012)

month_str = st.sidebar.selectbox(
    "Month",
    [
        "January", "February", "March", "April",
        "May", "June", "July", "August",
        "September", "October", "November", "December"
    ]
)

month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

month = month_map[month_str]


weekday_str = st.sidebar.selectbox(
    "Weekday",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
}

Weekday = weekday_map[weekday_str]
is_weekend = 1 if Weekday in [5, 6] else 0

# ---- Weather & Work Info ----
st.sidebar.subheader("Weather & Work Info")

holiday_str = st.sidebar.selectbox(
    "Holiday",
    ["Not a Holiday", "Holiday"]
)
holiday = 1 if holiday_str == "Holiday" else 0

workingday_str = st.sidebar.selectbox(
    "Working Day",
    ["Working Day", "Non-Working Day"]
)
workingday = 1 if workingday_str == "Working Day" else 0

temp_raw = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

hum_raw = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

windspeed_raw = st.sidebar.number_input(
    "Windspeed (m/s)",
    min_value=0.0,
    max_value=20.0,
    value=3.0,
    step=0.1
)

temp = (temp_raw - (-10)) / (50 - (-10))
hum = hum_raw / 100
windspeed = windspeed_raw / 20


# ---- Season ----
st.sidebar.subheader("Season")

season = st.sidebar.selectbox(
    "Season",
    ["spring", "summer", "fall", "winter"]
)

season_fall = 1 if season == "fall" else 0
season_springer = 1 if season == "spring" else 0
season_summer = 1 if season == "summer" else 0
season_winter = 1 if season == "winter" else 0

# ---- Weather Situation ----
st.sidebar.subheader("Weather Condition")

weather = st.sidebar.selectbox(
    "Weather Condition",
    ["Clear", "Mist", "Light Snow", "Heavy Rain"]
)

weathersit_Clear = 1 if weather == "Clear" else 0
weathersit_Mist = 1 if weather == "Mist" else 0
weathersit_Light_Snow = 1 if weather == "Light Snow" else 0
weathersit_Heavy_Rain = 1 if weather == "Heavy Rain" else 0

# Model Input DataFrame 

input_df = pd.DataFrame({
    'hr': [hr],
    'holiday': [holiday],
    'workingday': [workingday],
    'temp': [temp],
    'hum': [hum],
    'windspeed': [windspeed],
    'day': [day],
    'month': [month],
    'year': [year],
    'Weekday': [Weekday],
    'is_weekend': [is_weekend],
    'season_fall': [season_fall],
    'season_springer': [season_springer],
    'season_summer': [season_summer],
    'season_winter': [season_winter],
    'weathersit_Clear': [weathersit_Clear],
    'weathersit_Heavy Rain': [weathersit_Heavy_Rain],
    'weathersit_Light Snow': [weathersit_Light_Snow],
    'weathersit_Mist': [weathersit_Mist]
})

# User Input Display 

display_df = pd.DataFrame({
    "Hour": [hr],
    "Day": [day],
    "Month": [month_str],
    "Year": [year],
    "Weekday": [weekday_str],
    "Holiday": [holiday_str],
    "Day Type": [workingday_str],
    "Season": [season.capitalize()],
    "Weather": [weather],
     "Temperature (°C)": [temp_raw],
    "Humidity (%)": [hum_raw],
    "Windspeed (km/h)": [windspeed_raw]
})

st.subheader("User Input Summary")
st.dataframe(display_df, use_container_width=True)


# Prediction

if st.button("🚴 Predict Bike Demand"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Bike Rental Demand: **{int(prediction[0])}** bikes")

st.subheader("📊 Model Performance: Actual vs Predicted")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(y_test.values, label="Actual Demand")
ax.plot(y_test_pred, label="Predicted Demand", linestyle="--")

ax.set_xlabel("Test Data Samples")
ax.set_ylabel("Bike Demand")
ax.set_title("Actual vs Predicted Bike Demand")
ax.legend()

st.pyplot(fig)

 
