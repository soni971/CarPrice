# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# =========================
# CLEAN UI STYLE (FIXED + MODERN)
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7");
        background-size: cover;
        background-attachment: fixed;
    }

    .block-container {
        background-color: rgba(0,0,0,0.80);
        padding: 25px;
        border-radius: 15px;
        color: white;
    }

    h1 {
        text-align: center;
        color: #00ffcc;
        font-size: 42px;
        font-weight: bold;
    }

    h3 {
        color: #00ffcc;
    }

    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-size: 16px;
        border-radius: 10px;
        width: 100%;
    }

    .stSelectbox, .stNumberInput {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("carprice.csv")

df.columns = df.columns.str.strip().str.lower()
df.replace("?", np.nan, inplace=True)

num_cols = ['horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.median(numeric_only=True), inplace=True)

# =========================
# ENCODING
# =========================
le = LabelEncoder()

df['fuel-type'] = le.fit_transform(df['fuel-type'])
df['engine-location'] = le.fit_transform(df['engine-location'])
df['engine-type'] = le.fit_transform(df['engine-type'])

# =========================
# FEATURES & MODEL
# =========================
X = df.drop(['price'], axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# NEW TITLE (CHANGED)
# =========================
st.title("🚗Car Price Prediction System")
st.write("Predict car price instantly using Machine Learning")

# =========================
# INPUT SECTION (FIXED LAYOUT)
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔧 Engine Info")
    fuel_type = st.selectbox("Fuel Type ⛽", ["gas", "diesel"])
    engine_location = st.selectbox("Engine Location", ["front", "rear"])
    engine_type = st.selectbox("Engine Type", ["dohc", "ohcv", "ohc", "l"])

with col2:
    st.subheader("⚙ Performance")
    horsepower = st.number_input("Horsepower", 40, 300, 100)
    peak_rpm = st.number_input("Peak RPM", 4000, 9000, 5000)

with col3:
    st.subheader("📊 Mileage")
    city_mpg = st.number_input("City MPG", 10, 60, 25)
    highway_mpg = st.number_input("Highway MPG", 10, 70, 30)

# =========================
# FIXED MAPPING (IMPORTANT)
# =========================
fuel_map = {"gas": 2, "diesel": 0}
engine_loc_map = {"front": 0, "rear": 1}
engine_type_map = {"dohc": 0, "ohcv": 1, "ohc": 2, "l": 3}

# =========================
# PREDICTION
# =========================
if st.button("🚗 Predict Car Price"):

    input_data = np.array([[
        fuel_map[fuel_type],
        engine_loc_map[engine_location],
        engine_type_map[engine_type],
        horsepower,
        peak_rpm,
        city_mpg,
        highway_mpg
    ]])

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Car Price: {prediction[0]:.2f}")

# =========================
# ONLY ONE GRAPH
# =========================
st.subheader("📊 Model Performance (Actual vs Predicted)")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color="cyan")
ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")

st.pyplot(fig)