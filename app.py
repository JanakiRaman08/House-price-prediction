# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(page_title=" House Price Prediction", page_icon="", layout="centered")

st.title(" House Price Prediction App")
st.write("### Predict house price using ML model with geo & lifestyle features")

# =================== SAMPLE DATA ===================
data = {
    'bedrooms': [2, 3, 4, 3, 5, 2, 4, 3],
    'bathrooms': [1, 2, 3, 2, 4, 1, 3, 2],
    'sqft': [1000, 1500, 2000, 1700, 3000, 900, 2200, 1600],
    'year_built': [2000, 2010, 2005, 2008, 2015, 1995, 2012, 2003],
    'lat': [47.60, 47.61, 47.58, 47.62, 47.65, 47.59, 47.63, 47.60],
    'lon': [-122.33, -122.31, -122.35, -122.32, -122.36, -122.34, -122.30, -122.33],
    'crime_rate': [0.2, 0.1, 0.4, 0.3, 0.5, 0.6, 0.2, 0.1],
    'school_rating': [8, 9, 7, 8, 6, 5, 9, 8],
    'price': [400000, 550000, 650000, 600000, 900000, 350000, 720000, 580000]
}
df = pd.DataFrame(data)

# =================== MODEL TRAINING ===================
X = df[['bedrooms', 'bathrooms', 'sqft', 'year_built', 'lat', 'lon', 'crime_rate', 'school_rating']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.sidebar.write(" Model Performance")
st.sidebar.write(f"**RMSE:** {rmse:.2f}")
st.sidebar.write(f"**R² Score:** {r2:.2f}")

# =================== USER INPUT ===================
st.write("Enter House Details to Predict Price")

col1, col2 = st.columns(2)
with col1:
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)
    sqft = st.number_input("Sqft Living Area", 500, 10000, 1800)
    year_built = st.number_input("Year Built", 1950, 2025, 2010)
with col2:
    lat = st.number_input("Latitude", 47.50, 47.70, 47.61)
    lon = st.number_input("Longitude", -122.40, -122.25, -122.33)
    crime_rate = st.slider("Crime Rate (0 = safe, 1 = high)", 0.0, 1.0, 0.2)
    school_rating = st.slider("School Rating (1 = poor, 10 = excellent)", 1, 10, 8)

if st.button("Predict Price"):
    new_house = pd.DataFrame({
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'sqft': [sqft],
        'year_built': [year_built],
        'lat': [lat],
        'lon': [lon],
        'crime_rate': [crime_rate],
        'school_rating': [school_rating]
    })
    scaled_input = scaler.transform(new_house)
    predicted_price = model.predict(scaled_input)[0]

    st.success(f" **Predicted House Price:** ₹{predicted_price:,.0f}")
