import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import os
from statsmodels.tsa.arima.model import ARIMAResults

# Setup
st.set_page_config(page_title="Food Forecast", layout="wide")
st.title("📊 Food Consumption Forecast")

# Load full dataset
df = pd.read_csv("All_India_Index_july2019_20Aug2020_dec20_1_2.csv")
df = df[df['Sector']=="Rural"]
df["Date"] = pd.to_datetime(df["Month"] + ' ' + df["Year"].astype(str), format='%B %Y')
df['Date'] = pd.to_datetime(df['Date'])

# Sidebar dropdown for product selection
products = ["Meat and fish","Egg","Milk and products","Vegetables"]
selected_product = st.selectbox("Select a food category", products)

# Load the ARIMA model for the selected product
model_path = f"{selected_product}_arima_model.joblib"
if os.path.exists(model_path):
    model = load(model_path)

    # Prepare data for selected product
    df1 = df[["Date",selected_product]]
    df1.set_index("Date", inplace=True)
    monthly = df1.dropna(subset=[selected_product])

    # Forecast
    n_periods = 24
    forecast = model.get_forecast(steps=n_periods)
    predicted = forecast.predicted_mean
    conf_int = forecast.conf_int()

    # Fix forecast index
    last_date = monthly.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    predicted.index = future_dates
    conf_int.index = future_dates

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly, label="Historical", color='blue')
    ax.plot(predicted, label="Forecast", color='orange')
    ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
    ax.set_title(f"{selected_product} Forecast (Next {n_periods} Months)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning(f"No saved model found for '{selected_product}'. Please train and save the model first.")
