import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
import os
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import numpy as np
from PIL import Image




def model_func(df, i,selected_city):
    print(selected_city)
    df1 = df[df['Sector']==selected_city]
    df1.set_index("Date", inplace=True)
    df1 = df1.dropna(subset=[i])

    train = df1.loc[:"2022-01-01"]
    test = df1.loc["2022-01-01":]

    auto_model = auto_arima(train[i], seasonal=True, m=12, trace=True)

    model = ARIMA(train[i], order=auto_model.order)  
    fitted_model = model.fit()

    forecast = fitted_model.get_forecast(steps=len(test))
    forecast_values = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    forecast = fitted_model.get_forecast(steps=24)
    predicted = forecast.predicted_mean
    conf_int = forecast.conf_int()

    n_periods = 24
    last_date = df1.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    predicted.index = future_dates
    conf_int.index = future_dates

    return df1,predicted,conf_int,selected_product,n_periods

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df1, label="Historical", color='blue')
    ax.plot(predicted, label="Forecast", color='orange')
    ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
    ax.set_title(f"{selected_product} Forecast (Next {n_periods} Months)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)




# Setup
st.set_page_config(page_title="Food Forecast", layout="wide")
st.title("📊 Food Consumption Forecast")

# Load full dataset
df = pd.read_csv("All_India_Index_july2019_20Aug2020_dec20_1_2.csv")

df["Date"] = pd.to_datetime(df["Month"] + ' ' + df["Year"].astype(str), format='%B %Y')
df['Date'] = pd.to_datetime(df['Date'])

selected_city = "Rural"
city = ["Rural","Urban","Urban+Rural"]
selected_city = st.selectbox("Select a city category",city )


# Sidebar dropdown for product selection
products = ["Meat and fish","Egg","Milk and products","Vegetables"]
selected_product = st.selectbox("Select a food category", products)

# Load the ARIMA model for the selected product
filename = f"my_plot_{selected_product}_{selected_city}.png"

if os.path.exists(filename):
    st.image(Image.open(filename), caption=f"{selected_product} in {selected_city}")

else:
    df1,predicted,conf_int,selected_product,n_periods = model_func(df, selected_product, selected_city)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df1[selected_product], label="Historical", color='blue')
    ax.plot(predicted, label="Forecast", color='orange')
    ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
    ax.set_title(f"{selected_product} Forecast (Next {n_periods} Months)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


