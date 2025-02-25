import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Using Nifty 50 data as sample
df = pd.read_csv('Nifty 50 Historical Data.csv')
df1 = df[['Date', 'Price']]
df1.rename(columns={'Date': 'ds', 'Price': 'y'}, inplace=True)
df1.sort_values(by='ds', inplace=True)
df1.reset_index(drop=True, inplace=True)
df1['ds'] = pd.to_datetime(df1['ds'], format='%d-%m-%Y')
if df1['y'].dtype == object:
    df1['y'] = df1['y'].str.replace(',', '').astype(float)

# Using Streamlit app
st.title('Facebook Prophet Time Series Forecasting')

# Adding a date input to select the start date for training the model
start_date = st.date_input('The start date for training the model is set automatically. The earlier the start date, the better the prediction accuracy:', value=pd.to_datetime('2009-01-01'))
start_date = pd.to_datetime(start_date)

# Filtering the data based on the selected start date
df1_filtered = df1[df1['ds'] >= start_date].copy()

# Initializing the Prophet model
model = Prophet()

# Fitting the model to the data
model.fit(df1_filtered)

# Adding a slider to select the number of days for future predictions
days = st.slider('Select the number of days for future predictions:', min_value=1, max_value=30, value=5)

# Creating a dataframe for future predictions
future = model.make_future_dataframe(periods=days, freq='D')

# Ensuring predictions start from 1st Feb 2025
future = future[future['ds'] >= pd.Timestamp('2025-02-01')]

# Filtering out weekends
future = future[future['ds'].dt.dayofweek < 5]

# Making predictions
forecast = model.predict(future)

# Renaming forecast columns for clarity and changing to a readable format
forecast.rename(columns={'ds': 'Date', 'yhat': 'Price', 'yhat_lower': 'Lower Price', 'yhat_upper': 'Upper Price'}, inplace=True)
forecast['Date'] = pd.to_datetime(forecast['Date'])
forecast['Date'] = forecast['Date'].dt.strftime('%d-%m-%Y')
forecast['Price'] = forecast
