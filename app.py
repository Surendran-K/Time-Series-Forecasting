import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Using Nifty 50 data as sample
try:
    df = pd.read_csv('Nifty 50 Historical Data.csv')
except FileNotFoundError:
    st.error("Error: 'Nifty 50 Historical Data.csv' not found. Please make sure the file exists in the current directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Data preprocessing
try:
    df1 = df[['Date', 'Price']]
    df1.rename(columns={'Date': 'ds', 'Price': 'y'}, inplace=True)
    df1.sort_values(by='ds', inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df1['ds'] = pd.to_datetime(df1['ds'], format='%d-%m-%Y')
    
    # Handle comma in price values
    if df1['y'].dtype == object:
        df1['y'] = df1['y'].str.replace(',', '').astype(float)
except Exception as e:
    st.error(f"Error preprocessing data: {e}")
    st.stop()

# Using Streamlit app
st.title('Facebook Prophet Time Series Forecasting')

# Set fixed model parameters
changepoint_prior_scale = 0.05
seasonality_prior_scale = 1.0
daily_seasonality = "auto"
weekly_seasonality = True
yearly_seasonality = True

# Set training start date
start_date = pd.to_datetime('2009-01-01')
df1_filtered = df1[df1['ds'] >= start_date].copy()

# Fixed test size for evaluation metrics (20%)
test_size = 20
train_size = 100 - test_size
split_index = int(len(df1_filtered) * (train_size/100))
train_data = df1_filtered[:split_index].copy()
test_data = df1_filtered[split_index:].copy()

# Train model on training data
with st.spinner('Training the Prophet model... This may take a moment.'):
    # Initializing the Prophet model with fixed parameters
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality
    )
    
    # Add country holidays
    model.add_country_holidays(country_name='IN')
    
    # Fit the model to the training data
    model.fit(train_data)

# Calculate evaluation metrics on test data
future_test = model.make_future_dataframe(periods=0)
future_test = pd.concat([future_test, test_data[['ds']]])
forecast_test = model.predict(future_test)

# Merge forecasts with actual values for test data
eval_df = forecast_test[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
eval_df = pd.merge(eval_df, test_data, on='ds', how='inner')

# Calculate MAPE only
y_true = eval_df['y'].values
y_pred = eval_df['yhat'].values
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Display only MAPE
st.header('Model Evaluation')
st.metric("MAPE (%)", f"{mape:.2f}", 
          help="Mean Absolute Percentage Error - lower is better")

# Set forecast start date to February 1, 2025
forecast_start_date = pd.to_datetime('2025-02-01')

# Calculate number of days between last historical date and forecast start date
last_historical_date = df1_filtered['ds'].max()
days_to_forecast_start = (forecast_start_date - last_historical_date).days
if days_to_forecast_start < 0:
    st.warning(f"Warning: Forecast start date ({forecast_start_date.strftime('%Y-%m-%d')}) is before the last historical data point ({last_historical_date.strftime('%Y-%m-%d')}). Using last historical date instead.")
    forecast_start_date = last_historical_date
    days_to_forecast_start = 0

# Fixed number of days to forecast (10 days from Feb 1)
days = 10

# Creating a dataframe for future predictions starting from Feb 1
future = model.make_future_dataframe(periods=days_to_forecast_start + days, freq='D')

# Keep only dates from Feb 1 onwards for the forecast display
future_display
