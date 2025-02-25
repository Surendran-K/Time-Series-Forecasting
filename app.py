import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

# Using Nifty 50 data as sample
try:
    df = pd.read_csv('Nifty 50 Historical Data.csv')
    st.success("Data loaded successfully!")
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

# Sidebar for parameters
st.sidebar.header("Model Parameters")

# Adding a date input to select the start date for training the model
start_date = st.sidebar.date_input(
    'Select start date for training:',
    value=pd.to_datetime('2009-01-01'),
    help="Earlier start date typically improves prediction accuracy"
)
start_date = pd.to_datetime(start_date)

# Filtering the data based on the selected start date
df1_filtered = df1[df1['ds'] >= start_date].copy()

# Model parameters
with st.sidebar.expander("Advanced Prophet Parameters"):
    changepoint_prior_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, step=0.001, 
                                        help="Controls flexibility of the trend. Higher values allow more trend changes.")
    seasonality_prior_scale = st.slider("Seasonality Prior Scale", 0.01, 10.0, 1.0, step=0.01,
                                       help="Controls flexibility of the seasonality. Higher values allow more seasonal fluctuations.")
    daily_seasonality = st.selectbox("Daily Seasonality", [True, False, "auto"], index=2,
                                    help="Whether to include daily seasonality component.")
    weekly_seasonality = st.selectbox("Weekly Seasonality", [True, False, "auto"], index=0,
                                     help="Whether to include weekly seasonality component.")
    yearly_seasonality = st.selectbox("Yearly Seasonality", [True, False, "auto"], index=0,
                                     help="Whether to include yearly seasonality component.")

# Split data into train and test for evaluation metrics
test_size = st.sidebar.slider('Test size (%) for evaluation metrics:', 5, 30, 20)
train_size = 100 - test_size
split_index = int(len(df1_filtered) * (train_size/100))
train_data = df1_filtered[:split_index].copy()
test_data = df1_filtered[split_index:].copy()

# Train model on training data
with st.spinner('Training the Prophet model... This may take a moment.'):
    # Initializing the Prophet model with selected parameters
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

# Calculate metrics
y_true = eval_df['y'].values
y_pred = eval_df['yhat'].values

mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Display evaluation metrics
st.header('Model Evaluation Metrics')
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric("MAPE (%)", f"{mape:.2f}")
metrics_col2.metric("MAE", f"{mae:.2f}")
metrics_col3.metric("RMSE", f"{rmse:.2f}")

# Adding a slider to select the number of days for future predictions
days = st.slider('Select the number of days for future predictions:', min_value=5, max_value=60, value=10)

# Creating a dataframe for future predictions
future = model.make_future_dataframe(periods=days, freq='D')

# Filter out weekends (Saturday=5, Sunday=6)
future['day_of_week'] = future['ds'].dt.dayofweek
future = future[future['day_of_week'] < 5].copy()
future.drop('day_of_week', axis=1, inplace=True)

# Making predictions
forecast = model.predict(future)

# Renaming forecast columns for clarity and changing to a readable format
forecast.rename(columns={'ds': 'Date', 'yhat': 'Price', 'yhat_lower': 'Lower Price', 'yhat_upper': 'Upper Price'}, inplace=True)
forecast['Date_fmt'] = pd.to_datetime(forecast['Date']).dt.strftime('%d-%m-%Y')

# Find where historical data ends and forecast begins
last_historical_date = df1_filtered['ds'].max()
forecast['Historical'] = forecast['Date'] <= last_historical_date

# Add the actual values to the forecast dataframe for plotting
historical_data = df1_filtered.copy()
historical_data.rename(columns={'ds': 'Date', 'y': 'Actual'}, inplace=True)
plot_data = pd.merge(forecast, historical_data, on='Date', how='left')

# Showing forecasted data in a line graph with data points called out
st.header('Visualization')
tab1, tab2 = st.tabs(["Full Forecast", "Future Predictions Only"])

with tab1:
    # Plot full forecast with historical data
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot historical data
    historical_mask = plot_data['Historical']
    ax1.plot(plot_data.loc[historical_mask, 'Date'], plot_data.loc[historical_mask, 'Actual'], 
             marker='o', color='black', alpha=0.6, label='Historical', linewidth=1, markersize=3)
    
    # Plot forecast line
    ax1.plot(plot_data['Date'], plot_data['Price'], 
             linestyle='-', color='blue', label='Forecast', linewidth=2)
    
    # Plot confidence interval
    ax1.fill_between(plot_data['Date'], plot_data['Lower Price'], plot_data['Upper Price'], 
                    color='blue', alpha=0.2, label='Confidence Interval')
    
    # Mark where the forecast begins
    ax1.axvline(x=last_historical_date, color='red', linestyle='--', label='Forecast Start')
    
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Historical Data and Forecast')
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig1)
    
    # Display forecast statistics
    st.write("Forecast Summary Statistics")
    forecast_future = forecast[~historical_mask].copy()
    st.dataframe(forecast_future[['Date_fmt', 'Price', 'Lower Price', 'Upper Price']].describe())

with tab2:
    # Create a new figure for future predictions only
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Get only future forecast (excluding historical data)
    future_mask = ~plot_data['Historical']
    future_forecast = plot_data[future_mask].copy()
    
    # Plot the forecast line
    ax2.plot(future_forecast['Date'], future_forecast['Price'], 
             marker='o', linestyle='-', color='blue', linewidth=2)
    
    # Plot confidence interval
    ax2.fill_between(future_forecast['Date'], future_forecast['Lower Price'], 
                     future_forecast['Upper Price'], color='blue', alpha=0.2)
    
    # Annotate each data point with its value
    for i, (date, price) in enumerate(zip(future_forecast['Date'], future_forecast['Price'])):
        ax2.annotate(f"{price:.0f}", (date, price), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Format the plot
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Future Price Forecast (Weekdays Only)')
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Show the forecasted values in a table
    st.subheader('Forecasted Values (Weekdays Only)')
    forecast_table = future_forecast[['Date_fmt', 'Price', 'Lower Price', 'Upper Price']].copy()
    forecast_table = forecast_table.round(0).astype({'Price': int, 'Lower Price': int, 'Upper Price': int})
    forecast_table.rename(columns={'Date_fmt': 'Date'}, inplace=True)
    st.table(forecast_table)

# Components plot
with st.expander("View Model Components"):
    fig3 = model.plot_components(forecast)
    st.pyplot(fig3)
