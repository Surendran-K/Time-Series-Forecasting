import streamlit as st
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
df = pd.read_csv('Nifty 50 Historical Data.csv')
df1 = df[['Date', 'Price']]
df1.rename(columns={'Date': 'ds', 'Price': 'y'}, inplace=True)
df1.sort_values(by='ds', inplace=True)
df1.reset_index(drop=True, inplace=True)
df1['ds'] = pd.to_datetime(df1['ds'], format='%d-%m-%Y')
if df1['y'].dtype == object:
    df1['y'] = df1['y'].str.replace(',', '').astype(float)

# Streamlit app
st.title('Facebook Prophet Time Series Forecasting')

# Add a date input to select the start date for training the model
start_date = st.date_input('Select the start date for training the model:', value=pd.to_datetime('2023-01-01'))
start_date = pd.to_datetime(start_date)

# Filter the data based on the selected start date
df1_filtered = df1[df1['ds'] >= start_date].copy()

# Initialize the Prophet model
model = Prophet()

# Fit the model to the data
model.fit(df1_filtered)

# Add a slider to select the number of days for future predictions
days = st.slider('Select the number of days for future predictions:', min_value=1, max_value=30, value=5)

# Create a dataframe for future predictions
future = model.make_future_dataframe(periods=days, freq='D')

# Make predictions
forecast = model.predict(future)

# Rename forecast columns for clarity and format
forecast.rename(columns={'ds': 'Date', 'yhat': 'Price', 'yhat_lower': 'Lower Price', 'yhat_upper': 'Upper Price'}, inplace=True)
forecast['Date'] = pd.to_datetime(forecast['Date'])
forecast['Date'] = forecast['Date'].dt.strftime('%d-%m-%Y')
forecast['Price'] = forecast['Price'].round(0).astype(int)
forecast['Lower Price'] = forecast['Lower Price'].round(0).astype(int)
forecast['Upper Price'] = forecast['Upper Price'].round(0).astype(int)

# Show forecasted data in a line graph with data points called out
st.subheader('Forecast Plot')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot only the future forecast points
future_forecast = forecast.tail(days)
dates = future_forecast['Date']
prices = future_forecast['Price']

ax.plot(dates, prices, marker='o', linestyle='-', color='b')

# Annotate each data point with its value
for i, txt in enumerate(prices):
    ax.annotate(txt, (dates.iloc[i], prices.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Future Forecasted Price')
st.pyplot(fig)
