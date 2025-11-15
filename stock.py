import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st

# The path '/content/...' is for Google Colab. Use a local path.
# Make sure the model file is in the same folder as the script for this relative path to work.
try:
    model = load_model('Stock Predictions Model.keras')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.error("Please make sure 'Stock Predictions Model.keras' is in the same directory as your script.")
    st.stop()


st.header('Stock Trend Prediction')

stock = st.text_input('Enter the stock symbol', 'GOOG')
start_date = '2012-01-01'
end_date = '2025-12-31'

if st.button('Predict'):
    with st.spinner(f'Fetching data for {stock}...'):
        try:
            # Use yfinance to fetch the data
            data = yf.download(stock, start=start_date, end=end_date)
        except Exception as e:
            data = pd.DataFrame() # Create an empty dataframe on error
            st.error(f"An error occurred while fetching data: {e}")

    if data.empty:
        st.error(f"Could not fetch data for '{stock}'. Please check the symbol or your network connection.")
    else:
        st.success(f"Successfully fetched data for {stock}.")
        st.subheader('Raw Data')
        st.write(data.tail()) # Show the last few rows of the fetched data

        # Prepare training and testing dataframes
        data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
        data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))

        # Get the last 100 days of training data to use for the first prediction
        pas_100_days = data_train.tail(100)

        # Concatenate the last 100 days of training data with the test data
        data_test = pd.concat([pas_100_days, data_test], ignore_index=True)

        # Scale the test data
        data_test_scale = scaler.fit_transform(data_test)

        # Create sequences for prediction
        x = []
        y = []

        for i in range(100, data_test_scale.shape[0]):
          x.append(data_test_scale[i-100:i])
          y.append(data_test_scale[i,0])

        x, y = np.array(x), np.array(y)

        # Make predictions
        predict = model.predict(x)

        # Inverse transform the predictions and original values to get actual prices
        scale_factor = 1 / scaler.scale_[0]
        predict = predict * scale_factor
        y = y * scale_factor

        # Plot the results
        st.subheader('Original Price vs Predicted Price')
        fig1 = plt.figure(figsize=(10,6))
        plt.plot(y, 'g', label='Original Price')
        plt.plot(predict, 'r', label = 'Predicted Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig1)