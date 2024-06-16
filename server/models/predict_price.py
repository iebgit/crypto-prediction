from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler
import sys

# Ensure UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')
model = load_model('bitcoin_lstm_model.keras')

data = pd.read_csv('bitcoin_weekly_closing_prices.csv', encoding='utf-8')

# Extract the relevant column (price) and normalize it
prices = data['price'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

seq_length = 4  # Same as the sequence length used during training
last_sequence = scaled_prices[-seq_length:]

def predict_future_prices(model, last_sequence, steps, scaler):
    future_prices = []
    current_sequence = last_sequence.copy()

    for _ in range(steps):
        current_sequence_reshaped = np.reshape(current_sequence, (1, seq_length, 1))
        predicted_scaled_price = model.predict(current_sequence_reshaped)
        future_prices.append(predicted_scaled_price[0][0])
        
        # Append the predicted price to the current sequence and remove the oldest price
        current_sequence = np.append(current_sequence[1:], predicted_scaled_price, axis=0)

    # Inverse the scaling of the predicted prices
    future_prices = np.array(future_prices).reshape(-1, 1)
    future_prices = scaler.inverse_transform(future_prices)

    return future_prices


def get_weeks_between_dates(start_date, end_date):
    return (end_date - start_date).days // 7

# Define the start date and the target future date
start_date = datetime.datetime.strptime(data['timestamp'].iloc[-1], '%Y-%m-%d')
future_date = datetime.datetime(2024, 7, 31)  # Replace with your target date

# Calculate the number of weeks between the start date and the future date
weeks_ahead = get_weeks_between_dates(start_date, future_date)

# Predict the future prices
future_prices = predict_future_prices(model, last_sequence, weeks_ahead, scaler)

# Get the predicted price at the target future date
predicted_price_at_future_date = future_prices[-1][0]
print(f'Predicted price on {future_date.date()}: {predicted_price_at_future_date}')