import requests
import pandas as pd
from datetime import datetime

# Function to fetch historical price data for Bitcoin
def fetch_bitcoin_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {
        'vs_currency': 'usd',
        'days': 365,  # 365 days of historical data
        'interval': 'daily'  # Daily interval
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()['prices']
    else:
        raise Exception(f'Failed to fetch data. Status code: {response.status_code}')

# Function to filter weekly closing prices
def filter_weekly_closing(prices):
    filtered_data = []
    current_week = None
    
    for i, price in enumerate(prices):
        timestamp = datetime.fromtimestamp(price[0] / 1000)  # Convert milliseconds to seconds
        
        price_data = {
            'timestamp_epoch': int(price[0] / 1000),  # Unix timestamp in seconds
            'timestamp': timestamp.strftime('%Y-%m-%d'),  # Formatted date as YYYY-MM-DD
            'price': price[1]
        }
        
        if current_week is None:
            current_week = timestamp.isocalendar()[1]  # Get ISO week number
            
        if i == len(prices) - 1 or timestamp.isocalendar()[1] != datetime.fromtimestamp(prices[i + 1][0] / 1000).isocalendar()[1]:
            # Add the last price of the week
            filtered_data.append(price_data)
        
    return filtered_data

# Function to save data as CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    try:
        # Fetch historical data
        bitcoin_data = fetch_bitcoin_data()
        
        # Filter weekly closing prices
        weekly_closing_data = filter_weekly_closing(bitcoin_data)
        
        # Save data as CSV
        save_to_csv(weekly_closing_data, 'bitcoin_weekly_closing_prices.csv')
        
        print('Data saved successfully as bitcoin_weekly_closing_prices.csv')
    
    except Exception as e:
        print(f'Error occurred: {e}')
