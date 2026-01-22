import yfinance as yf
import pandas as pd

def fetch_data(tickers, start_date, end_date):
    """
    Inputs:
    - tickers: A list of strings (e.g., ['AAPL', 'MSFT'])
    - start_date: String 'YYYY-MM-DD'
    - end_date: String 'YYYY-MM-DD'
    
    Output:
    - A DataFrame of 'Adj Close' prices.
    """
    print(f"Attempting to download data for: {tickers}")
    
    # 1. Download the raw data
    # We allow auto_adjust=False so we can explicitly look for 'Adj Close'
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # 2. Extract just the prices we need
    # Different versions of yfinance behave differently, so we check what we got.
    if 'Adj Close' in raw_data.columns:
        data = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        print("Note: 'Adj Close' not found, using 'Close'.")
        data = raw_data['Close']
    else:
        # If the structure is flat (single ticker), it might just be the dataframe itself
        data = raw_data

    # 3. VALIDATION: Check if data is empty
    if data is None or data.empty:
        print("Error: No data was downloaded. Check your internet or ticker symbols.")
        return None

    # 4. CLEANING
    # Drop rows where ALL columns are empty (market holidays)
    data = data.dropna(how='all')

    # Drop rows where ANY column is empty (to ensure we compare apples to apples)
    data = data.dropna()

    print(f"Data downloaded successfully. Shape: {data.shape}")
    return data