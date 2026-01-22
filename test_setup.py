import data_loader

# Test with a known "short" history stock (like Airbnb - ABNB) and a long one (Apple - AAPL)
prices = data_loader.fetch_data(['AAPL', 'ABNB'], '2020-01-01', '2023-01-01')

print(prices.head())

