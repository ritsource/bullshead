import finplot as fplt
import os
import pandas as pd

print(os.getcwd())

start_date = '2024-01-01'
end_date = '2024-11-30'

# Check if file exists
file_path = '../scraper/binance_data_merged/klines/30m/merged.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError("CSV file not found at path: " + file_path)

# Read the CSV data
df = pd.read_csv(file_path)

# Convert timestamp to datetime
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df.set_index('open_time', inplace=True)

# Plot candlestick chart using finplot
fplt.candlestick_ochl(df[['open', 'close', 'high', 'low']])
fplt.show()
