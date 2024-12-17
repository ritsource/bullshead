import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('../scraper/binance_data_merged/klines/30m/merged.csv')
df = df[['close']] # Use closing prices
data = df.values.astype('float32')

# Normalize the data
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Parameters
sequence_length = 60
split_ratio = 0.8

# Create sequences
X, y = create_sequences(data_normalized, sequence_length)

# Split into train and test sets
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Initialize virtual wallet
wallet = {
    'usd': 1000.0,
    'btc': 0.0
}

# Trading simulation
trading_data = data_normalized[-len(X_test):]  # Use test data for trading simulation
current_position = None

for i in range(len(trading_data) - sequence_length - 5):  # -5 to ensure we have enough data for predictions
    sequence = trading_data[i:i+sequence_length]
    sequence = sequence.reshape((1, sequence_length, 1))
    
    # Predict next 5 values for buy signal
    next_5_pred = []
    temp_seq = sequence.copy()
    for _ in range(5):
        next_val = model.predict(temp_seq)
        next_5_pred.append(next_val[0][0])
        temp_seq = np.roll(temp_seq, -1)
        temp_seq[0][-1] = next_val
        
    # Predict next 3 values for sell signal    
    next_3_pred = next_5_pred[:3]
    
    # Convert predictions back to real prices
    current_price = scaler.inverse_transform([[trading_data[i+sequence_length]]])[0][0]
    next_5_prices = scaler.inverse_transform([[x] for x in next_5_pred])
    
    # Calculate price changes
    buy_signal = all(p > current_price for p in next_5_prices)
    sell_signal = all(p < current_price for p in next_5_prices[:3])
    
    # Execute trades
    if buy_signal and wallet['usd'] > 0 and current_position != 'buy':
        # Buy BTC
        btc_amount = wallet['usd'] / current_price
        wallet['btc'] = btc_amount
        wallet['usd'] = 0
        current_position = 'buy'
        print(f'Bought {btc_amount:.8f} BTC at ${current_price:.2f}')
        
    elif sell_signal and wallet['btc'] > 0 and current_position != 'sell':
        # Sell BTC
        usd_amount = wallet['btc'] * current_price
        wallet['usd'] = usd_amount
        wallet['btc'] = 0
        current_position = 'sell'
        print(f'Sold BTC for ${usd_amount:.2f}')

# Calculate final portfolio value
final_price = scaler.inverse_transform([[trading_data[-1]]])[0][0]
final_value = wallet['usd'] + (wallet['btc'] * final_price)
profit = final_value - 1000  # Initial investment was 1000 USD

print("\nTrading Results:")
print(f"Final portfolio value: ${final_value:.2f}")
print(f"Total profit: ${profit:.2f}")
print(f"Return on investment: {((final_value/1000)-1)*100:.2f}%")

# Plot results
plt.figure(figsize=(15,6))
plt.plot(scaler.inverse_transform(trading_data[sequence_length:]), label='Actual Price')
plt.title('Bitcoin Price During Trading Period')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
