from models.Random import Random
from wallet.wallet import Wallet
from datetime import datetime, timedelta
from models.HMM import Sentiment
import pandas as pd
from typing import Union

def backtest_model(
    model: Random,
    wallet: Wallet, 
    dataset_file_path: str,
    start_date: datetime,
    end_date: datetime
) -> dict:
    # Load and prepare dataset
    df = pd.read_csv(dataset_file_path)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df['close_time'] = pd.to_datetime(df['close_time'])
    df = df[(df['open_time'] >= start_date) & (df['close_time'] <= end_date)]
    
    trades_made = 0
    initial_balance = wallet.get_balance()
    current_position = None
    
    # Iterate through each day
    current_date = start_date
    while current_date <= end_date:
        for hour in range(24):
            current_datetime = current_date + timedelta(hours=hour)
            current_data = df[df['open_time'] == current_datetime]
            
            print(f"Processing data for: {current_datetime}")
            print(len(current_data))
            
            break
            
            if len(current_data) > 0:
                # Get model prediction
                prediction = model.predict()
                current_price = float(current_data['close'].iloc[0])
                
                # Execute trades based on prediction
                if prediction['prediction'] == Sentiment.Buy and current_position != 'long':  # Buy
                    # Buy
                    if current_position == 'short':
                        wallet.add_balance(current_price)
                    wallet.sub_balance(current_price)
                    current_position = 'long'
                    trades_made += 1
                    
                elif prediction['prediction'] == Sentiment.Sell and current_position != 'short':  # Sell
                    # Sell
                    if current_position == 'long':
                        wallet.add_balance(current_price)
                    wallet.sub_balance(current_price)
                    current_position = 'short'
                    trades_made += 1
                elif prediction['prediction'] == Sentiment.Hold:
                    # Hold
                    pass
        
        current_date += timedelta(days=1)
    
    # Close any open positions
    if current_position is not None:
        final_price = float(df.iloc[-1]['close'])
        if current_position == 'long':
            wallet.add_balance(final_price)
        else:
            wallet.add_balance(final_price)
    
    final_balance = wallet.get_balance()
    
    return {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_trades': trades_made,
        'profit': final_balance - initial_balance,
        'return_pct': ((final_balance - initial_balance) / initial_balance) * 100,
        'wallet': wallet
    }
