# from datetime import datetime, timedelta
# import pandas as pd
# from typing import Dict
# import numpy as np
# import torch
# # from algorithms.Basic import Result
# from interfaces.algo import Result
# from interfaces import backtest

# # SimulatorParallel
# class Backtester:
#     def __init__(self, algo, initial_balance: int = 1000):
#         self._algo = algo
#         self._total_balance = initial_balance
#         self._balance = initial_balance
#         self._holdings = 0
    
#     def buy(self, price: float, amount: float):
#         self._balance -= price * amount
#         self._holdings += amount
        
#     def sell(self, price: float, amount: float):
#         self._balance += price * amount
#         self._holdings -= amount
        
#     def run(self, df, start_idx, length=100) -> Dict:
#         end_idx = start_idx + length
        
#         print(df.head())
        
#         display_df = df.iloc[start_idx:end_idx+1].copy()
#         features_df = df[self._algo.features()].copy()
#         # Updated datetime conversion
#         for col in features_df.select_dtypes(include=['datetime64']).columns:
#             features_df[col] = pd.to_numeric(features_df[col].astype(np.int64)) // 10**9
            
#         print(display_df.head())
            
#         self._balance = self._total_balance # Use initial balance from constructor
#         self._holdings = 0
#         buys = []
#         sells = []
#         holds = []
#         entry_price = 0
#         entry_idx = 0
#         trades = [] # Track all completed trades
           
#         for idx in range(0, len(display_df)-1):
#             test_datum = features_df.iloc[idx+start_idx].astype(float).values
#             test_sample = torch.FloatTensor(test_datum)
#             pred = self._algo.predict(self._algo.get_model(), test_sample)
            
#             current_price = display_df.iloc[idx]['close']
#             next_price = display_df.iloc[idx+1]['close']
            
#             open_time = display_df.iloc[idx]['open_time']
#             close_time = display_df.iloc[idx]['close_time']
            
#             profit_pct = ((current_price - entry_price) / entry_price) * 100
            
#             holding_period_limit = 5;
            
#             # Check stop loss if holding position
#             if self._holdings > 0:
#                 loss_pct = (current_price - entry_price) / entry_price
#                 if loss_pct <= -0.2:  # 0.5% stop loss
#                     # Sell at stop loss
#                     self.sell(current_price, self._holdings)
#                     sells.append({
#                         'price': current_price,
#                         'event': "stoploss",
#                         'date': close_time,
#                         'low': display_df.iloc[idx]['low'],
#                         'high': display_df.iloc[idx]['high'],
#                         'entry_idx': entry_idx
#                     })
#                     trades.append({
#                         'buy_price': entry_price,
#                         'sell_price': current_price,
#                         'pct_change': ((current_price - entry_price) / entry_price) * 100
#                     })
#                     continue
            
#             to_buy = pred['result'] == Result.TargetPositive and self._balance > 0
#             to_sell = self._holdings > 0 and profit_pct > 0.1 and holding_period_limit <= (idx - entry_idx)
            
#             # if pred['result'] == Result.Negative and profit_pct < 0.1:
#             #     to_sell = True
            
#             # Execute trading logic
#             if to_buy:
#                 # Buy
#                 amount = self._balance / current_price # Convert all USDT to asset
#                 self.buy(current_price, amount)
#                 entry_price = current_price
#                 entry_idx = idx
#                 buys.append({
#                     'price': current_price,
#                     'event': "buy",
#                     'date': open_time,
#                     'low': display_df.iloc[idx]['low'],
#                     'high': display_df.iloc[idx]['high'],
#                     'entry_idx': entry_idx
#                 })
#             elif to_sell:
#                 # Sell
#                 self.sell(current_price, self._holdings)
#                 sells.append({
#                     'price': current_price,
#                     'event': "sell",
#                     'date': close_time,
#                     'low': display_df.iloc[idx]['low'],
#                     'high': display_df.iloc[idx]['high'],
#                     'entry_idx': entry_idx
#                 })
#                 trades.append({
#                     'buy_price': entry_price,
#                     'sell_price': current_price,
#                     'pct_change': ((current_price - entry_price) / entry_price) * 100
#                 })
#             else:
#                 # Hold
#                 holds.append({
#                     'price': current_price,
#                     'event': "hold",
#                     'date': close_time,
#                     'entry_idx': entry_idx
#                 })
        
#         return {
#             'total_balance': self._total_balance,
#             'balance': self._balance,
#             'holdings': self._holdings,
#             'buys': buys,
#             'sells': sells,
#             'holds': holds,
#             'display_df': display_df,
#             'trades': trades,
#         }

#     def construct_results(self, results) -> Dict:
#         final_balance = results['balance'] + (results['holdings'] * results['display_df'].iloc[-1]['close'])
        
#         return {
#             'original': results['total_balance'],
#             'final': final_balance,
#             'buys': results['buys'],
#             'sells': results['sells'],
#             'holds': results['holds'],
#             'trades': results['trades'],
#             'display_df': results['display_df']
#         }
        
#     def rand(self, df, length=100) -> Dict:
#         ma_range = self._algo.get_calc_ma_range()
#         buf = ma_range[-1]
#         start_idx = np.random.randint(buf, len(df) - length)
#         end_idx = start_idx + length

#         res = self.run(df, start_idx, length)
#         return self.construct_results(res)


# from models.Random import Random
# from wallet.wallet import Wallet
# from datetime import datetime, timedelta
# from models.HMM import Sentiment
# import pandas as pd
# from typing import Union

# def backtest_model(
#     model: Random,
#     wallet: Wallet, 
#     dataset_file_path: str,
#     start_date: datetime,
#     end_date: datetime
# ) -> dict:
#     # Load and prepare dataset
#     df = pd.read_csv(dataset_file_path)
#     df['open_time'] = pd.to_datetime(df['open_time'])
#     df['close_time'] = pd.to_datetime(df['close_time'])
#     df = df[(df['open_time'] >= start_date) & (df['close_time'] <= end_date)]
    
#     trades_made = 0
#     initial_balance = wallet.get_balance()
#     current_position = None
    
#     # Iterate through each day
#     current_date = start_date
#     while current_date <= end_date:
#         for hour in range(24):
#             current_datetime = current_date + timedelta(hours=hour)
#             current_data = df[df['open_time'] == current_datetime]
            
#             print(f"Processing data for: {current_datetime}")
#             print(len(current_data))
            
#             break
            
#             if len(current_data) > 0:
#                 # Get model prediction
#                 prediction = model.predict()
#                 current_price = float(current_data['close'].iloc[0])
                
#                 # Execute trades based on prediction
#                 if prediction['prediction'] == Sentiment.Buy and current_position != 'long':  # Buy
#                     # Buy
#                     if current_position == 'short':
#                         wallet.add_balance(current_price)
#                     wallet.sub_balance(current_price)
#                     current_position = 'long'
#                     trades_made += 1
                    
#                 elif prediction['prediction'] == Sentiment.Sell and current_position != 'short':  # Sell
#                     # Sell
#                     if current_position == 'long':
#                         wallet.add_balance(current_price)
#                     wallet.sub_balance(current_price)
#                     current_position = 'short'
#                     trades_made += 1
#                 elif prediction['prediction'] == Sentiment.Hold:
#                     # Hold
#                     pass
        
#         current_date += timedelta(days=1)
    
#     # Close any open positions
#     if current_position is not None:
#         final_price = float(df.iloc[-1]['close'])
#         if current_position == 'long':
#             wallet.add_balance(final_price)
#         else:
#             wallet.add_balance(final_price)
    
#     final_balance = wallet.get_balance()
    
#     return {
#         'initial_balance': initial_balance,
#         'final_balance': final_balance,
#         'total_trades': trades_made,
#         'profit': final_balance - initial_balance,
#         'return_pct': ((final_balance - initial_balance) / initial_balance) * 100,
#         'wallet': wallet
#     }
