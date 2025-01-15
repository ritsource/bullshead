from datetime import datetime, timedelta
import pandas as pd
from typing import Dict
import numpy as np
import torch
from algorithms.BasicClassification import Output
from interfaces.algo import Result
import uuid
from enum import Enum
from algorithms.types import Output

class TradeSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    
class Trade:
    def __init__(self, side, amount, price):
        self.side = side
        self.price = price
        
    def buy(self, amount, price):
        Trade(TradeSide.BUY, amount, price)
        
    def sell(self, amount, price):
        Trade(TradeSide.SELL, amount, price)

class Holding:
    def __init__(self, amount, price):
        self.id = uuid.uuid4()
        self.amount = amount
        self.bought_at = price

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Holding(self.amount + other, self.bought_at)
        elif isinstance(other, Holding):
            # Average the bought_at price weighted by amounts
            total_amount = self.amount + other.amount
            avg_price = (self.bought_at * self.amount + other.bought_at * other.amount) / total_amount
            return Holding(total_amount, avg_price)
        else:
            raise TypeError("Unsupported operand type")
            
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return Holding(self.amount - other, self.bought_at)
        elif isinstance(other, Holding):
            return Holding(self.amount - other.amount, self.bought_at)
        else:
            raise TypeError("Unsupported operand type")
            
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Holding(self.amount * other, self.bought_at)
        else:
            raise TypeError("Can only multiply by numbers")
            
    def __lt__(self, other):
        if isinstance(other, Holding):
            return self.amount < other.amount
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)
     
    def value(self, at_price):
        return self.amount * at_price
    
    def pct_change(self, at_price):
        return (at_price - self.bought_at) / self.bought_at

class Backtester:
    def __init__(self, algo):
        self._algo = algo
        self._balance = 0
        self._holdings = []
        self._default_stoploss_pct = 2
        self._holding_period_limit = 5
        self._history = []
        self._trades = []
    
    def buy(self, price: float, amount: float):
        self._balance -= price * amount
        self._holdings.append(Holding(amount, price))
        
    def sell(self, price: float, holdings: list[Holding]):
        amount = sum(holdings)
        self._balance += price * amount
        self._holdings = [h for h in self._holdings if h.id not in [h.id for h in holdings]]
    
    def record_history(self, call, price):
        self._history.append({
            'call': call,
            'price': price,
        })
    
    def record_trade(self, side, amount, price, event=None):
        self._trades.append(Trade(side, amount, price))

    def run(self, df, start, length=100, initial_balance: int = 1000, stoploss_pct: float = 2) -> Dict:
        end = start + length
        
        stoploss_pct = stoploss_pct if isinstance(stoploss_pct, float) and 0 <= stoploss_pct <= 100 else self._default_stoploss_pct
        
        print(f"Running with stoploss of {stoploss_pct}%")
        
        # print(df.head())
        
        display_df = df.iloc[start:end+1].copy() # display_df
        features_df = df[self._algo.features()].copy() # features_df
        
        # Updated datetime conversion
        for col in features_df.select_dtypes(include=['datetime64']).columns:
            features_df[col] = pd.to_numeric(features_df[col].astype(np.int64)) // 10**9
            
        # print(display_df.head())

        trades = []
        
        self._balance = initial_balance
        self._holdings = []
           
        for idx in range(0, len(display_df)-1):
            test_datum = features_df.iloc[idx+start].astype(float).values
            test_sample = torch.FloatTensor(test_datum)
            out = self._algo.predict(test_sample)
            # print(f"out: {out}")
            
            current_price = display_df.iloc[idx]['open']
            # next_price = display_df.iloc[idx+1]['open']
            
            # open_time = display_df.iloc[idx]['open_time']
            # close_time = display_df.iloc[idx]['close_time']
            
            res = None
            
            no_money = self._balance <= 0 and len(self._holdings) <= 0 and sum(self._holdings) <= 0
            
            if no_money:
                print("No money to invest, balance is 0 and no surrent holdings with value")
                break
            
            prev_vals = [hold.value(hold.bought_at) for hold in self._holdings]
            curr_vals = [hold.value(current_price) for hold in self._holdings]
            
            trigger_stoploss = any((cv - pv) / pv <= -0.2 for i, cv in enumerate(curr_vals) if i < len(prev_vals) and (pv := prev_vals[i]))
            
            call = out
            
            if trigger_stoploss:
                call = "stoploss"
                holds = self._holdings
                self.sell(current_price, holds)
                self.record_trade("sell", sum(holds), current_price)

            elif out == Output.BUY:
                amount = self._balance / current_price # Convert all USDT to asset
                self.buy(current_price, amount)
                self.record_trade(call, amount, current_price)
                
            elif out == Output.SELL:
                holds = self._holdings
                self.sell(current_price, holds)
                self.record_trade(call, sum(holds), current_price)
                
            elif out == Output.HOLD:
                call = "hold"
            
            # print(f"call: {call}")
            # print(f"out: {out}")
            self.record_history(call, current_price)

        return self

    def result(self):
        return {
            'original': self._balance + sum(hold.value(hold.bought_at) for hold in self._holdings),
            'final': self._balance + sum(hold.value(hold.bought_at) for hold in self._holdings),
            'trades': self._trades,
            'history': self._history,
        }
        
    def rand(self, df, length=100) -> Dict:
        buf = self._algo.min_required_context_length()
        start = np.random.randint(buf, len(df) - length)
        end = start + length

        return self.run(df, start, length)