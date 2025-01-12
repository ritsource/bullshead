from interfaces.fin import TradeSide

class BacktestHolding:
    def __init__(self, amount, price):
        self.amount = amount
        self.bought_at = price
    
    def value(self, at_price):
        return self.amount * at_price
    
    def pct_change(self, at_price):
        return (at_price - self.bought_at) / self.bought_at

class BacktestResultTrades:
    def __init__(self, side, price, holdings):
        self.side = side
        self.price = price
        self.pct_change = pct_change
        
    def buy(self, amount, price):
        self.side = TradeSide.BUY
        self.amount = amount
        self.price = price
        
    def sell(self, amount, price):
        self.side = TradeSide.SELL
        self.amount = amount
        self.price = price

class BacktestResult:
    def __init__(self, display_df, trades, buys, sells):
        self.display_df = display_df
        self.trades = trades
        self.buys = buys
        self.sells = sells