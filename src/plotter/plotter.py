import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from constants import colors
import pandas as pd
import numpy as np

def plot_trades(display_df, trades, buys, sells):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    print("LOL")
    print(display_df.head())
    
    # Plot price data
    ax.plot(display_df['close_time'], display_df['close'], color=colors.GRAY, alpha=0.6, label='Price')
    
    # Plot buy points
    for buy in buys:
        # Convert timestamp to UTC
        date = buy['date'].tz_localize(None).tz_localize('UTC')
        ax.scatter(date, buy['price'], color=colors.GREEN, marker='^', s=20, 
                  label='Buy' if 'Buy' not in ax.get_legend_handles_labels()[1] else '')
        
    # Plot sell points 
    for sell in sells:
        # Convert timestamp to UTC
        date = sell['date'].tz_localize(None).tz_localize('UTC')
        color = colors.RED if sell['event'] == 'stoploss' else colors.RED
        label = 'Stop Loss' if sell['event'] == 'stoploss' else 'Sell'
        if label not in ax.get_legend_handles_labels()[1]:
            ax.scatter(date, sell['price'], color=color, marker='v', s=20, label=label)
        else:
            ax.scatter(date, sell['price'], color=color, marker='v', s=20)

    # Customize plot
    ax.set_title('Trading Activity')
    ax.set_xlabel('close_time')
    ax.set_ylabel('close')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
    
    
def plot_trades_on_candle(display_df, trades, buys, sells):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot candlesticks
    for i in range(len(display_df)):
        # Plot the candlestick body
        color = colors.GREEN if display_df['close'].iloc[i] > display_df['open'].iloc[i] else colors.RED
        ax.vlines(x=display_df['close_time'].iloc[i], 
                 ymin=min(display_df['open'].iloc[i], display_df['close'].iloc[i]),
                 ymax=max(display_df['open'].iloc[i], display_df['close'].iloc[i]),
                 color=color, linewidth=4)
        
        # Plot the wicks
        ax.vlines(x=display_df['close_time'].iloc[i], 
                 ymin=display_df['low'].iloc[i], 
                 ymax=display_df['high'].iloc[i],
                 color=color, linewidth=1)
    
    # Plot buy points
    for buy in buys:
        # Convert timestamp to UTC
        date = buy['date'].tz_localize(None).tz_localize('UTC')
        ax.text(date, buy['price'], f'({buy["entry_idx"]})', color=colors.GREEN, fontsize=7, weight='bold',
               horizontalalignment='center', verticalalignment='bottom',
               bbox=dict(facecolor='white', edgecolor=colors.GREEN, alpha=0.8))
        
    # Plot sell points
    for sell in sells:
        # Convert timestamp to UTC 
        date = sell['date'].tz_localize(None).tz_localize('UTC')
        # label = ('STOP' if sell['event'] == 'stoploss' else 'SELL') + f'({sell["entry_idx"]})'
        label = ('*' if sell['event'] == 'stoploss' else '') + f'({sell["entry_idx"]})'
        ax.text(date, sell['price'], label, color=colors.RED, fontsize=7, weight='bold',
               horizontalalignment='center', verticalalignment='top',
               bbox=dict(facecolor='white', edgecolor=colors.RED, alpha=0.8))

    # Customize plot
    ax.set_title('Trading Activity (Candlestick)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()