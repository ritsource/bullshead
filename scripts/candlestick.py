import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import os
from sklearn.preprocessing import MinMaxScaler

print(os.getcwd())

# Check if file exists
file_path = '../scraper/binance_data_merged/klines/30m/merged.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV file not found at path: {file_path}")

# Read the CSV data
df = pd.read_csv(file_path)

# Preprocess Data
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df['date_num'] = mdates.date2num(df['open_time'])  # Convert to matplotlib date format

# Select required columns for candlestick chart
data = df[['date_num', 'open', 'high', 'low', 'close', 'volume']].copy()

# Normalize the data
scaler = MinMaxScaler()
price_data = df[['close']].values
normalized_data = scaler.fit_transform(price_data)

# Print shape of data for debugging
print(f"Shape of data: {data.shape}")
print(f"Shape of normalized data: {normalized_data.shape}")

# Plotting
fig, ax = plt.subplots(figsize=(15, 8))
plt.subplots_adjust(bottom=0.25)  # Leave more space for controls

# Plot initial candlestick chart
candlestick_ohlc(ax, data.values, width=0.03, colorup='g', colordown='r', alpha=0.8)
ax.xaxis_date()  # Format the x-axis as dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
plt.xticks(rotation=45)
plt.grid(True)

# Add zoom slider
ax_zoom = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position the zoom slider
zoom_slider = Slider(ax_zoom, 'Zoom', 1, len(data), valinit=len(data), valstep=1)

# Add scroll slider
ax_scroll = plt.axes([0.2, 0.05, 0.65, 0.03])  # Position the scroll slider
scroll_slider = Slider(ax_scroll, 'Scroll', 0, len(data)-1, valinit=0, valstep=1)

# Add zoom buttons
ax_zoomin = plt.axes([0.9, 0.1, 0.08, 0.03])
ax_zoomout = plt.axes([0.9, 0.05, 0.08, 0.03])
btn_zoomin = Button(ax_zoomin, 'Zoom In')
btn_zoomout = Button(ax_zoomout, 'Zoom Out')

# Function to update the chart when any control changes
def update(val=None):
    zoom_level = int(zoom_slider.val)
    scroll_pos = int(scroll_slider.val)
    
    # Calculate visible range based on zoom and scroll
    start_index = scroll_pos
    end_index = min(start_index + zoom_level, len(data))
    visible_data = data.iloc[start_index:end_index]
    
    # Print shape of visible data for debugging
    print(f"Shape of visible_data: {visible_data.shape}")
    
    # Update scroll slider range based on zoom level
    scroll_slider.valmax = len(data) - zoom_level
    scroll_slider.ax.set_xlim(0, len(data) - zoom_level)
    
    # Clear the axis and re-plot
    ax.clear()
    candlestick_ohlc(ax, visible_data.values, width=0.03, colorup='g', colordown='r', alpha=0.8)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.grid(True)
    fig.canvas.draw_idle()

# Zoom button callbacks
def zoom_in(event):
    current_zoom = zoom_slider.val
    new_zoom = max(1, current_zoom * 0.5)  # Zoom in by reducing the range by half
    zoom_slider.set_val(new_zoom)
    
def zoom_out(event):
    current_zoom = zoom_slider.val
    new_zoom = min(len(data), current_zoom * 2)  # Zoom out by doubling the range
    zoom_slider.set_val(new_zoom)

# Connect controls to update function
zoom_slider.on_changed(update)
scroll_slider.on_changed(update)
btn_zoomin.on_clicked(zoom_in)
btn_zoomout.on_clicked(zoom_out)

plt.show()
