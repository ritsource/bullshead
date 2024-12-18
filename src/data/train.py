import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
import joblib
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

# Define a method to construct the input data X and Y
def construct_lstm_data(data, sequence_size, target_attr_idx):    
    # Initialize constructed data variables
    data_X = []
    data_y = []
    
    # Iterate over the dataset
    for i in range(sequence_size, len(data)):
        data_X.append(data[i-sequence_size:i,0:data.shape[1]])
        data_y.append(data[i,target_attr_idx])
        
    # Return constructed variables
    return np.array(data_X), np.array(data_y)

def train(data_dir="data/processed/", models_dir="models/lstm/", data_source_name="btc_usdt_30m"):
  # pass
  
  # Prepare data file location and name
  data_file_location = data_dir
  data_file_name_train = data_source_name + "_train_scaled.csv"
  data_file_name_validate = data_source_name + "_validate_scaled.csv"
  data_file_name_test = data_source_name + "_test_scaled.csv"

  # Load data files
  data_train_df = pd.read_csv(data_file_location + data_file_name_train)
  data_validate_df = pd.read_csv(data_file_location + data_file_name_validate)
  data_test_df = pd.read_csv(data_file_location + data_file_name_test)
  
  # Check loaded datasets shape
  print(f"Training Dataset Shape: {data_train_df.shape}")
  print(f"Validation Dataset Shape: {data_validate_df.shape}")
  print(f"Testing Dataset Shape: {data_test_df.shape}")
  
  # Display a summary of each dataset
  print("Training Dataset:")
  print(data_train_df.head())
  print("Validation Dataset:")
  print(data_validate_df.head())
  print("Testing Dataset:")
  print(data_test_df.head())
  
  # Convert open_time column to a valid Datetime format
  data_train_df["open_time"] = pd.to_datetime(data_train_df["open_time"])
  data_validate_df["open_time"] = pd.to_datetime(data_validate_df["open_time"]) 
  data_test_df["open_time"] = pd.to_datetime(data_test_df["open_time"])
  
  # Extract dates from each dataset
  data_train_dates = data_train_df["open_time"]
  data_validate_dates = data_validate_df["open_time"]
  data_test_dates = data_test_df["open_time"]
  
  # Extract features
  features = ['open', 'high', 'low', 'close', 'volume', 
             'quote_asset_volume', 'number_of_trades',
             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
  data_train_scaled = data_train_df[features].values
  data_validate_scaled = data_validate_df[features].values
  data_test_scaled = data_test_df[features].values
  
  # Define the sequence size
  sequence_size = 60

  # Construct training data
  X_train, y_train = construct_lstm_data(data_train_scaled, sequence_size, 0)
  
  # Combine scaled datasets all together
  data_all_scaled = np.concatenate([data_train_scaled, data_validate_scaled, data_test_scaled], axis=0)

  # Calculate data size
  train_size = len(data_train_scaled)
  validate_size = len(data_validate_scaled)
  test_size = len(data_test_scaled)

  # Construct validation dataset
  X_validate, y_validate = construct_lstm_data(data_all_scaled[train_size-sequence_size:train_size+validate_size,:], sequence_size, 0)

  # Construct testing dataset
  X_test, y_test = construct_lstm_data(data_all_scaled[-(test_size+sequence_size):,:], sequence_size, 0)
  
  # Check original data and data splits shapes
  print(f"Full Scaled Data: {data_all_scaled.shape}")
  print(f"\n Data Train Scaled: {data_train_scaled.shape}")
  print(f"> Data Train X: {X_train.shape}")
  print(f"> Data Train y: {y_train.shape}")

  print(f"\n Data Validate Scaled: {data_validate_scaled.shape}")
  print(f"> Data Validate X: {X_validate.shape}")
  print(f"> Data Validate y: {y_validate.shape}")

  print(f"\n Data Test Scaled: {data_test_scaled.shape}")
  print(f"> Data Test X: {X_test.shape}")
  print(f"> Data Test y: {y_test.shape}")