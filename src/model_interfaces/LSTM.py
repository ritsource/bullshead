import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint

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

class LSTM:
    def __init__(
        self,
        data_dir="data/processed/", 
        models_dir="models/lstm/", 
        data_source_name="btc_usdt_30m", 
        model_name="btc_usdt_30m_lstm"
    ):
        self.model = None
        
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.data_source_name = data_source_name
        self.model_name = model_name
        
        self.features = ['open', 'high', 'low', 'close', 'volume', 
               'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        self.target = "close"

    def construct_data(self, sequence_size, target_attr_idx):
        data_file_name_train = self.data_source_name + "_train_scaled.csv"
        data_file_name_validate = self.data_source_name + "_validate_scaled.csv"
        data_file_name_test = self.data_source_name + "_test_scaled.csv"
        
        # Load data files
        data_train_df = pd.read_csv(self.data_dir + data_file_name_train)
        data_validate_df = pd.read_csv(self.data_dir + data_file_name_validate)
        data_test_df = pd.read_csv(self.data_dir + data_file_name_test)
  
        # Extract dates from each dataset
        data_train_dates = data_train_df["open_time"]
        data_validate_dates = data_validate_df["open_time"]
        data_test_dates = data_test_df["open_time"]
        
        # Convert open_time column to a valid Datetime format
        data_train_df["open_time"] = pd.to_datetime(data_train_df["open_time"])
        data_validate_df["open_time"] = pd.to_datetime(data_validate_df["open_time"]) 
        data_test_df["open_time"] = pd.to_datetime(data_test_df["open_time"])
        
        # Extract features
        data_train_scaled = data_train_df[self.features].values
        data_validate_scaled = data_validate_df[self.features].values
        data_test_scaled = data_test_df[self.features].values
        
        # Combine scaled datasets all together
        data_all_scaled = np.concatenate([data_train_scaled, data_validate_scaled, data_test_scaled], axis=0)
        
        # Calculate data size
        train_size = len(data_train_scaled)
        validate_size = len(data_validate_scaled)
        test_size = len(data_test_scaled)
        
        # Construct training data
        X_train, y_train = construct_lstm_data(data_train_scaled, sequence_size, 0)
        
        # Construct validation dataset
        X_validate, y_validate = construct_lstm_data(data_all_scaled[train_size-sequence_size:train_size+validate_size,:], sequence_size, 0)

        # Construct testing dataset
        X_test, y_test = construct_lstm_data(data_all_scaled[-(test_size+sequence_size):,:], sequence_size, 0)
        
        return X_train, y_train, X_validate, y_validate, X_test, y_test, data_all_scaled

    def train(self, sequence_size, target_attr_idx):
        # Construct data
        X_train, y_train, X_validate, y_validate, X_test, y_test, data_all_scaled = self.construct_data(sequence_size, target_attr_idx)

        # Initializing the model
        regressor = Sequential()
        
        # Add input layer
        regressor.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        
        # Add first LSTM layer and dropout regularization layer
        regressor.add(KerasLSTM(units = 100, return_sequences = True))
        regressor.add(Dropout(rate = 0.2))
        
        # Add second LSTM layer and dropout regularization layer
        regressor.add(KerasLSTM(units = 100, return_sequences = True))
        regressor.add(Dropout(rate = 0.2))
        
        # Add third LSTM layer and dropout regularization layer
        regressor.add(KerasLSTM(units = 100, return_sequences = True))
        regressor.add(Dropout(rate = 0.2))
        
        # Add forth LSTM layer and dropout regularization layer
        regressor.add(KerasLSTM(units = 100))
        regressor.add(Dropout(rate = 0.2))
        
        # Add last dense layer/output layer
        regressor.add(Dense(units = 1))
        
        # Compiling the model
        regressor.compile(optimizer = "adam", loss="mean_squared_error")
        
        # Create a checkpoint to monitor the validation loss and save the model with the best performance.
        model_location = self.models_dir
        model_name = self.model_name + ".model.keras"
        best_model_checkpoint_callback = ModelCheckpoint(
            os.path.join(model_location, model_name),
            monitor="val_loss", 
            save_best_only=True, 
            mode="min", 
            verbose=0)

        # Training the model
        history = regressor.fit(
            x = X_train,
            y = y_train, 
            validation_data=(X_validate, y_validate), 
            epochs=100,
            batch_size = 64, 
            callbacks = [best_model_checkpoint_callback])
        
        return history

    def predict(self, sequence_size, target_attr_idx):
        # Prepare model location and name
        model_location = self.models_dir
        model_name = self.model_name + ".model.keras"
        
        btc_usdt_30m_lstm.model.keras

        # Load the best performing model
        best_model = load_model(os.path.join(model_location, model_name))
        
        # Load training data from saved files
        data_file_name_train = self.data_source_name + "_train_scaled.csv"
        data_train_df = pd.read_csv(os.path.join(self.data_dir, data_file_name_train))
        
        # Extract features
        features = ['open', 'high', 'low', 'close', 'volume', 
                   'quote_asset_volume', 'number_of_trades',
                   'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        data_train_scaled = data_train_df[features].values
        
        # Define sequence size and construct training data
        X_train, y_train = construct_lstm_data(data_train_scaled, sequence_size, target_attr_idx)
        X_validate, y_validate = construct_lstm_data(data_train_scaled, sequence_size, target_attr_idx)
        X_test, y_test = construct_lstm_data(data_train_scaled, sequence_size, target_attr_idx)
        
        # Predict stock price for all data splits
        y_train_predict = best_model.predict(X_train)
        y_validate_predict = best_model.predict(X_validate)
        y_test_predict = best_model.predict(X_test)
        
        # Prepare scaler model name and location
        scaler_model_location = self.models_dir
        scaler_model_name = self.model_name + "_scaler"
        scaler_model_ext = "gz"

        # Store the scaler model
        sc = joblib.load(os.path.join(scaler_model_location, scaler_model_name + "." + scaler_model_ext))
        
        # Restore actual distribution for predicted prices
        y_train_inv = sc.inverse_transform(np.concatenate((y_train.reshape(-1,1), np.ones((len(y_train.reshape(-1,1)), 5))), axis=1))[:,0]
        y_validate_inv = sc.inverse_transform(np.concatenate((y_validate.reshape(-1,1), np.ones((len(y_validate.reshape(-1,1)), 5))), axis=1))[:,0]
        y_test_inv = sc.inverse_transform(np.concatenate((y_test.reshape(-1,1), np.ones((len(y_test.reshape(-1,1)), 5))), axis=1))[:,0]

        y_train_predict_inv = sc.inverse_transform(np.concatenate((y_train_predict, np.ones((len(y_train_predict), 5))), axis=1))[:,0]
        y_validate_predict_inv = sc.inverse_transform(np.concatenate((y_validate_predict, np.ones((len(y_validate_predict), 5))), axis=1))[:,0]
        y_test_predict_inv = sc.inverse_transform(np.concatenate((y_test_predict, np.ones((len(y_test_predict), 5))), axis=1))[:,0]
        
        # Define chart colors
        train_actual_color = "cornflowerblue"
        validate_actual_color = "orange"
        test_actual_color = "green"
        train_predicted_color = "lightblue"
        validate_predicted_color = "peru"
        test_predicted_color = "limegreen"
        
        recent_samples = 50
        plt.figure(figsize=(18,6))
        plt.plot(data_train_dates[-recent_samples:,], y_train_inv[-recent_samples:,], label="Training Data", color=train_actual_color, linewidth=4)
        plt.plot(data_train_dates[-recent_samples:,], y_train_predict_inv[-recent_samples:,], label="Training Predictions", linewidth=2, color=train_predicted_color)

        plt.plot(data_validate_dates, y_validate_inv, label="Validation Data", color=validate_actual_color, linewidth=4)
        plt.plot(data_validate_dates, y_validate_predict_inv, label="Validation Predictions", linewidth=2, color=validate_predicted_color)

        plt.plot(data_test_dates, y_test_inv, label="Testing Data", color=test_actual_color, linewidth=4)
        plt.plot(data_test_dates, y_test_predict_inv, label="Testing Predictions", linewidth=2, color=test_predicted_color)

        plt.title("Google Stock Price Predictions With LSTM (last 50 financial days)")
        plt.xlabel("Time")
        plt.ylabel("Stock Price (USD)")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        # plt.xticks(rotation=45)
        plt.legend()
        plt.grid(color="lightgray")