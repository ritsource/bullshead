import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def preprocess_data(data_file_path, data_source_name="btc_usdt_30m", exported_data_path="data/processed/", exported_model_path="models/lstm/"):
    data = pd.read_csv(data_file_path)

    # Convert timestamp to datetime
    data["open_time"] = pd.to_datetime(data["open_time"], unit='ms')
    data["close_time"] = pd.to_datetime(data["close_time"], unit='ms')

    # Define features to scale
    features = ['open', 'high', 'low', 'close', 'volume', 
               'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    target = "close"

    # Define date ranges for train/validate/test splits
    test_start_date = pd.to_datetime("2024-01-01")
    test_end_date = pd.to_datetime("2024-02-29")
    validate_start_date = pd.to_datetime("2024-03-01") 
    validate_end_date = pd.to_datetime("2024-04-30")
    train_start_date = pd.to_datetime("2024-05-01")
    train_end_date = pd.to_datetime("2024-11-30")

    # Split data into train/validate/test sets
    data_train = data[data["open_time"] <= train_end_date][features]
    data_train_dates = data[data["open_time"] <= train_end_date]["open_time"]
    
    data_validate = data[(data["open_time"] >= validate_start_date) & 
                        (data["open_time"] <= validate_end_date)][features]
    data_validate_dates = data[(data["open_time"] >= validate_start_date) & 
                              (data["open_time"] <= validate_end_date)]["open_time"]
    
    data_test = data[(data["open_time"] >= test_start_date) & 
                     (data["open_time"] <= test_end_date)][features]
    data_test_dates = data[(data["open_time"] >= test_start_date) & 
                          (data["open_time"] <= test_end_date)]["open_time"]

    # Initialize scaler with range [0,1]
    sc = MinMaxScaler(feature_range=(0, 1))

    # Fit scaler on training data and transform all sets
    data_train_scaled = sc.fit_transform(data_train)
    data_validate_scaled = sc.transform(data_validate)
    data_test_scaled = sc.transform(data_test)

    # Store scaler model
    os.makedirs(exported_model_path, exist_ok=True)
    scaler_model_ext = ".gz"
    export_model_file_name = data_source_name + "_lstm_scaler" + scaler_model_ext
    scaler_path = os.path.join(exported_model_path, export_model_file_name)
    joblib.dump(sc, scaler_path)

    # Create DataFrames with scaled features
    data_train_scaled_df = pd.DataFrame(data_train_scaled, columns=features)
    data_train_scaled_df["open_time"] = data_train_dates.values

    data_validate_scaled_df = pd.DataFrame(data_validate_scaled, columns=features)
    data_validate_scaled_df["open_time"] = data_validate_dates.values

    data_test_scaled_df = pd.DataFrame(data_test_scaled, columns=features)
    data_test_scaled_df["open_time"] = data_test_dates.values

    # Create processed data directory if it doesn't exist
    os.makedirs(exported_data_path, exist_ok=True)

    # Store processed datasets
    data_train_scaled_df.to_csv(os.path.join(exported_data_path, f"{data_source_name}_train_scaled.csv"), index=False)
    data_validate_scaled_df.to_csv(os.path.join(exported_data_path, f"{data_source_name}_validate_scaled.csv"), index=False)
    data_test_scaled_df.to_csv(os.path.join(exported_data_path, f"{data_source_name}_test_scaled.csv"), index=False)

    return data_train_scaled_df, data_validate_scaled_df, data_test_scaled_df
