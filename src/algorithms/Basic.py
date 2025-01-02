import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np

# Get cpu, gpu or mps device for training.
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
device = "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        # Convert features to float, handling datetime columns
        features_float = features.copy()
        for col_idx in range(features.shape[1]):
            if isinstance(features[0, col_idx], pd.Timestamp):
                # Convert timestamps to Unix timestamp (float)
                features_float[:, col_idx] = [x.timestamp() for x in features[:, col_idx]]
        
        self.features = torch.FloatTensor(features_float.astype(float))
        self.labels = torch.FloatTensor(labels.astype(float))
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, current_price):
        # MSE component
        mse_loss = self.mse(pred, target)
        
        # Directional component
        pred_direction = (pred - current_price) > 0
        true_direction = (target - current_price) > 0
        directional_loss = (pred_direction != true_direction).float().mean()
        
        # Combine both losses
        return mse_loss + self.alpha * directional_loss

class Result(Enum):
    Buy = "buy"
    Sell = "sell"

class BasicAlgorithm():
    def __init__(self, export_dir="./models/basic_torch.pth"):
        super().__init__()
        
        self._export_dir = export_dir
        
        self._batch_size = len(BasicAlgorithm.features())
        self._epochs = 5

        self._loss_fn = None
        self._optimizer = None
        
    @staticmethod
    def get_name():
        return "Basic"
    
    @staticmethod
    def raw_data_schema():
        return ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades']
    
    @staticmethod
    def processed_data_schema():
        processed_schema = ['open_time', 'close_time', 'close']
        
        for i in BasicAlgorithm.get_calc_ma_range():
            processed_schema.append(f'MA{i}')
            
        for i in BasicAlgorithm.get_calc_ma_range():
            for col in BasicAlgorithm.raw_data_schema():
                processed_schema.append(f'{col}_{i}')
                
        return processed_schema
    
    @staticmethod
    def get_calc_ma_range():
        return range(1,11);
      
    @staticmethod
    def features():
        label = BasicAlgorithm.label()
        features = [col for col in BasicAlgorithm.processed_data_schema() if col != label]
        
        return features
    
    @staticmethod
    def label():
        return 'close'
    
    @staticmethod
    def read_csv(file_path = "./scraper/binance_data_merged/klines/1d/merged.csv"):
        df = pd.read_csv(file_path)

        df['open_time'] = pd.to_datetime(df['open_time'])
        df['close_time'] = pd.to_datetime(df['close_time'])

        return df
    
    @staticmethod
    def columns_to_drop():
        schema_columns = set(BasicAlgorithm.raw_data_schema())  # Get schema columns
        return (schema_columns | {'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'}) - {'open_time', 'close_time', 'close'}
    
    @staticmethod
    def preprocess_data(df):
        # Ensure 'close' column exists
        if 'close' not in df.columns:
            raise ValueError("The DataFrame must contain a 'close' column.")
        
        # Add current time columns
        df['open_time_0'] = df['open_time']
        df['close_time_0'] = df['close_time']
        
        for i in BasicAlgorithm.get_calc_ma_range():
            df[f'MA{i}'] = df['close'].rolling(window=i).mean()
            
        for i in BasicAlgorithm.get_calc_ma_range():
            for col in BasicAlgorithm.raw_data_schema():
                df[f'{col}_{i}'] = df[col].shift(i)
        
        columns_to_keep = BasicAlgorithm.processed_data_schema()
        df = df[columns_to_keep]
            
        df.dropna(inplace=True)
            
        # Convert timestamp columns to float before splitting
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].apply(lambda x: x.timestamp())
        
        # Calculate split point at 70% of data
        split_idx = int(len(df) * 0.7)
        
        # Split data into training (first 70%) and test (remaining 30%)
        training_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print("--- training_df.shape ---\n", training_df.shape)
        print("--- test_df.shape ---\n", test_df.shape)
        
        return training_df, test_df
    
    def plot_data(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(df['open_time'], df['close'], label='Close Price')
        plt.title('Close Price Over Time')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()
        plt.show()
    
    def get_model(self):
        model = NeuralNetwork(self._batch_size).to(device)
        self._loss_fn = DirectionalLoss(alpha=0.5)
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-5  # L2 regularization to prevent overfitting
        )
        return model
      
    def train(self, model, training_df):
        training_data = CustomDataset(
            training_df[self.features()].values,  # Features
            training_df[self.label()].values  # Labels
        )

        train_dataloader = DataLoader(training_data, batch_size=self._batch_size)

        for t in range(self._epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            size = len(train_dataloader.dataset)
            model.train()
            for batch, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)

                # Compute prediction error
                pred = model(X)
                loss = self._loss_fn(pred, y, X[:, -1])

                # Backpropagation
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()

                if batch % 100 == 0:
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                    
        torch.save(model.state_dict(), self._export_dir)
        print(f"Saved PyTorch Model State to {self._export_dir}")

    def load_weights(self, model, export_dir):
        model.load_state_dict(torch.load(export_dir if export_dir else self._export_dir, weights_only=True))

    def predict(self, model, test_sample):
        # Ensure model is in evaluation mode
        model.eval()
        
        # Add batch dimension if needed
        if len(test_sample.shape) == 1:
            test_sample = test_sample.unsqueeze(0)
            
        # Move to device and make prediction
        test_sample = test_sample.to(device)
        with torch.no_grad():
            prediction = model(test_sample)
            
        # Get current price from input features (last column)
        current_price = test_sample[0, -1].item()
        
        # Get predicted price
        predicted_price = prediction.item()
        
        # Calculate confidence based on difference from current price
        price_diff = abs(predicted_price - current_price)
        confidence = min(100.0, (price_diff / current_price) * 100)
        
        # Determine buy/sell signal
        signal = Result.Buy if predicted_price > current_price else Result.Sell
        
        return {
            "prediction": signal,
            "confidence": confidence,
            "current_price": current_price,
            "predicted_price": predicted_price
        }