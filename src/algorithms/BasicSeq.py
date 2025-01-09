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

class Result(Enum):
    TargetPositive = 0
    Positive = 1 
    Negative = 2
    Neutral = 3

class ResultSequence(Enum):
    Other = 0
    Target1 = 1
    # Target2 = 2

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, d=16, N=5, X=10, SL=0.02, Q=0.5):
        super().__init__()
        
        self._d = d
        self._N = N
        self._X = X
        self._SL = SL
        self._Q = Q
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.ReLU(), 
            nn.Linear(d, d),
            nn.ReLU(),
            # nn.Linear(d, d),
            # nn.ReLU(),
            # nn.Linear(d, d),
            # nn.ReLU(),
            nn.Linear(d, len(ResultSequence))  # 3 classes for direction + 1 for movement
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        direction = self.softmax(logits[:, :3])  # First 3 outputs for Up/Down/Neutral
        movement = logits[:, 3].unsqueeze(1)  # Last output for movement
        return torch.cat((direction, movement), dim=1)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        # Convert features to float, handling datetime columns
        features_float = features.copy()
        for col_idx in range(features.shape[1]):
            if isinstance(features[0, col_idx], pd.Timestamp):
                # Convert timestamps to Unix timestamp (float)
                features_float[:, col_idx] = [x.timestamp() for x in features[:, col_idx]]
        
        self.features = torch.FloatTensor(features_float.astype(float))
        # Convert move to one-hot encoding and combine with movement
        move_one_hot = pd.get_dummies(labels[0]).values  # One-hot encoding for Up/Down/Neutral
        self.labels = torch.FloatTensor(np.column_stack((move_one_hot, labels[1])))
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target, current_price):
        # Separate predictions and targets
        pred_direction, pred_movement = pred[:, :3], pred[:, 3]
        
        # Ensure target has correct dimensions
        if target.shape[1] == 4:  # If target already has 4 columns
            target_direction, target_movement = target[:, :3], target[:, 3]
        else:  # If target has 3 columns (one-hot encoded direction only)
            target_direction = target
            target_movement = torch.zeros_like(pred_movement)  # Default to zero movement
        
        # MSE component for movement prediction
        movement_loss = self.mse(pred_movement, target_movement)
        
        # Cross entropy for direction classification
        direction_loss = self.ce(pred_direction, target_direction)
        
        # Combine losses
        return self.alpha * direction_loss + self.beta * movement_loss

class Result(Enum):
    Buy = "buy"
    Sell = "sell"
    
class Direction(Enum):
    Up = "up"
    Down = "down"
    Neutral = "neutral"
    
class BasicAlgorithm():
    def __init__(self, export_dir="./models/basic_torch.pth", epochs=500, batch_size=32):
        super().__init__()
        
        self._export_dir = export_dir
        
        self._epochs = epochs
        # self._batch_size = batch_size
        self._batch_size = len(BasicAlgorithm.features())

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
        processed_schema = ['open_time', 'close_time', 'open', 'close']
        # processed_schema = ['open', 'close']
        
        for i in BasicAlgorithm.get_calc_ma_range():
            processed_schema.append(f'MA{i}')
            
        for i in BasicAlgorithm.get_price_history_range():
            for col in BasicAlgorithm.raw_data_schema():
                processed_schema.append(f'{col}_{i}')
                
        processed_schema.append('direction')
        processed_schema.append('movement')
                
        return processed_schema
    
    @staticmethod
    def get_calc_ma_range():
        return [1, 5, 10, 20];
    
    @staticmethod
    def get_price_history_range():
        return range(1, 11)
    
    @staticmethod
    def features():
        labels = BasicAlgorithm.labels()
        features = [col for col in BasicAlgorithm.processed_data_schema() if col not in labels]
        
        return features
    
    @staticmethod
    def labels():
        return ['direction', 'movement']
    
    # @staticmethod
    # def movement():
    #     return 'movement'
    
    @staticmethod
    def read_csv(file_path = "./scraper/binance_data_merged/klines/1d/merged.csv"):
        df = pd.read_csv(file_path)

        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        return df
    
    # @staticmethod
    # def columns_to_drop():
    #     schema_columns = set(BasicAlgorithm.raw_data_schema())  # Get schema columns
    #     return (schema_columns | {'quote_asset_volume', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'}) - {'close', BasicAlgorithm.labels()}
    
    @staticmethod
    def preprocess_data(df):
        # Ensure 'close' column exists
        if 'close' not in df.columns:
            raise ValueError("The DataFrame must contain a 'close' column.")
        
        for i in BasicAlgorithm.get_calc_ma_range():
            df[f'MA{i}'] = df['close'].rolling(window=i).mean()
            
        for i in BasicAlgorithm.get_price_history_range():
            for col in BasicAlgorithm.raw_data_schema():
                df[f'{col}_{i}'] = df[col].shift(i)
                
        # Calculate price movement (up/down) and percentage move
        df['movement'] = df['close'].diff()
        
        movement_buffer = 0
        
        # Determine move direction (Up/Down/Neutral)
        df['direction'] = Direction.Neutral.value
        df.loc[df['movement'] > 0 + movement_buffer, 'direction'] = Direction.Up.value
        df.loc[df['movement'] < 0 - movement_buffer, 'direction'] = Direction.Down.value
        df.loc[(df['movement'] >= 0 - movement_buffer) & (df['movement'] <= 0 + movement_buffer), 'direction'] = Direction.Neutral.value
        
        # Create a new DataFrame instead of a view
        columns_to_keep = BasicAlgorithm.processed_data_schema()
        df = df[columns_to_keep].copy()  # Add .copy() here
            
        # Drop first 20 rows and any rows with NaN values
        ma_range = BasicAlgorithm.get_calc_ma_range()
        start_idx = ma_range[-1]  # Get last element of range
        df = df.iloc[start_idx:]
        
        # Calculate split point at 80% of data
        split_idx = int(len(df) * 0.8)
        
        # Split data into training (first 80%) and test (remaining 20%)
        training_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return training_df, test_df
    
    def plot_data(self, df):
        plt.figure(figsize=(10, 6))
        plt.plot(df['open_time'], df['close'], label='Close Price')
        # plt.plot(df['open'], df['close'], label='Close Price')
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
            [training_df[self.labels()].values, training_df['movement'].values]  # Both labels
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
            direction_probs = prediction[0, :3]  # First 3 outputs are direction probabilities
            predicted_movement = prediction[0, 3].item()  # Fourth output is movement
            
            # Get predicted direction class
            direction_idx = torch.argmax(direction_probs).item()
            directions = [Direction.Up.value, Direction.Down.value, Direction.Neutral.value]
            predicted_direction = directions[direction_idx]

        return {
            "direction": predicted_direction,  # Predicted direction class
            "direction_probs": direction_probs.cpu().numpy(),  # Array of [Up, Down, Neutral] probabilities
            "movement": predicted_movement  # Predicted movement value
        }
        
    def simulate(self, test_df, length=30):
        start_idx = np.random.randint(20, len(test_df) - length)
        end_idx = start_idx + length
        
        display_df = test_df.iloc[start_idx:end_idx].copy()
        
        features_df = test_df[self.features()].copy()
        # Updated datetime conversion
        for col in features_df.select_dtypes(include=['datetime64']).columns:
            features_df[col] = pd.to_numeric(features_df[col].astype(np.int64)) // 10**9
        
        predictions = []
        for idx in range(start_idx, end_idx):
            test_datum = features_df.iloc[idx].astype(float).values
            test_sample = torch.FloatTensor(test_datum)
            pred = self.predict(self.get_model(), test_sample)
            predictions.append(pred)
        
        return predictions
        
    def plot_predictions(self, test_df):    
        start_idx = np.random.randint(20, len(test_df) - 30)
        end_idx = start_idx + 30
        
        display_df = test_df.iloc[start_idx:end_idx].copy()
        
        features_df = test_df[self.features()].copy()
        # Updated datetime conversion
        for col in features_df.select_dtypes(include=['datetime64']).columns:
            features_df[col] = pd.to_numeric(features_df[col].astype(np.int64)) // 10**9
            
        # Get predictions for each day in the range
        predictions = []
        for idx in range(start_idx, end_idx):
            test_datum = features_df.iloc[idx].astype(float).values
            test_sample = torch.FloatTensor(test_datum)
            pred = self.predict(self.get_model(), test_sample)
            predictions.append(pred)
            
        # Add predictions to display dataframe
        display_df['predicted_up'] = [p['direction_probs'][0] for p in predictions]
        display_df['predicted_down'] = [p['direction_probs'][1] for p in predictions]
        display_df['predicted_neutral'] = [p['direction_probs'][2] for p in predictions]
        display_df['predicted_direction'] = [p['direction'] for p in predictions]
        display_df['predicted_movement'] = [p['movement'] for p in predictions]
        
        # Determine actual direction
        display_df['actual_direction'] = Direction.Neutral.value
        display_df.loc[display_df['close'] > display_df['open'], 'actual_direction'] = Direction.Up.value
        display_df.loc[display_df['close'] < display_df['open'], 'actual_direction'] = Direction.Down.value
        
        # Calculate accuracy
        correct_predictions = (display_df['predicted_direction'] == display_df['actual_direction'])
        accuracy = (correct_predictions.sum() / len(correct_predictions)) * 100
        
        print(f"\nPrediction Results:")
        print(f"Total predictions: {len(correct_predictions)}")
        print(f"Correct predictions: {correct_predictions.sum()}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Plot confusion matrix as a bar chart
        results = {
            'Correct': correct_predictions.sum(),
            'Incorrect': len(correct_predictions) - correct_predictions.sum()
        }
        
        plt.figure(figsize=(12, 5))
        
        # First subplot for direction probabilities
        plt.subplot(1, 2, 1)
        x = range(len(display_df))
        plt.plot(x, display_df['predicted_up'], 'g-', label='Up')
        plt.plot(x, display_df['predicted_down'], 'r-', label='Down')
        plt.plot(x, display_df['predicted_neutral'], 'b-', label='Neutral')
        
        plt.title('Predicted Direction Probabilities Over Time')
        plt.xlabel('Days')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        plt.xticks(x, display_df['open_time'].dt.strftime('%Y-%m-%d'), rotation=45)
        
        # Second subplot for accuracy bars
        plt.subplot(1, 2, 2)
        plt.bar('Correct', results['Correct'], color='green')
        plt.bar('Incorrect', results['Incorrect'], color='red')
        
        # Add value labels on bars
        for i, (label, value) in enumerate(results.items()):
            plt.text(i, value, str(int(value)), ha='center', va='bottom')
        
        plt.title('Prediction Results')
        plt.ylabel('Number of Predictions')
        plt.text(0.5, 0.95, f'Accuracy: {accuracy:.1f}%', 
                ha='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.show()
        
    def plot_predictions_distribution(self, test_df):
        # Calculate monthly distribution
        test_df['month'] = pd.to_datetime(test_df['open_time']).dt.strftime('%Y-%m')
        monthly_dist = test_df.groupby(['month', 'predicted_direction']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        
        # Create grouped bar chart
        x = np.arange(len(monthly_dist.index))
        width = 0.25
        
        plt.bar(x - width, monthly_dist['up'], width, label='Up', color='green')
        plt.bar(x, monthly_dist['neutral'], width, label='Neutral', color='blue') 
        plt.bar(x + width, monthly_dist['down'], width, label='Down', color='red')
        
        plt.title('Monthly Distribution of Predicted Directions')
        plt.xlabel('Month')
        plt.ylabel('Number of Predictions')
        plt.xticks(x, monthly_dist.index, rotation=45)
        plt.legend()
        
        # Add value labels on bars
        for i in x:
            plt.text(i - width, monthly_dist['up'].iloc[i], str(monthly_dist['up'].iloc[i]), 
                    ha='center', va='bottom')
            plt.text(i, monthly_dist['neutral'].iloc[i], str(monthly_dist['neutral'].iloc[i]),
                    ha='center', va='bottom')
            plt.text(i + width, monthly_dist['down'].iloc[i], str(monthly_dist['down'].iloc[i]),
                    ha='center', va='bottom')
            
        plt.tight_layout()
        plt.show()