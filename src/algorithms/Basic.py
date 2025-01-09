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


DIRECTION_COLORS = {
    'green': '#16C47F',
    'yellow': '#FFC145', 
    'red': '#D84040'
}


class Result(Enum):
    Neutral = 0
    TargetPositive = 1
    Positive = 2
    Negative = 3

# class ResultSequence(Enum):
#     Other = 0
#     Target1 = 1
#     # Target2 = 2

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        
        # self._d = d
        # self._N = N
        # self._X = X
        # self._SL = SL
        # self._Q = Q
        
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
            nn.Linear(d, len(Result))
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        direction = self.softmax(logits[:, :3])  # First 3 outputs for Up/Down/Neutral
        movement = logits[:, 3].unsqueeze(1)  # Last output for movement
        return torch.cat((direction, movement), dim=1)
    
    # def forward(self, x):
    #     logits = self.linear_relu_stack(x)
    #     return logits

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        # Convert features to float, handling datetime columns
        features_float = features.copy()
        for col_idx in range(features.shape[1]):
            if isinstance(features[0, col_idx], pd.Timestamp):
                # Convert timestamps to Unix timestamp (float)
                features_float[:, col_idx] = [x.timestamp() for x in features[:, col_idx]]
        
        self.features = torch.FloatTensor(features_float.astype(float))
        # Convert labels to one-hot encoding
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(labels[0].flatten(), dtype=torch.long),
            num_classes=len(Result)
        ).float()
    
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

class Direction(Enum):
    Up = "up"
    Down = "down"
    Neutral = "neutral"
    
class BasicAlgorithm():
    def __init__(self, export_dir="./models/basic_torch.pth", epochs=50, batch_size=32):
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
        processed_schema = ['open_time', 'close_time', 'open', 'close', 'high', 'low']
        # processed_schema = ['open', 'close']
        
        for i in BasicAlgorithm.get_calc_ma_range():
            processed_schema.append(f'MA{i}')
            
        for i in BasicAlgorithm.get_price_history_range():
            for col in BasicAlgorithm.raw_data_schema():
                processed_schema.append(f'{col}_{i}')
                
        # processed_schema.append('direction')
        # processed_schema.append('movement')
        
        processed_schema.append('result')
                
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
        # return ['direction', 'movement']
        return ['result']
    
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
        # df['direction'] = Direction.Neutral.value
        # df.loc[df['movement'] > 0 + movement_buffer, 'direction'] = Direction.Up.value
        # df.loc[df['movement'] < 0 - movement_buffer, 'direction'] = Direction.Down.value
        # df.loc[(df['movement'] >= 0 - movement_buffer) & (df['movement'] <= 0 + movement_buffer), 'direction'] = Direction.Neutral.value
        
        # Default to Neutral
        df['result'] = Result.Neutral.value
        
        # Case 1: TargetPositive - Close > Open and Close-Open >= High-Open
        # Case 2: Positive - Close > Open but Close-Open < High-Open
        # Case 3: Negative - Close < Open
        # Case 4: Everything else remains Neutral (default value)
        df.loc[(df['close'] > df['open']) & ((df['close'] - df['open']) >= (df['high'] - df['close'])), 'result'] = Result.TargetPositive.value
        df.loc[(df['close'] > df['open']) & ((df['close'] - df['open']) < (df['high'] - df['close'])), 'result'] = Result.Positive.value
        df.loc[df['close'] < df['open'], 'result'] = Result.Negative.value
        
        
        # Print count of TargetPositive results
        target_positive_count = len(df[df['result'] == Result.TargetPositive.value])
        print(f"Number of TargetPositive results: {target_positive_count}")
        
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
    
    def plot_candles(self, df):
        plt.figure(figsize=(10, 6))
        
        # Create candlestick colors based on result
        colors = []
        for result in df['result']:
            if result == Result.TargetPositive.value:
                colors.append(DIRECTION_COLORS['green'])
            elif result == Result.Positive.value:
                colors.append(DIRECTION_COLORS['yellow'])
            elif result == Result.Negative.value:
                colors.append(DIRECTION_COLORS['red']) 
            else:  # Neutral
                colors.append('grey')
                
        # # Print column names
        # print("\nDataframe columns:")
        # for col in df.columns:
        #     print(f"- {col}")
                
        # Plot candlesticks
        for i in range(len(df)):
            # Plot the candlestick body
            plt.vlines(x=i, ymin=min(df['open'].iloc[i], df['close'].iloc[i]),
                      ymax=max(df['open'].iloc[i], df['close'].iloc[i]),
                      color=colors[i], linewidth=4)
            
            # Plot the wicks
            plt.vlines(x=i, ymin=df['low'].iloc[i], ymax=df['high'].iloc[i],
                      color=colors[i], linewidth=1)
            
        plt.title('Candlestick Chart with Result Colors')
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=DIRECTION_COLORS['green'], label='Target Positive'),
            Patch(facecolor=DIRECTION_COLORS['yellow'], label='Positive'), 
            Patch(facecolor=DIRECTION_COLORS['red'], label='Negative'),
            Patch(facecolor='grey', label='Neutral')
        ]
        plt.legend(handles=legend_elements)
        
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
            [training_df[self.labels()].values]  # Both labels
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
            
            # Get predicted result class
            result_idx = torch.argmax(prediction).item()
            result = Result(result_idx)

        return {
            "result": result,  # Predicted Result enum value
            "probabilities": prediction[0].cpu().numpy()  # Array of probabilities for each Result
        }
        
    def simulate(self, test_df, length=30):
        start_idx = np.random.randint(20, len(test_df) - length)
        end_idx = start_idx + length
        
        display_df = test_df.iloc[start_idx:end_idx+1].copy()  # +1 to get next day's prices
        
        features_df = test_df[self.features()].copy()
        # Updated datetime conversion
        for col in features_df.select_dtypes(include=['datetime64']).columns:
            features_df[col] = pd.to_numeric(features_df[col].astype(np.int64)) // 10**9
        
        balance = 1000  # Starting USDT balance
        holdings = 0    # Amount of crypto held
        num_buy = 0
        num_sell = 0 
        num_hold = 0
        entry_price = 0  # Track entry price for stop loss
        
        for idx in range(start_idx, end_idx):
            # Get prediction for current day
            test_datum = features_df.iloc[idx].astype(float).values
            test_sample = torch.FloatTensor(test_datum)
            pred = self.predict(self.get_model(), test_sample)
            
            current_price = display_df.iloc[idx-start_idx]['close']
            next_price = display_df.iloc[idx-start_idx+1]['close']
            
            # Check stop loss if holding position
            if holdings > 0:
                loss_pct = (current_price - entry_price) / entry_price
                if loss_pct <= -0.05:  # 5% stop loss
                    # Sell at stop loss
                    balance = holdings * current_price
                    holdings = 0
                    num_sell += 1
                    continue
            
            # Execute trading logic
            if pred['result'] == Result.TargetPositive and balance > 0:
                # Buy
                holdings = balance / current_price
                balance = 0
                entry_price = current_price  # Set entry price for stop loss
                num_buy += 1
            elif pred['result'] == Result.Negative and holdings > 0:
                # Sell
                balance = holdings * current_price
                holdings = 0
                num_sell += 1
            else:
                # Hold
                num_hold += 1
                
        # Calculate final balance based on last price
        final_balance = balance
        if holdings > 0:
            final_balance = holdings * display_df.iloc[-1]['close']
            
        print(f"Original: 1000 USDT")
        print(f"Number of Buy: {num_buy}")
        print(f"Number of Sell: {num_sell}")
        print(f"Number of Hold: {num_hold}")
        print(f"Current: {final_balance:.2f} USDT")
        
        return {
            'original': 1000,
            'buys': num_buy,
            'sells': num_sell,
            'holds': num_hold,
            'final': final_balance
        }
        
    def plot_predictions(self, test_df, length=30):    
        start_idx = np.random.randint(20, len(test_df) - length)
        end_idx = start_idx + length
        
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
            print(f"{idx+1}. {pred['result']} o,h,l,c:({int(test_df.iloc[idx]['open'])} {int(test_df.iloc[idx]['high'])}, {int(test_df.iloc[idx]['low'])}, {int(test_df.iloc[idx]['close'])})")
            predictions.append(pred)
            
        # Add predictions to display dataframe
        display_df['predicted_probabilities'] = [p['probabilities'] for p in predictions]
        display_df['predicted_result'] = [p['result'] for p in predictions]
        
        # Determine actual result based on price relationships
        display_df['actual_result'] = Result.Neutral  # Default case
        
        # Case 1: Strong positive - Close > Open with large upward movement
        display_df.loc[(display_df['close'] > display_df['open']) & 
                      ((display_df['close'] - display_df['open']) >= (display_df['high'] - display_df['close'])),
                      'actual_result'] = Result.TargetPositive
        
        # Case 2: Positive - Close > Open but smaller upward movement
        display_df.loc[(display_df['close'] > display_df['open']) & 
                      ((display_df['close'] - display_df['open']) < (display_df['high'] - display_df['close'])),
                      'actual_result'] = Result.Positive
                      
        # Case 3: Negative - Close < Open
        display_df.loc[display_df['close'] < display_df['open'],
                      'actual_result'] = Result.Negative
        
        # Case 4: Neutral - Everything else remains as the default Result.Neutral
        
        # Calculate accuracy metrics
        correct_predictions = (display_df['predicted_result'] == display_df['actual_result'])
        incorrect_predictions = ~correct_predictions
        
        # Strict TargetPositive metrics
        correct_target_strict = (display_df['predicted_result'] == Result.TargetPositive) & (display_df['actual_result'] == Result.TargetPositive)
        incorrect_target_strict = (display_df['predicted_result'] == Result.TargetPositive) & (display_df['actual_result'] != Result.TargetPositive)
        
        # Dynamic TargetPositive metrics
        correct_target_dynamic = (display_df['predicted_result'] == Result.TargetPositive) & (
            (display_df['actual_result'] == Result.TargetPositive) | 
            (display_df['actual_result'] == Result.Positive)
        )
        
        # Dynamic incorrect - only count as incorrect if large negative movement
        price_change = (display_df['close'] - display_df['open']) / display_df['open']
        incorrect_target_dynamic = (display_df['predicted_result'] == Result.TargetPositive) & (
            (display_df['actual_result'] == Result.Negative) & 
            (abs(price_change) > 0.05)
        )
        
        accuracy = (correct_predictions.sum() / len(correct_predictions)) * 100
        
        print(f"\nPrediction Results:")
        print(f"Total predictions: {len(correct_predictions)}")
        print(f"Correct predictions: {correct_predictions.sum()}")
        print(f"Incorrect predictions: {incorrect_predictions.sum()}")
        print(f"Correct target predictions (strict): {correct_target_strict.sum()}")
        print(f"Incorrect target predictions (strict): {incorrect_target_strict.sum()}")
        print(f"Correct target predictions (dynamic): {correct_target_dynamic.sum()}")
        print(f"Incorrect target predictions (dynamic): {incorrect_target_dynamic.sum()}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        plt.figure(figsize=(12, 5))
        
        # First subplot for result probabilities
        plt.subplot(1, 2, 1)
        x = range(len(display_df))
        probabilities = np.array([p for p in display_df['predicted_probabilities']])
        plt.plot(x, probabilities[:, 0], 'g-', label='Target Positive')
        plt.plot(x, probabilities[:, 1], 'r-', label='Positive')
        plt.plot(x, probabilities[:, 2], 'b-', label='Negative')
        
        plt.title('Predicted Result Probabilities Over Time')
        plt.xlabel('Days')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        plt.xticks(x, display_df['open_time'].dt.strftime('%Y-%m-%d'), rotation=45)
        
        # Second subplot for accuracy bars
        plt.subplot(1, 2, 2)
        results = [
            ('Correct', correct_predictions.sum()),
            ('Incorrect', incorrect_predictions.sum()),
            ('Correct Target\nStrict', correct_target_strict.sum()),
            ('Incorrect Target\nStrict', incorrect_target_strict.sum()),
            ('Correct Target\nDynamic', correct_target_dynamic.sum()),
            ('Incorrect Target\nDynamic', incorrect_target_dynamic.sum())
        ]
        
        x_pos = np.arange(len(results))
        plt.bar(x_pos, [r[1] for r in results], color=['green', 'red'] * 3)
        
        # Add value labels on bars
        for i, (label, value) in enumerate(results):
            plt.text(i, value, str(int(value)), ha='center', va='bottom')
        
        plt.title('Prediction Results')
        plt.ylabel('Number of Predictions')
        plt.xticks(x_pos, [r[0] for r in results], rotation=45, ha='right')
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