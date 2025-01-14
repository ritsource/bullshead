import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.NNClassifier import NNClassifier
from interfaces.algo import Result
from algorithms.types import Output
from backtest.backtest import Backtester
from constants import colors
from plotter.plotter import plot_trades
from enum import Enum

class BasicAlgorithm():
    def __init__(self, model_path="./models/basic_torch.pth", epochs=50):
        self._model_path = model_path
        self._model = NNClassifier(BasicAlgorithm.features(), BasicAlgorithm.labels(), epochs)
        
    @staticmethod
    def get_name():
        return "Basic"
    
    @staticmethod
    def raw_data_schema():
        return ['open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'number_of_trades']
    
    @staticmethod
    def processed_data_schema():
        processed_schema = ['open_time', 'close_time', 'open', 'close', 'high', 'low']
        
        for i in BasicAlgorithm.moving_average_points():
            processed_schema.append(f'MA{i}')
            
        for i in BasicAlgorithm.price_history_range():
            for col in BasicAlgorithm.raw_data_schema():
                processed_schema.append(f'{col}_{i}')
                
        processed_schema.append('result')
                
        return processed_schema
    
    @staticmethod
    def price_history_range():
        return range(1, 11)
    
    @staticmethod
    def moving_average_points():
        return [5, 10, 20, 40, 80]
        
    @staticmethod
    def min_required_context_length():
        return max(max(BasicAlgorithm.moving_average_points()), max(BasicAlgorithm.price_history_range()))
    
    @staticmethod
    def features():
        labels = BasicAlgorithm.labels()
        features = [col for col in BasicAlgorithm.processed_data_schema() if col not in labels]
        return features
    
    @staticmethod
    def labels():
        return ['result']

    @staticmethod
    def read_csv_file(file_path = "./scraper/binance_data_merged/klines/1d/merged.csv"):
        df = pd.read_csv(file_path)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        return df

    @staticmethod
    def preprocess_data(df):
        if 'close' not in df.columns:
            raise ValueError("The DataFrame must contain a 'close' column.")
        
        for i in BasicAlgorithm.moving_average_points():
            df[f'MA{i}'] = df['close'].rolling(window=i).mean()
            
        for i in BasicAlgorithm.price_history_range():
            for col in BasicAlgorithm.raw_data_schema():
                df[f'{col}_{i}'] = df[col].shift(i)

        
        # # Convert datetime columns to Unix timestamps
        # df['open_time'] = pd.to_numeric(df['open_time'].astype(np.int64)) // 10**9
        # df['close_time'] = pd.to_numeric(df['close_time'].astype(np.int64)) // 10**9
        
        # # Convert numeric columns to float32
        # numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'number_of_trades']
        # for col in numeric_columns:
        #     df[col] = df[col].astype(np.float32)
            
        # # Calculate moving averages
        # for i in BasicAlgorithm.moving_average_points():
        #     df[f'MA{i}'] = df['close'].rolling(window=i).mean().astype(np.float32)
            
        # # Calculate historical features
        # for i in BasicAlgorithm.price_history_range():
        #     for col in BasicAlgorithm.raw_data_schema():
        #         df[f'{col}_{i}'] = df[col].shift(i).astype(np.float32)
          
        # Initialize result column with Neutral value
        df['result'] = Result.Neutral.value
        
        # Set result values based on conditions
        df.loc[(df['close'] > df['open']) & ((df['close'] - df['open']) >= (df['high'] - df['close'])), 'result'] = Result.TargetPositive.value
        df.loc[(df['close'] > df['open']) & ((df['close'] - df['open']) < (df['high'] - df['close'])), 'result'] = Result.Positive.value
        df.loc[df['close'] < df['open'], 'result'] = Result.Negative.value
        
        # 4. Everything else remains Neutral (initialized earlier)
        
        df.dropna(inplace=True)

        print("******")
        print(df.head())
        
        # Print count of TargetPositive results
        target_positive_count = len(df[df['result'] == Result.TargetPositive.value])
        
        # Create a new DataFrame with only needed columns
        columns_to_keep = BasicAlgorithm.processed_data_schema()
        df = df[columns_to_keep].copy()
            
        # Drop first N rows based on moving average window
        ma_range = BasicAlgorithm.moving_average_points()
        start_idx = ma_range[-1]
        df = df.iloc[start_idx:]
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        # Calculate split point at 80% of data
        split_idx = int(len(df) * 0.8)
        
        # Split data into training and test sets
        training_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        return training_df, test_df

    def train(self, training_df, model_path=None):
        self._model.train(training_df, model_path or self._model_path)

    def predict(self, test_sample):
        pred = self._model.predict(test_sample)
        
        if pred['result'] == Result.TargetPositive:
            return Output.BUY
        elif pred['result'] == Result.Positive:
            return Output.HOLD
        elif pred['result'] == Result.Negative:
            return Output.SELL
        else:
            return Output.SELL
    
    def load_weights(self, model_path=None):
        self._model.load_weights(model_path or self._model_path)

    def simulate(self, df, length=100, log_trades=True):
        s = Backtester(self)
        results = s.rand(df, length).result()
        
        return results
    
    def plot_candles(self, df):
        plt.figure(figsize=(10, 6))
        
        colors_map = []
        for result in df['result']:
            if result == Result.TargetPositive.value:
                colors_map.append(colors.GREEN)
            elif result == Result.Positive.value:
                colors_map.append(colors.YELLOW)
            elif result == Result.Negative.value:
                colors_map.append(colors.RED) 
            else:
                colors_map.append('grey')
                
        for i in range(len(df)):
            plt.vlines(x=i, ymin=min(df['open'].iloc[i], df['close'].iloc[i]),
                      ymax=max(df['open'].iloc[i], df['close'].iloc[i]),
                      color=colors_map[i], linewidth=4)
            plt.vlines(x=i, ymin=df['low'].iloc[i], ymax=df['high'].iloc[i],
                      color=colors_map[i], linewidth=1)
            
        plt.title('Candlestick Chart with Result Colors')
        plt.xlabel('Time')
        plt.ylabel('Price')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors.GREEN, label='Target Positive'),
            Patch(facecolor=colors.YELLOW, label='Positive'), 
            Patch(facecolor=colors.RED, label='Negative'),
            Patch(facecolor='grey', label='Neutral')
        ]
        plt.legend(handles=legend_elements)
        plt.show()

    def plot_predictions(self, test_df, length=30):    
        start_idx = np.random.randint(20, len(test_df) - length)
        end_idx = start_idx + length
        
        display_df = test_df.iloc[start_idx:end_idx].copy()
        
        features_df = test_df[self.features()].copy()
        for col in features_df.select_dtypes(include=['datetime64']).columns:
            features_df[col] = pd.to_numeric(features_df[col].astype(np.int64)) // 10**9
            
        predictions = []
        for idx in range(start_idx, end_idx):
            test_datum = features_df.iloc[idx].astype(float).values
            test_sample = torch.FloatTensor(test_datum)
            pred = self.predict(test_sample)
            print(f"{idx+1}. {pred['result']} o,h,l,c:({int(test_df.iloc[idx]['open'])} {int(test_df.iloc[idx]['high'])}, {int(test_df.iloc[idx]['low'])}, {int(test_df.iloc[idx]['close'])})")
            predictions.append(pred)
            
        display_df['predicted_probabilities'] = [p['probabilities'] for p in predictions]
        display_df['predicted_result'] = [p['result'] for p in predictions]
        
        display_df['actual_result'] = Result.Neutral
        
        display_df.loc[(display_df['close'] > display_df['open']) & 
                      ((display_df['close'] - display_df['open']) >= (display_df['high'] - display_df['close'])),
                      'actual_result'] = Result.TargetPositive
        
        display_df.loc[(display_df['close'] > display_df['open']) & 
                      ((display_df['close'] - display_df['open']) < (display_df['high'] - display_df['close'])),
                      'actual_result'] = Result.Positive
                      
        display_df.loc[display_df['close'] < display_df['open'],
                      'actual_result'] = Result.Negative
        
        correct_predictions = (display_df['predicted_result'] == display_df['actual_result'])
        incorrect_predictions = ~correct_predictions
        
        correct_target_strict = (display_df['predicted_result'] == Result.TargetPositive) & (display_df['actual_result'] == Result.TargetPositive)
        incorrect_target_strict = (display_df['predicted_result'] == Result.TargetPositive) & (display_df['actual_result'] != Result.TargetPositive)
        
        correct_target_dynamic = (display_df['predicted_result'] == Result.TargetPositive) & (
            (display_df['actual_result'] == Result.TargetPositive) | 
            (display_df['actual_result'] == Result.Positive)
        )
        
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
        test_df['month'] = pd.to_datetime(test_df['open_time']).dt.strftime('%Y-%m')
        monthly_dist = test_df.groupby(['month', 'predicted_direction']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(12, 6))
        
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
        
        for i in x:
            plt.text(i - width, monthly_dist['up'].iloc[i], str(monthly_dist['up'].iloc[i]), 
                    ha='center', va='bottom')
            plt.text(i, monthly_dist['neutral'].iloc[i], str(monthly_dist['neutral'].iloc[i]),
                    ha='center', va='bottom')
            plt.text(i + width, monthly_dist['down'].iloc[i], str(monthly_dist['down'].iloc[i]),
                    ha='center', va='bottom')
            
        plt.tight_layout()
        plt.show()