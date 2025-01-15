import argparse
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from algorithms.Basic import BasicAlgorithm, Result
from plotter.plotter import plot_trades_on_candle

def main():
    parser = argparse.ArgumentParser(description='ML Model CLI')
    parser.add_argument('--basic', action='store_true', help='Train the model')
    parser.add_argument('--train', action='store_true', help='Test the model')
    parser.add_argument('--run', action='store_true', help='Preprocess the data')
    parser.add_argument('--plot', action='store_true', help='Plot the data')
    parser.add_argument('--plot_dist', action='store_true', help='Plot the distribution of predictions')
    # parser.add_argument('--dev', action='store_true', help='Input the data')
    parser.add_argument('--epochs', type=int, default=350, help='Number of epochs to train for')
    parser.add_argument('--candle', action='store_true', help='Train the model')
    parser.add_argument('--sim', action='store_true', help='Train the model')
    
    args = parser.parse_args()

    if args.basic:
        model_path = "./models/basic_nn_torch_1d.pth"
        
        # Load and preprocess data
        # dataset_file_path = "data/downloads/klines/BTCUSDT/merged/4h/BTCUSDT-4h-2017-08-01-to-2024-11-30.csv"
        dataset_file_path = "data/downloads/klines/BTCUSDT/merged/1d/BTCUSDT-1d-2017-08-01-to-2024-11-30.csv"
        df = BasicAlgorithm.read_csv_file(dataset_file_path)
        # print(df.head())
        
        algo = BasicAlgorithm(model_path, epochs=args.epochs)
        model = algo.get_model()

        training_df, test_df = BasicAlgorithm.preprocess_data(df)
        
        # # Plot distribution of results in training data
        # plt.figure(figsize=(10, 6))
        # result_counts = training_df['result'].value_counts()
        # result_names = [Result(i).name for i in result_counts.index]
        # plt.bar(result_names, result_counts.values)
        # plt.title('Distribution of Results in Training Data')
        # plt.xlabel('Result Class')
        # plt.ylabel('Count')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()
        
        # print("3. --- training_df.shape ---\n", training_df.shape)
        # print("4. --- test_df.shape ---\n", test_df.shape)
        
        # print("5. --- training_df.head() ---\n", training_df.head())
        # print("6. --- test_df.head() ---\n", test_df.head())
        
        # return

        if args.train:
            # Train the model
            algo.train(model, training_df)
        elif args.sim:
            # Run simulation
            days = 200
            ticks = days * 6  # 6 ticks per day (4 hour intervals)
            results = algo.simulate(test_df, length=days)
            
            # plot_trades_on_candle(results['display_df'], results['trades'], results['buys'], results['sells'])
        elif args.run:
            # Create a display copy of the dataframe
            pd.set_option('display.max_columns', None)
            display_df = test_df.copy()
            # Keep only specified columns
            display_df = display_df[['open_time', 'close_time', 'open', 'close']]
            print("\nFirst 5 rows:")
            print(display_df.head())
            print("\nLast 5 rows:") 
            print(display_df.tail())
            
            print(f"Picking a random date between 10 and {len(test_df)}")
            
            # Pick a random index between 10 and length of test data
            random_idx = np.random.randint(10, len(test_df))
            print(f"Random index: {random_idx}")
            
            # Convert features to numeric types before creating tensor
            features_df = test_df[algo.features()].copy()  # Create explicit copy
            # Convert any timestamp columns to numeric (Unix timestamp)
            for col in features_df.select_dtypes(include=['datetime64']).columns:
                features_df.loc[:, col] = features_df[col].astype(np.int64) // 10**9
            
            test_datum = features_df.iloc[random_idx].astype(float).values
            test_sample = torch.FloatTensor(test_datum)
            print(f"Test sample shape: {test_sample.shape}")
            print(f"Any NaN in input?: {torch.isnan(test_sample).any()}")
            
            # Get close time and original close value
            close_time = pd.to_datetime(test_df['close_time'].iloc[random_idx])
            print(f"Close time: {close_time}")
            print(f"Open (original): {test_df['open'].iloc[random_idx]:.2f}")
            print(f"Close (original): {test_df['close'].iloc[random_idx]:.2f}")
            
            # Load trained model weights
            algo.load_weights(model_path)
            
            # Make prediction
            result = algo.predict(model, test_sample)
            print(f"\nPrediction for data point {random_idx}:")
            print(f"Predicted result: {result['result']}")
            
            # Determine if prediction is positive/negative/neutral
            if result['result'] == Result.Up:
                prediction_type = "POSITIVE"
            elif result['result'] == Result.Down:
                prediction_type = "NEGATIVE" 
            else:
                prediction_type = "NEUTRAL"
                
            print(f"Prediction type: {prediction_type}")
            # print(f"Probabilities:")
            # print(f"  Up: {result['probabilities'][0]:.2%}")
            # print(f"  Down: {result['probabilities'][1]:.2%}") 
            # print(f"  Neutral: {result['probabilities'][2]:.2%}")
            
        elif args.plot:
            # Plot test data
            print("\nPlotting test data...")
            preds = algo.plot_predictions(test_df, length=60)
            print(f"Predictions: {len(preds)}")
            print(preds)
            
        elif args.candle:
            print("Plotting candles...")
            # Get random 7 day window from test_df
            start_idx = np.random.randint(0, len(test_df) - 90)
            plot_df = test_df.iloc[start_idx:start_idx + 90]
            
            print(plot_df.head())
            algo.plot_candles(plot_df)
            
        elif args.plot_dist:
            results = algo.simulate(test_df, length=90)
            print(f"Original balance: ${results['original']:.2f}")
            print(f"Final balance: ${results['final']:.2f}")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
