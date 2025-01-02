import argparse
import sys
import pandas as pd
import numpy as np
import torch
from algorithms.Basic import BasicAlgorithm

def main():
    parser = argparse.ArgumentParser(description='ML Model CLI')
    parser.add_argument('--basic', action='store_true', help='Train the model')
    parser.add_argument('--train', action='store_true', help='Test the model')
    parser.add_argument('--run', action='store_true', help='Preprocess the data')
    parser.add_argument('--plot', action='store_true', help='Plot the data')
    parser.add_argument('--dev', action='store_true', help='Input the data')
    
    args = parser.parse_args()

    if args.basic:
        model_path = "./models/basic_nn_torch.pth"
        
        # Load and preprocess data
        dataset_file_path = "scraper/binance_data_merged/klines/1d/merged.csv"
        df = BasicAlgorithm.read_csv(dataset_file_path)
        # print(df.head())
        
        algo = BasicAlgorithm(model_path)
        model = algo.get_model()

        training_df, test_df = BasicAlgorithm.preprocess_data(df)
        
        print("Test DataFrame Columns:", len(test_df.columns))
        print(test_df.columns)

        if args.dev:
            # Input the data
            close_time = test_df.iloc[20]['close_time']
            algo.input_matrix(test_df, close_time)
        elif args.train:
            # Train the model
            algo.train(model, training_df)

        elif args.run:
            # Pick a random index between 10 and length of test data
            random_idx = np.random.randint(10, len(test_df))
            print(f"Random index: {random_idx}")
            
            print(test_df.head())
            
            # Convert features to numeric types before creating tensor
            test_datum = test_df[algo.features()].iloc[random_idx].astype(float).values
            test_sample = torch.FloatTensor(test_datum)
            
            print(f"Test sample shape: {test_sample.shape}")
            print(f"Test sample values: {test_sample}")
            print(f"Any NaN in input?: {torch.isnan(test_sample).any()}")
            
            # Load trained model weights
            algo.load_weights(model, model_path)
            
            # Make prediction
            result = algo.predict(model, test_sample)
            print(f"\nPrediction for data point {random_idx}:")
            print(f"Signal: {result['prediction'].value}")
            print(f"Confidence: {result['confidence']:.2f}%")
            print(f"Current Price: {result['current_price']:.2f}")
            print(f"Predicted Price: {result['predicted_price']:.2f}")
                
        if args.plot:
            algo.plot_data(training_df)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
