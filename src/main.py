import argparse
import sys
import os
import numpy as np
from serve.app import create_app
from train import train
from algorithms.prediction import PredictionAlgorithm


def main():
    parser = argparse.ArgumentParser(description='ML Model CLI')
    parser.add_argument('--serve', action='store_true', help='Start the API server')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    
    args = parser.parse_args()
    
    data_source_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "scraper",
            "binance_data_merged",
            "klines",
            "30m",
            "merged.csv"
        )
    )

    # Get the absolute path to the weights file
    lstm_weights_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "models",
            "weights",
            "lstm.weights.h5"
        )
    )
    
    # Load and prepare data from CSV
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Read the CSV file
    df = pd.read_csv(data_source_path)
    
    # Prepare features (X) and labels (y)
    # Using all columns except the last one as features
    X = df.iloc[:,:-1].values  
    y = df.iloc[:, -1].values  # Using last column as labels
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  # 5:1 ratio (train:test)
        random_state=42  # For reproducibility
    )

    if args.serve:
        app = create_app()
        app.run(host="0.0.0.0", port=8000)
    elif args.train:
        # Initialize model controller with proper input shape
        input_shape = (X_train.shape[1], 1)  # Reshape for LSTM input
        controller = PredictionAlgorithm("LSTM", input_shape=input_shape, weights_path=lstm_weights_path)

        # Reshape X data for LSTM (samples, timesteps, features)
        X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # Train model with actual data
        controller.train(
            X_train_reshaped,
            y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            save_path=lstm_weights_path
        )
        print("LSTM model trained successfully.")
    
    elif args.test:
        # Initialize controller with same input shape
        input_shape = (X_test.shape[1], 1)
        controller = PredictionAlgorithm("LSTM", input_shape=input_shape)
        controller.load_weights(lstm_weights_path)

        # Select 10 random indices from test set
        random_indices = np.random.choice(len(X_test), size=10, replace=False)
        
        correct_predictions = 0
        print("\nTesting 10 random predictions:")
        
        for idx in random_indices:
            # Reshape single sample for prediction
            X_sample = X_test[idx].reshape(1, X_test.shape[1], 1)
            prediction = controller.predict(X_sample)
            # Handle prediction as a dictionary
            predicted_value = prediction['predictions'][0]  # Access the predictions from dict
            predicted_class = 1 if predicted_value > 0.5 else 0
            
            is_correct = predicted_class == y_test[idx]
            if is_correct:
                correct_predictions += 1
                
            print("Sample " + str(idx) + ":")
            print("Predicted: " + str(predicted_class) + ", Actual: " + str(y_test[idx]) + ", Correct: " + str(is_correct))
        
        accuracy = (correct_predictions / 10) * 100
        print("\nAccuracy on random samples: " + str(accuracy) + "%")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
