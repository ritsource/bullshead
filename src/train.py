import numpy as np
from algorithms.prediction import PredictionAlgorithm

def train(model_name, weights_path):
    """
    Train a model with the given name.
    Args:
        model_name: Name of the model to train ("HMM", "LSTM", or "Random")
        weights_path: Path to the weights file
    """
    # Example data (features and labels for training)
    X = np.random.rand(100, 1)  # Features
    y = np.random.randint(0, 2, 100)  # Binary labels (positive=1, negative=0)
    
    # Initialize model controller
    controller = PredictionAlgorithm(model_name, input_shape=(1, 1), weights_path=weights_path)

    # Train model
    controller.train(X, y, epochs=10, batch_size=16, validation_split=0.2, save_path=weights_path)
    print("Model " + model_name + " trained.")


if __name__ == "__main__":
    train("LSTM", "models/weights/lstm.weights.h5")
