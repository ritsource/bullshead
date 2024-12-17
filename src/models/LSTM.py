import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

class LSTMModel:
    def __init__(self, input_shape, weights_path=None, **kwargs):
        """
        Initialize LSTM model for predicting binary outcomes from market data.
        Args:
            input_shape: Shape of input data (timesteps, features)
            weights_path: Optional path to load pre-trained weights
        """
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary classification output
        ])
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        if weights_path:
            self.load_weights(weights_path)
    
    def save_weights(self, filepath):
        """Save model weights to file"""
        self.model.save_weights(filepath)
        print("Model weights saved to {}".format(filepath))
    
    def load_weights(self, filepath):
        """Load model weights from file"""
        try:
            if not os.path.exists(filepath):
                print("Warning: Weights file not found at {}".format(filepath))
                return
            self.model.load_weights(filepath)
            print("Model weights loaded from {}".format(filepath))
        except Exception as e:
            print("Error loading weights: {}".format(str(e)))
            raise

    def train(self, X, y, epochs=50, batch_size=32, validation_split=0.2, save_path=None):
        """
        Train model on market data.
        Args:
            X: Market data features shaped (samples, timesteps, features)
            y: Binary labels (0/1) for price movement direction
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            save_path: Optional path to save weights after training
        """
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.save_weights(save_path)
            
        return history

    def predict(self, X):
        """
        Make predictions on market data.
        Args:
            X: Market data features shaped (samples, timesteps, features)
        Returns:
            Dict with predictions and confidence scores
        """
        predictions = self.model.predict(X)
        return {
            'predictions': predictions.flatten()
        }
