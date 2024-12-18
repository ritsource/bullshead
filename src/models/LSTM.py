import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout

class LSTMModel:
    def __init__(self, input_shape):
        """
        Initialize LSTM model for stock price prediction.
        Args:
            input_shape: Shape of input data (timesteps, features)
        """
        self.model = Sequential([
            Input(shape=input_shape),
            LSTM(50, return_sequences=True, activation='tanh'),
            Dropout(0.2),
            LSTM(50, return_sequences=True, activation='tanh'),
            Dropout(0.2), 
            LSTM(50, activation='tanh'),
            Dropout(0.2),
            Dense(1)  # Single continuous output for price prediction
        ])
        self.model.compile(
            optimizer="adam",
            loss="mean_squared_error"  # MSE loss for regression
        )

    def train(self, X_train, y_train, X_validate=None, y_validate=None, epochs=200, batch_size=32, callbacks=None):
        """
        Train model on stock price data.
        Args:
            X_train: Training features shaped (samples, timesteps, features)
            y_train: Training target values
            X_validate: Optional validation features
            y_validate: Optional validation targets  
            epochs: Number of training epochs
            batch_size: Training batch size
            callbacks: Optional list of Keras callbacks
        """
        validation_data = (X_validate, y_validate) if X_validate is not None else None
        
        return self.model.fit(
            x=X_train,
            y=y_train, 
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

    def predict(self, X):
        """
        Make predictions on stock price data.
        Args:
            X: Input features shaped (samples, timesteps, features)
        Returns:
            Array of predicted prices
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        return self.model.predict(X)
