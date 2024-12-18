import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

class LSTM:
    def __init__(self):
        self.model = None
        
    def train(self, X_train, y_train, X_validate, y_validate, 
              epochs=200, batch_size=32, model_path="models/lstm_model.keras"):
        """
        Train the LSTM model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like 
            Training target values
        X_validate : array-like
            Validation features
        y_validate : array-like
            Validation target values
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        model_path : str
            Path to save the best model
        """
        # Build LSTM model
        self.model = Sequential([
            KerasLSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            KerasLSTM(units=50, return_sequences=True),
            Dropout(0.2),
            KerasLSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        # Compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Create ModelCheckpoint callback
        checkpoint = ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        
        # Train model
        self.model.fit(
            X_train, 
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_validate, y_validate),
            callbacks=[checkpoint],
            verbose=1
        )
        
        # Load best model
        self.model = load_model(model_path)
        
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : array-like
            Input features
            
        Returns:
        --------
        array-like
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        return self.model.predict(X)
