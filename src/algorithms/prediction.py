from models.HMM import HMM
from models.LSTM import LSTMModel
from models.Random import Random

class PredictionAlgorithm:
    def __init__(self, model_name, **kwargs):
        """
        Initialize the model controller.
        Args:
            model_name: "HMM", "LSTM", or "Random".
            kwargs: Additional parameters for models.
        """
        if model_name == "HMM":
            self.model = HMM(**kwargs)
        elif model_name == "LSTM":
            self.model = LSTMModel(**kwargs)
        elif model_name == "Random":
            self.model = Random()
        else:
            raise ValueError("Unknown model: " + model_name)
    
    def train(self, X, y=None, epochs=10, batch_size=16, validation_split=0.2, save_path=None):
        self.model.train(X, y, epochs, batch_size, validation_split, save_path)
    
    def load_weights(self, filepath):
        """
        Load model weights from a file.
        Args:
            filepath: Path to the weights file
        """
        if hasattr(self.model, 'load_weights'):
            self.model.load_weights(filepath)
        else:
            print("Warning: {} does not support loading weights".format(type(self.model).__name__))
    
    def predict(self, X):
        """
        Make predictions using the model.
        Args:
            X: Input features
        Returns:
            Model predictions in the format {"prediction": str, "confidence": float}
        """
        return self.model.predict(X)
