import numpy as np
from hmmlearn import hmm

class HMM:
    def __init__(self, n_components=2):
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100)
    
    def train(self, X):
        """
        Train the HMM model.
        Args:
            X: Array-like, shape (n_samples, n_features)
        """
        self.model.fit(X)
        print("HMM model trained successfully.")

    def predict(self, X):
        """
        Predict whether the next day will be positive or negative.
        Args:
            X: Features for the last day.
        Returns:
            {"prediction": "positive"/"negative", "confidence": float}
        """
        logprob, state_sequence = self.model.decode(X)
        confidence = np.abs(logprob / len(X))
        prediction = "positive" if state_sequence[-1] == 1 else "negative"
        return {"prediction": prediction, "confidence": confidence}
