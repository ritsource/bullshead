import numpy as np
from models.HMM import Sentiment

class Random:
    def __init__(self):
        self.sentiments = [Sentiment.Buy, Sentiment.Hold, Sentiment.Sell]

    def predict(self, data=None):
        return {
            'prediction': np.random.choice(self.sentiments),
            'confidence': 1/3
        }

    def train(self, *args, **kwargs):
        pass
