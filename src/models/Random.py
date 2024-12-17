import random

class Random:
    def __init__(self):
        pass
    
    def train(self, *args, **kwargs):
        """
        Random model does not require training.
        """
        print("Random model does not require training.")

    def predict(self, *args, **kwargs):
        """
        Randomly predict positive or negative.
        Returns:
            {"prediction": "positive"/"negative", "confidence": float}
        """
        prediction = random.choice(["positive", "negative"])
        confidence = round(random.uniform(0.5, 1.0), 2)
        return {"prediction": prediction, "confidence": confidence}
