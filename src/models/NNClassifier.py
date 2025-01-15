import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from interfaces.algo import Result
import pandas as pd
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        features_float = features.copy()
        for col_idx in range(features.shape[1]):
            if isinstance(features[0, col_idx], pd.Timestamp):
                # Convert timestamps to Unix timestamp (float)
                features_float[:, col_idx] = [x.timestamp() for x in features[:, col_idx]]
                
        self.features = torch.FloatTensor(features_float.astype(float))
        # # self.features = torch.tensor(features, dtype=torch.float32)
        # # self.labels = torch.tensor(labels, dtype=torch.long).squeeze()
        # self.labels = torch.nn.functional.one_hot(
        #     torch.tensor(labels.flatten(), dtype=torch.long),
        #     num_classes=len(Result)
        # ).float()
        self.labels = torch.tensor(labels.flatten(), dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Get device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self, d, t):
        super().__init__()
    
        self._d = d
        self._t = t
        
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, t),
        )
        # self.softmax = nn.Softmax(dim=1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        log_probs = self.softmax(logits)
        return log_probs

class NNClassifier(nn.Module):
    def __init__(self, features, labels, epochs=50, batch_size=64, device=device):
        super(NNClassifier, self).__init__()

        self._features = features
        self._labels = labels
        
        print(f"features: {features}")
        print(f"labels: {labels}")
        
        self._batch_size = len(features) if batch_size is None else batch_size
        self._epochs = epochs
        self._device = device
        
        model = NeuralNetwork(len(features), len(Result)).to(device)
        self._criterion = nn.NLLLoss()
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-5  # L2 regularization to prevent overfitting
        )
        self._model = model
    
    def features(self):
        return self._features
    
    def labels(self):
        return self._labels
      
    def train(self, training_df, model_path):
        # Create dataset and dataloader
        training_data = CustomDataset(
            training_df[self.features()].values,
            training_df[self.labels()].values.reshape(-1, 1)
        )
        
        train_dataloader = DataLoader(
            training_data, 
            batch_size=self._batch_size,
            shuffle=True
        )
        
        epochs = self._epochs
        
        # Training loop
        for epoch in range(epochs):
            self._model.train()
            running_loss = 0.0
            
            for features, labels in train_dataloader:
                self._optimizer.zero_grad()
                
                features, labels = features.to(self._device), labels.to(self._device)

                log_probs = self._model(features)
                loss = self._criterion(log_probs, labels)
                loss.backward()
                
                self._optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader)}")

        print("Training Complete")
        
        # Save model
        torch.save(self._model.state_dict(), model_path)
        print(f"Saved PyTorch Model State to {model_path}")

    def load_weights(self, model_path):
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()
    
    def predict(self, sample):
        self._model.eval()
        
        # Convert input to tensor
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(sample.shape) == 1:
            sample = sample.unsqueeze(0)
            
        # Move to device and predict
        sample = sample.to(self._device)
        with torch.no_grad():
            log_probs = self._model(sample)
            probabilities = torch.exp(log_probs)
            result_idx = torch.argmax(probabilities, dim=1).item()
            result = Result(result_idx)
            print(f"Result: {result}")

        return {
            "result": result,
            "probabilities": probabilities[0].cpu().numpy()
        }