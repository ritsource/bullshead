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
        # self.features = torch.tensor(features, dtype=torch.float32)
        # self.labels = torch.tensor(labels, dtype=torch.long).squeeze()
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(labels.flatten(), dtype=torch.long),
            num_classes=len(Result)
        ).float()

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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        probs = self.softmax(logits)
        return probs

# class CustomLoss(nn.Module):
#     def __init__(self):
#         super(CustomLoss, self).__init__()
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, inputs, targets):
#         return self.criterion(inputs, targets)

class NNClassifier(nn.Module):
    def __init__(self, features, labels, epochs=50, batch_size=None, device=device):
        super(NNClassifier, self).__init__()

        self._features = features
        self._labels = labels
        
        print(f"features: {features}")
        print(f"labels: {labels}")
        
        self._batch_size = len(features) if batch_size is None else batch_size
        self._epochs = epochs
        self._device = device

        self._model = NeuralNetwork(len(features), len(Result)).to(device)
        # self._loss_fn = CustomLoss()
        # self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
    
    def features(self):
        return self._features
    
    def labels(self):
        return self._labels
    
    def model(self):
        return self._model
      
    def train(self, training_df, model_path):
        # print(f"Training data shape: {training_df[self.features()].values.shape}")
        
        # print(f"Training data labels shape: {training_df.head()}")
        
        # # Print schema information about the training DataFrame
        # print("\nTraining DataFrame Schema:")
        # print("-" * 50)
        # print(training_df.info())
        # print("\nFeature columns:")
        # for col in self.features():
        #     print(f"{col}")
        # print("\nLabel columns:") 
        # for col in self.labels():
        #     print(f"{col}: {training_df[col].dtype}")
        # print("-" * 50)
        
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

                outputs = self._model(features)

                loss = self._criterion(outputs, labels)
                loss.backward()
                
                self._optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader)}")

        print("Training Complete")

        # # Training loop
        # for epoch in range(self._epochs):
        #     print(f"Epoch {epoch+1}\n-------------------------------")
        #     self._model.train()
            
        #     for batch, (X, y) in enumerate(train_dataloader):
        #         X, y = X.to(self._device), y.to(self._device)

        #         # Forward pass
        #         pred = self._model(X)
        #         loss = self._loss_fn(pred, y)

        #         # Backward pass
        #         self._optimizer.zero_grad()
        #         loss.backward()
        #         self._optimizer.step()

        #         if batch % 100 == 0:
        #             loss, current = loss.item(), batch * len(X)
        #             print(f"loss: {loss:>7f}  [{current:>5d}/{len(training_data):>5d}]")
        
        # self._model
        
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
            pred = self._model(sample)
            probabilities = torch.softmax(pred, dim=1)
            result_idx = torch.argmax(probabilities, dim=1).item()
            result = Result(result_idx)
            print(f"Result: {result}")

        return {
            "result": result,
            "probabilities": probabilities[0].cpu().numpy()
        }