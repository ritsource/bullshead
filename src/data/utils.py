import torch
from torch.utils.data import Dataset
import pandas as pd
from interfaces.algo import Result

class TorchCustomDataset(Dataset):
    def __init__(self, features, labels):
        # Convert features to float, handling datetime columns
        features_float = features.copy()
        for col_idx in range(features.shape[1]):
            if isinstance(features[0, col_idx], pd.Timestamp):
                # Convert timestamps to Unix timestamp (float)
                features_float[:, col_idx] = [x.timestamp() for x in features[:, col_idx]]
        
        self.features = torch.FloatTensor(features_float.astype(float))
        # Convert labels to one-hot encoding
        
        print(self.labels)
        
        self.labels = torch.nn.functional.one_hot(
            torch.tensor(labels[0].flatten(), dtype=torch.long),
            num_classes=len(Result)
        ).float()
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
