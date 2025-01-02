import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class BasicModel(nn.Module):
    def __init__(self):
        self._device = "cpu"
        # self._model = NeuralNetwork().to(self._device)
        # print(self._model)
        super(BasicModel, self).__init__()
        
    def train(self, train_dataloader, loss_fn, optimizer):
        print(f"Training using {self._device} device")
        pass
    
    def test(self, test_dataloader, loss_fn):
        pass
    
    def predict(self, x):
        pass
