import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network model using nn.Sequential
class FeatureClassifier(nn.Module):
    def __init__(self, input_size=1024, hidden1=512, hidden2=256, num_classes=100, continual_backprop=False):
        super(FeatureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

        if continual_backprop:
            self.model = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                CBPLinear(in_layer=self.fc1, out_layer=self.fc2),
                self.fc2,
                nn.ReLU(),
                CBPLinear(in_layer=self.fc2, out_layer=self.fc3),
                self.fc3
                )
        else:
           self.model = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                self.fc2,
                nn.ReLU(),
                self.fc3
                ) 
        
        

    def forward(self, x):
        return self.model(x)


