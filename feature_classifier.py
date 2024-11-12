import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from cbp_linear import CBPLinear
from cbp_conv import CBPConv


default_cbp_config = {'replacement_rate': 10e-4, 'maturity_threshold': 50, 'decay_rate': 0}

# Define the neural network model using nn.Sequential
class FeatureClassifier(nn.Module):
    def __init__(self, input_size=1024, hidden1=512, hidden2=256, num_classes=100, continual_backprop=False, cbp_config = default_cbp_config):
        super(FeatureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(0.3)

        if continual_backprop:
            self.model = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                self.dropout,
                CBPLinear(in_layer=self.fc1, out_layer=self.fc2, replacement_rate=cbp_config['replacement_rate'], maturity_threshold=cbp_config['maturity_threshold'], decay_rate=cbp_config['decay_rate']),
                self.fc2,
                nn.ReLU(),
                self.dropout,
                CBPLinear(in_layer=self.fc2, out_layer=self.fc3, replacement_rate=cbp_config['replacement_rate'], maturity_threshold=cbp_config['maturity_threshold'], decay_rate=cbp_config['decay_rate']),
                self.fc3
                )
        else:
           self.model = nn.Sequential(
                self.fc1,
                nn.ReLU(),
                self.dropout,
                self.fc2,
                nn.ReLU(),
                self.dropout,
                self.fc3
                ) 
        
        

    def forward(self, x):
        return self.model(x)


