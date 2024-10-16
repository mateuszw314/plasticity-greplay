import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class CustomNumpyDataset(Dataset):
    def __init__(self, data_path, labels_path, one_hot_labels = True):
        # Load the numpy arrays
        self.data = np.load(data_path)
        
        self.labels = np.load(labels_path)
        
        self.data = np.reshape(self.data, (len(self.labels), 2048))

        # Check if the data and labels have consistent length
        assert len(self.data) == len(self.labels), f"Mismatch between data and labels length: {len(self.data)} vs {len(self.labels)}"
        
        # Convert numpy arrays to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        if one_hot_labels:
            self.labels = torch.nn.functional.one_hot(self.labels, 50).type(torch.float32)

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve the sample and its corresponding label based on the index
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label