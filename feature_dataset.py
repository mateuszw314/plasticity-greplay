import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomNumpyDataset(Dataset):
    """
        A custom Dataset class for loading data and labels from numpy arrays.
    """
    def __init__(self, data_path, labels_path, one_hot_labels=True, vector_len=2048):
        """
                Initialize the dataset by loading data and labels from numpy files.

                Args:
                    data_path (str): Path to the numpy file containing data.
                    labels_path (str): Path to the numpy file containing labels.
                    one_hot_labels (bool): If True, convert labels to one-hot encoding.
                    vector_len (int): The length of each data vector (for reshaping).
                """

        # Load the numpy arrays
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

        self.data = np.reshape(self.data, (len(self.labels), vector_len))

        # Check if the data and labels have consistent length
        assert len(self.data) == len(
            self.labels), f"Mismatch between data and labels length: {len(self.data)} vs {len(self.labels)}"

        # Convert numpy arrays to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        if one_hot_labels:
            self.labels = torch.nn.functional.one_hot(self.labels, 50).type(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
