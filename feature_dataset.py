import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os


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


class ImageNetDataset(Dataset):
    """Dataset for loading preprocessed ImageNet task data"""

    def __init__(self, task_dir: str, task_id: str, is_train: bool = True):
        # Determine file path
        mode = 'train' if is_train else 'val'
        file_path = os.path.join(task_dir, f'task_{task_id}_{mode}.npz')
        # Load data
        loaded = np.load(file_path)
        self.data = loaded['data']
        self.labels = loaded['labels']

        # Reshape if needed and convert to float32
        if len(self.data.shape) == 2:
            self.data = self.data.reshape(-1, 3, 64, 64)
        self.data = self.data.astype(np.float32) / 255.0

        # Load task definitions to get label mapping
        with open(os.path.join(task_dir, 'task_definitions.json'), 'r') as f:
            self.task_definitions = json.load(f)

        # Create label mapping
        self.label_map = {
            old_label: new_label
            for new_label, old_label in enumerate(self.task_definitions[f'task_{task_id}'])
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.label_map[int(self.labels[idx])], dtype=torch.long)
        return img, label

