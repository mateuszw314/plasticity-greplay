import pytest
from torch.utils.data import DataLoader
from utils import load_dataset, get_subset_of_classes
from feature_dataset import CustomNumpyDataset
import torch
from torchvision import datasets


def test_get_subset_of_classes():
    class_indices = [1, 5, 10]
    dataset = CustomNumpyDataset('data/core50_features/full.npy', 'data/core50_features/full_labels.npy', one_hot_labels=False,)

    dataset_len = len(dataset)
    train_set, test_set = torch.utils.data.random_split(dataset, [int(0.75 * dataset_len), int(0.25 * dataset_len) + 1],
                                            generator=torch.Generator().manual_seed(42))

    subset = get_subset_of_classes(train_set, class_indices)
    print(len(dataset), len(train_set), len(subset))
    labels = subset[:][1]
    print(labels)
    assert torch.all((labels == 1) | (labels == 5) | (labels == 10))
