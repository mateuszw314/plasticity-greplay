import argparse
import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import yaml
from feature_classifier import FeatureClassifier
from feature_dataset import CustomNumpyDataset
from vae_models import VAE


def load_dataset(config):
    dataset_name = config['dataset']
    if dataset_name == 'custom':
        data_path = config['custom_dataset']['data_path']
        labels_path = config['custom_dataset']['labels_path']
        dataset = CustomNumpyDataset(data_path, labels_path, one_hot_labels=False,
                                     vector_len=config['custom_dataset']['vector_len'])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_len = len(dataset)
    train_set, test_set = data.random_split(dataset, [int(0.75 * dataset_len), int(0.25 * dataset_len) + 1],
                                            generator=torch.Generator().manual_seed(42))
    return train_set, test_set


def parse_args():
    parser = argparse.ArgumentParser(description="Incremental Learning Experiment")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_subset_of_classes(dataset, class_indices):
    """
        Get a subset of the dataset containing only the specified classes.

        Args:
            dataset (data.Dataset): The dataset.
            class_indices (list): List of class indices to include in the subset.

        Returns:
            data.Subset: Subset of the dataset containing only the specified classes.
        """

    if isinstance(dataset.dataset, CustomNumpyDataset):
        targets = dataset.dataset.labels[dataset.indices]
    else:
        raise ValueError("Unsupported dataset type")

    mask = torch.isin(targets, torch.tensor(class_indices))
    subset_indices = [dataset.indices[i] for i, m in enumerate(mask) if m]
    return data.Subset(dataset.dataset, subset_indices)


def evaluate_classifier(model: nn.Module, dataloader: data.DataLoader, device: torch.device) -> float:
    """
        Evaluate the classifier on the test dataset.

        Args:
            model (nn.Module): The classifier model.
            dataloader (data.DataLoader): DataLoader for the test dataset.
            device (torch.device): The device to run the model on.

        Returns:
            float: Accuracy of the classifier on the test dataset.
    """

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def train_model(model: nn.Module, optimizer: optim.Optimizer, dataloader: data.DataLoader, device: torch.device,
                epochs: int, is_classifier: bool = False) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss() if is_classifier else None  # the generator has a built-in loss function
    for epoch in range(epochs):
        running_loss = 0.0
        for data in dataloader:
            (images, labels) = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            if is_classifier:
                outputs = model(images)
                total_loss = criterion(outputs, labels)
            else:
                z_mean, z_log_var, reconstruction = model(images)
                total_loss, reconstruction_loss, kl_loss = model.loss_function(images, labels, z_mean, z_log_var,
                                                                               reconstruction)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}')
    print('Finished Training Task')
