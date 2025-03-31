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
from feature_dataset import CustomNumpyDataset, ImageNetDataset
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
    if dataset_name == 'custom':
        train_set, test_set = data.random_split(dataset, [int(0.75 * dataset_len), int(0.25 * dataset_len) + 1],
                                            generator=torch.Generator().manual_seed(42))
        #train_set, test_set = data.random_split(dataset, [int(0.75 * dataset_len), int(0.25 * dataset_len)],
                                                #generator = torch.Generator().manual_seed(42))

    return train_set, test_set


def load_imagenet_dataset(task_dir: str, task_id: int, is_train: bool,
                         num_classes_per_task: int = 10, initial_task_classes: int = 500) -> data.Dataset:
    """
    Load the ImageNet dataset for a specific task.

    Args:
        task_dir: Directory containing the ImageNet dataset.
        task_id: ID of the task.
        is_train: Whether to load the training or validation set.
        num_classes_per_task: Number of classes per task (default: 10).
        initial_task_classes: Number of classes for the first task (default: 500).
    Returns:
        torch.utils.data.Dataset: The ImageNet dataset.
    """


    if task_id == 0:
        # Load the first 500 classes (50 files)
        num_initial_files = 0
        num_files_to_load = initial_task_classes // num_classes_per_task
    else:
        # Load 10 classes (1 file)
        num_initial_files = initial_task_classes // num_classes_per_task
        num_files_to_load = 1

    datasets = []
    for i in range(num_initial_files + (task_id * num_files_to_load),
                   num_initial_files + ((task_id + 1) * num_files_to_load)):
        datasets.append(ImageNetDataset(task_dir, i, is_train=is_train))

    # Concatenate the datasets
    dataset = data.ConcatDataset(datasets)

    return dataset


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


def count_dead_neurons(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode

    # Initialize a dictionary to track activations
    activation_count = {}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)


            # Forward pass and check activations
            x = inputs
            for layer in model.model.children():
                print(layer.__class__.__name__)
                x = layer(x)
                if isinstance(layer, torch.nn.ReLU):
                    # Register the number of times neurons are activated
                    if layer in activation_count:
                        activation_count[layer] += (x != 0).sum(dim=0).cpu().numpy()
                    else:
                        activation_count[layer] = (x != 0).sum(dim=0).cpu().numpy()
    # Determine dead neurons
    dead_neurons = {}
    for layer, activations in activation_count.items():
        # A neuron is dead if it was never activated
        dead_neurons[layer] = (activations == 0).sum()

    return dead_neurons


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


def train_gan(generator: nn.Module, discriminator: nn.Module, optimizer_G: optim.Optimizer, optimizer_D: optim.Optimizer, dataloader: data.DataLoader, device: torch.device,
                epochs: int) -> None:

    criterion = nn.BCELoss()
    num_classes = 50

    # --- Training Loop ---
    for epoch in range(epochs):
        for i, (real_vectors, labels) in enumerate(dataloader):
            real_vectors = real_vectors.to(device)
            labels = labels.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss with real vectors
            real_labels_d = torch.ones(real_vectors.size(0), 1).to(device)
            real_output = discriminator(real_vectors, labels)
            d_real_loss = criterion(real_output, real_labels_d)

            # Loss with fake vectors
            noise = torch.randn(real_vectors.size(0), generator.latent_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (real_vectors.size(0),)).to(device)
            fake_vectors = generator(noise, fake_labels)
            fake_labels_d = torch.zeros(real_vectors.size(0), 1).to(device)
            fake_output = discriminator(fake_vectors.detach(), fake_labels)
            d_fake_loss = criterion(fake_output, fake_labels_d)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake vectors and labels for generator loss calculation
            noise = torch.randn(real_vectors.size(0), generator.latent_dim).to(device)
            fake_labels_g = torch.randint(0, num_classes, (real_vectors.size(0),)).to(device)
            generated_vectors = generator(noise, fake_labels_g)

            # Loss measures generator's ability to fool the discriminator
            output = discriminator(generated_vectors, fake_labels_g)
            g_loss = criterion(output, real_labels_d)  # Note: We use real_labels_d here to trick the generator

            g_loss.backward()
            optimizer_G.step()

            # --- Print Progress ---
            if i % 50 == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

    print("GAN training finished")