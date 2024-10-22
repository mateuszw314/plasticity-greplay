#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from vae_models import VAE
from feature_classifier import FeatureClassifier
from feature_dataset import CustomNumpyDataset
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Incremental Learning Experiment")
    parser.add_argument('--data-path', type=str, default='data/core50_features/full.npy', help='Path to the dataset file')
    parser.add_argument('--labels-path', type=str, default='data/core50_features/full_labels.npy', help='Path to the labels file')
    parser.add_argument('--output-dir', type=str, default='models_211024', help='Directory to save the models')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per task')
    parser.add_argument('--num-experiments', type=int, default=1, help='Number of experiments to run')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA for training')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    for experiment in range(args.num_experiments):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_path = f'replay_cbp_{experiment}_{timestamp}'
        root_path = os.path.join(args.output_dir, experiment_path)
        os.makedirs(root_path, exist_ok=True)
        train_incremental_tasks(root_path, device, args.data_path, args.labels_path, args.batch_size, args.epochs, cbp=True)

    logger.info("Finished all experiments")

def get_subset_of_classes(dataset: data.Dataset, class_indices: list) -> data.Subset:
    """
    Returns a subset of the dataset containing only the specified classes.
    
    Parameters:
    - dataset: Dataset
    - class_indices: List[int] of class indices to include in the subset
    
    Returns:
    - Subset of the dataset containing only the specified classes
    """
    targets = dataset[:][1]
    mask = torch.isin(targets, torch.tensor(class_indices))
    subset_indices = mask.nonzero(as_tuple=False).squeeze().tolist()
    return data.Subset(dataset, subset_indices)

def generate_samples(generator: VAE, classifier: FeatureClassifier, class_to_generate: int, examples_to_generate: int, device: torch.device) -> torch.Tensor:
    """
    Generates samples from the VAE for a specific class.

    Parameters:
    - generator: VAE model
    - classifier: Classifier model
    - class_to_generate: Class label to generate
    - examples_to_generate: Number of examples to generate

    Returns:
    - Generated images: torch.Tensor
    """
    accepted_samples = []
    generator.eval()
    classifier.eval()
    with torch.no_grad():
        counter = 0
        while (len(accepted_samples) < examples_to_generate) and counter < 15:
            counter += 1
            prior_means = generator.prior_means[class_to_generate].to(device)
            prior_logvars = generator.prior_logvars[class_to_generate].to(device)
            std_devs = torch.sqrt(torch.exp(prior_logvars))
            means_extended = prior_means.unsqueeze(0).expand(examples_to_generate, -1)
            std_devs_extended = std_devs.unsqueeze(0).expand(examples_to_generate, -1)
            random_latent_vectors = torch.normal(means_extended, std_devs_extended).to(device)
            generated_images = generator.decoder(random_latent_vectors)

            outputs = classifier(generated_images)
            _, predicted_labels = torch.max(outputs, 1)

            for img, pred_label in zip(generated_images, predicted_labels):
                if pred_label.item() == class_to_generate:
                    accepted_samples.append(img)
                if len(accepted_samples) >= examples_to_generate:
                    break

    return torch.stack(accepted_samples)

def train_model(model: nn.Module, optimizer: optim.Optimizer, dataloader: data.DataLoader, device: torch.device, epochs: int, is_classifier: bool = False) -> None:
    """
    Trains the model on the given dataloader.

    Parameters:
    - model: Neural network model (VAE or Classifier)
    - optimizer: Optimizer
    - dataloader: DataLoader for the current batch of tasks
    - device: Device to train on ('cuda' or 'cpu')
    - epochs: Number of training epochs
    - is_classifier: Boolean flag indicating if model is a classifier for loss function selection
    """
    model.train()
    criterion = nn.CrossEntropyLoss() if is_classifier else None
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            if is_classifier:
                outputs = model(images)
                total_loss = criterion(outputs, labels)
            else:
                z_mean, z_log_var, reconstruction = model(images)
                total_loss, reconstruction_loss, kl_loss = model.loss_function(images, labels, z_mean, z_log_var, reconstruction)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}')

    logger.info('Finished Training Task')

def train_incremental_tasks(dir_path: str, device: torch.device, data_path: str, labels_path: str, batch_size: int, epochs: int, cbp: bool) -> None:
    """
    Trains the VAE model and Classifier incrementally with replay mechanism.

    Parameters:
    - dir_path: Directory to save models
    - device: Device to train on ('cuda' or 'cpu')
    - data_path: Path to the dataset file
    - labels_path: Path to the labels file
    - batch_size: Training batch size
    - epochs: Number of epochs per task
    - cbp: Continual backprop boolean flag
    """
    dataset = CustomNumpyDataset(data_path, labels_path, one_hot_labels=False)
    train_set, _ = data.random_split(dataset, [int(0.75 * len(dataset)), int(0.25 * len(dataset)) + 1], generator=torch.Generator().manual_seed(42))

    total_classes = 50
    num_classes_per_task = 2
    current_classes = list(range(num_classes_per_task))

    model_vae = VAE(conditional=True, alpha=10., continual_backprop=cbp).to(device)
    model_classifier = FeatureClassifier(input_size=2048, hidden1=512, hidden2=256, num_classes=total_classes, continual_backprop=cbp).to(device)

    for task in range(1, total_classes // num_classes_per_task + 1):
        logger.info(f"Training on Task {task}: Classes {current_classes}")

        train_subset = get_subset_of_classes(train_set, current_classes)

        if task > 1:  # REPLAY
            generated_images = []
            generated_labels = []
            previous_classes = list(range((task - 1) * num_classes_per_task))
            for class_label in previous_classes:
                samples = generate_samples(model_vae, model_classifier, class_label, 2000, device)
                generated_images.append(samples)
                generated_labels.extend([class_label] * samples.size(0))

            generated_images = torch.cat(generated_images, dim=0).cpu()
            generated_labels = torch.tensor(generated_labels, dtype=torch.long).cpu()

            replay_dataset = data.TensorDataset(generated_images, generated_labels)
            train_subset = data.ConcatDataset([train_subset, replay_dataset])

        train_loader = data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Train Classifier
        optimizer_classifier = optim.Adam(model_classifier.parameters(), lr=1e-4)
        train_model(model_classifier, optimizer_classifier, train_loader, device, epochs, is_classifier=True)

        # Train VAE
        optimizer_vae = optim.Adam(model_vae.parameters(), lr=1e-4)
        train_model(model_vae, optimizer_vae, train_loader, device, epochs)

        if task * num_classes_per_task < total_classes:
            current_classes = list(range(task * num_classes_per_task, (task + 1) * num_classes_per_task))

        model_path = os.path.join(dir_path, f'generator_class_incremental_with_replay_task{task}.pth')
        torch.save(model_vae.state_dict(), model_path)
        model_path = os.path.join(dir_path, f'classifier_class_incremental_with_replay_task{task}.pth')
        torch.save(model_classifier.state_dict(), model_path)

    logger.info("Finished Incremental Training")

if __name__ == "__main__":
    main()