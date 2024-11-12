#!/usr/bin/env python
# coding: utf-8

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from vae_models import VAE
from feature_classifier import FeatureClassifier
from feature_dataset import CustomNumpyDataset
import logging
from datetime import datetime
import random
import yaml
import argparse
import shutil
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Set up logging
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(config):
    dataset_name = config['dataset']
    if dataset_name == 'custom':
        data_path = config['custom_dataset']['data_path']
        labels_path = config['custom_dataset']['labels_path']
        dataset = CustomNumpyDataset(data_path, labels_path, one_hot_labels=False, vector_len=config['custom_dataset']['vector_len'])
    elif dataset_name == 'emnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Ensure images are grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1] range
            transforms.Lambda(lambda x: torch.flatten(x))
        ])
        dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
        test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
        return dataset, test_dataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # For custom dataset, return only single dataset split
    dataset_len = len(dataset)
    train_set, test_set = data.random_split(dataset, [int(0.75 * dataset_len), int(0.25 * dataset_len) + 1], generator=torch.Generator().manual_seed(42))
    #train_set, test_set = data.random_split(dataset, [int(0.75 * dataset_len), int(0.25 * dataset_len)], generator=torch.Generator().manual_seed(42))
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

def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")
    for experiment in range(config['num_experiments']):
        #try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_type = config['experiment_type']
        experiment_path = f'replay_{experiment_type}_{experiment}_{timestamp}'
        root_path = os.path.join(config['output_dir'], experiment_path)
        os.makedirs(root_path, exist_ok=True)
        
        # Save a copy of the config file to the output directory
        shutil.copy(args.config, os.path.join(root_path, 'config.yaml'))

        # Randomize class order
        all_classes = list(range(config['num_classes']))
        random.shuffle(all_classes)

        train_incremental_tasks(root_path, device, all_classes, config)
        #except Exception as e:
        #    logger.error(f"Experiment {experiment} failed with error: {e}")
        #    continue

    logger.info("Finished all experiments")

def get_subset_of_classes(dataset, class_indices):
    # If the dataset is a Subset, get the indices and corresponding labels
    if isinstance(dataset, data.Subset):
        if isinstance(dataset.dataset, CustomNumpyDataset):
            # For custom dataset
            targets = dataset.dataset.labels[dataset.indices]
        elif isinstance(dataset.dataset, datasets.EMNIST):
            # For EMNIST dataset
            targets = torch.tensor(dataset.dataset.targets)[dataset.indices]
        else:
            raise ValueError("Unsupported dataset type")

        mask = torch.isin(targets, torch.tensor(class_indices))
        subset_indices = [dataset.indices[i] for i, m in enumerate(mask) if m]
        return data.Subset(dataset.dataset, subset_indices)
    
    # If the dataset is an EMNIST dataset directly
    elif isinstance(dataset, datasets.EMNIST):
        labels = torch.tensor(dataset.targets)
        mask = torch.isin(labels, torch.tensor(class_indices))
        subset_indices = torch.where(mask)[0].tolist()
        return data.Subset(dataset, subset_indices)
    
    else:
        raise ValueError("Unsupported dataset type")



def generate_samples(generator: VAE, classifier: FeatureClassifier, class_to_generate: int, examples_to_generate: int, device: torch.device, softmax_filter: float = 0., replay_limit: int = 15) -> torch.Tensor:
    accepted_samples = []
    generator.eval()
    classifier.eval()
    with torch.no_grad():
        counter = 0
        while (len(accepted_samples) < examples_to_generate) and counter < replay_limit:
            counter += 1
            prior_means = generator.prior_means[class_to_generate].to(device)
            prior_logvars = generator.prior_logvars[class_to_generate].to(device)
            std_devs = torch.sqrt(torch.exp(prior_logvars))
            means_extended = prior_means.unsqueeze(0).expand(examples_to_generate, -1)
            std_devs_extended = std_devs.unsqueeze(0).expand(examples_to_generate, -1)
            random_latent_vectors = torch.normal(means_extended, std_devs_extended).to(device)
            generated_images = generator.decoder(random_latent_vectors)

            outputs = classifier(generated_images)
            softmax_scores, predicted_labels = torch.max(outputs, 1)

            for img, pred_label, score in zip(generated_images, predicted_labels, softmax_scores):
                if (pred_label.item() == class_to_generate) and (score >= softmax_filter):
                    accepted_samples.append(img)
                if len(accepted_samples) >= examples_to_generate:
                    break

    return accepted_samples

def train_model(model: nn.Module, optimizer: optim.Optimizer, dataloader: data.DataLoader, device: torch.device, epochs: int, is_classifier: bool = False) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss() if is_classifier else None
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
                total_loss, reconstruction_loss, kl_loss = model.loss_function(images, labels, z_mean, z_log_var, reconstruction)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        #logger.info(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}')
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}')
    print('Finished Training Task')

def evaluate_model(model: nn.Module, dataloader: data.DataLoader, device: torch.device) -> float:
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

def train_incremental_tasks(dir_path: str, device: torch.device,   class_order: list,  config: dict) -> None:



    feature_classifier_params = config['feature_classifier']
    generator_params = config['generator']
    epochs = config['epochs']
    batch_size = config['batch_size']
    cbp_config = config['cbp_config']
    
    
    train_set, test_set = load_dataset(config)

    total_classes = config['num_classes']
    num_classes_per_task = 2
    current_classes = class_order[:num_classes_per_task]

    model_vae = VAE(conditional=True, alpha=generator_params['alpha'],
                    continual_backprop=generator_params['cbp'],
                    num_classes = generator_params['num_classes'],
                    latent_dim = generator_params['latent_dim'],
                    encoder_config=generator_params['encoder_config'],
                    decoder_config=generator_params['decoder_config'],
                    cbp_config = cbp_config
                    ).to(device)
    logger.info(f'Generator param size: {count_parameters(model_vae)}')
    print('Generator init')
    model_classifier = FeatureClassifier(
        input_size=feature_classifier_params['input_size'], 
        hidden1=feature_classifier_params['hidden1'], 
        hidden2=feature_classifier_params['hidden2'], 
        num_classes=feature_classifier_params['num_classes'], 
        continual_backprop=feature_classifier_params['cbp'],
        cbp_config = cbp_config
    ).to(device)
    logger.info(f'Classifier param size: {count_parameters(model_classifier)}')

    print('Classifier init')
    accuracy_file_path = os.path.join(dir_path, 'accuracy.txt')
    
    for task in range(1, total_classes // num_classes_per_task + 1):
        print(f"Training on Task {task}: Classes {current_classes}")

        train_subset = get_subset_of_classes(train_set, current_classes)
       
        if task > 1:  # REPLAY
            generated_images = []
            generated_labels = []
            previous_classes = class_order[:((task - 1) * num_classes_per_task)]
            logger.info(f'Generating replay for task {task}')
            for class_label in previous_classes:
                samples = generate_samples(model_vae, model_classifier, class_label, 2000, device, config['softmax_filter'])
                logger.info(f'Class: {class_label} Replay: {len(samples)}')
                if len(samples) > 0:
                    samples = torch.stack(samples)
                    generated_images.append(samples)
                    generated_labels.extend([class_label] * samples.size(0))
                else:
                    continue
            if len(generated_images) > 0:
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
            current_classes = class_order[(task * num_classes_per_task):((task + 1) * num_classes_per_task)]

        model_path = os.path.join(dir_path, f'generator_class_incremental_with_replay_task{task}.pth')
        torch.save(model_vae.state_dict(), model_path)
        model_path = os.path.join(dir_path, f'classifier_class_incremental_with_replay_task{task}.pth')
        torch.save(model_classifier.state_dict(), model_path)
        
        # Evaluate the classifier on all seen classes
        seen_classes = class_order[:task * num_classes_per_task]
        test_subset = get_subset_of_classes(test_set, seen_classes)
        test_loader = data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        accuracy = evaluate_model(model_classifier, test_loader, device)
        print(f"Task {task} - Accuracy on seen classes: {accuracy:.2f}%")

        # Store accuracy
        with open(accuracy_file_path, 'a') as f:
            f.write(f"Task {task}, Accuracy: {accuracy:.2f}%\n")

    logger.info("Finished Incremental Training")

if __name__ == "__main__":
    main()
