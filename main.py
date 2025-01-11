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
import utils as u
import yaml
from feature_classifier import FeatureClassifier
from feature_dataset import CustomNumpyDataset
from vae_models import VAE

# Set up logging
logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Parse command-line arguments and load the configuration file
    args = u.parse_args()
    config = u.load_config(args.config)
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    # Perform multiple experiments as specified in the configuration
    for experiment in range(config['num_experiments']):
        try:
            # Generate output directories
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_type = config['experiment_type']
            experiment_path = f'replay_{experiment_type}_{experiment}_{timestamp}'
            root_path = os.path.join(config['output_dir'], experiment_path)
            os.makedirs(root_path, exist_ok=True)

            # Save a copy of the config file to the output directory (for future reference)
            shutil.copy(args.config, os.path.join(root_path, 'config.yaml'))

            # Randomize class order – different order for each experiment instance
            all_classes = list(range(config['num_classes']))
            random.shuffle(all_classes)

            # Training
            train_incremental_tasks(root_path, device, all_classes, config)
        except Exception as e:
            logger.error(f"Experiment {experiment} failed with error: {e}")
            continue

    logger.info("Finished all experiments")


def generate_samples(generator: VAE, classifier: FeatureClassifier, class_to_generate: int, examples_to_generate: int,
                     device: torch.device, softmax_filter: float = 0., replay_limit: int = 15) -> torch.Tensor:
    """
        Generate replay samples for a given class.

        Args:
            generator (VAE): The VAE generator model.
            classifier (FeatureClassifier): The classifier model.
            class_to_generate (int): The class for which to generate samples.
            examples_to_generate (int): The number of examples to generate.
            device (torch.device): The device to run the models on.
            softmax_filter (float): Threshold for the softmax scores to filter generated samples.
            replay_limit (int): The limit on the number of replay attempts.

        Returns:
            torch.Tensor: Accepted generated samples.
        """
    accepted_samples = []
    generator.eval()
    classifier.eval()
    with torch.no_grad():
        counter = 0
        while (len(accepted_samples) < examples_to_generate) and (counter < replay_limit):
            counter += 1
            prior_means = generator.prior_means[class_to_generate].to(device)
            prior_logvars = generator.prior_logvars[class_to_generate].to(device)
            std_devs = torch.sqrt(torch.exp(prior_logvars))
            means_extended = prior_means.unsqueeze(0).expand(examples_to_generate, -1)
            std_devs_extended = std_devs.unsqueeze(0).expand(examples_to_generate, -1)
            random_latent_vectors = torch.normal(means_extended, std_devs_extended).to(device)
            generated_images = generator.decoder(random_latent_vectors)

            outputs = nn.Softmax(dim=1)(classifier(generated_images))
            softmax_scores, predicted_labels = torch.max(outputs, 1)

            for img, pred_label, score in zip(generated_images, predicted_labels, softmax_scores):
                if (pred_label.item() == class_to_generate) and (score >= softmax_filter):
                    accepted_samples.append(img)
                if len(accepted_samples) >= examples_to_generate:
                    break

    return accepted_samples


def train_incremental_tasks(dir_path: str, device: torch.device, class_order: list, config: dict) -> None:
    """
        Train incremental tasks using the VAE and classifier models. This function contains
        the main logic of the experiment.

        Args:
            dir_path (str): Directory path to save the results.
            device (torch.device): The device to run the models on.
            class_order (list): The order of classes for the incremental tasks.
            config (dict): Configuration dictionary.
        """
    feature_classifier_params = config['feature_classifier']
    generator_params = config['generator'] if 'generator' in config.keys() else None
    epochs = config['epochs']
    batch_size = config['batch_size']
    cbp_config = config['cbp_config']

    # Data loading
    train_set, test_set = u.load_dataset(config)

    total_classes = config['num_classes']
    num_classes_per_task = 2
    current_classes = class_order[:num_classes_per_task]

    # Initialize the necessary models
    if 'generator' in config.keys():
        model_vae = VAE(conditional=True, alpha=generator_params['alpha'], continual_backprop=generator_params['cbp'],
                        num_classes=generator_params['num_classes'], latent_dim=generator_params['latent_dim'],
                        encoder_config=generator_params['encoder_config'],
                        decoder_config=generator_params['decoder_config'], cbp_config=cbp_config).to(device)
        logger.info(f'Generator param size: {u.count_parameters(model_vae)}')
        print('Generator initialized')
    else:
        print('No generator configuration, using exact replay')
    model_classifier = FeatureClassifier(input_size=feature_classifier_params['input_size'],
        hidden1=feature_classifier_params['hidden1'], hidden2=feature_classifier_params['hidden2'],
        num_classes=feature_classifier_params['num_classes'], continual_backprop=feature_classifier_params['cbp'],
        cbp_config=cbp_config).to(device)
    logger.info(f'Classifier param size: {u.count_parameters(model_classifier)}')

    print('Classifier initialized')
    accuracy_file_path = os.path.join(dir_path, 'accuracy.txt')

    for task in range(1, total_classes // num_classes_per_task + 1):
        print(f"Training on Task {task}: Classes {current_classes}")

        # Get a subset of the original data corresponding to the classes in the current task.
        train_subset = u.get_subset_of_classes(train_set, current_classes)

        # Replay previous classes
        if task > 1:
            generated_images = []
            generated_labels = []
            previous_classes = class_order[:((task - 1) * num_classes_per_task)]
            logger.info(f'Generating replay for task {task}')
            if 'generator' in config.keys():
                for class_label in previous_classes:
                    samples = generate_samples(model_vae, model_classifier, class_label, 2000, device,
                                               config['softmax_filter'])

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
            else: # exact replay
                full_replay_subset = u.get_subset_of_classes(train_set, previous_classes)
                samples_per_class = 20
                random_indices = torch.randperm(len(full_replay_subset))[:(samples_per_class*len(previous_classes))]
                replay_dataset = data.Subset(full_replay_subset, random_indices)
                train_subset = data.ConcatDataset([train_subset, replay_dataset])


        train_loader = data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)

        # Train the classifier
        optimizer_classifier = optim.Adam(model_classifier.parameters(), lr=1e-4)
        u.train_model(model_classifier, optimizer_classifier, train_loader, device, epochs, is_classifier=True)

        # Train the generator
        if 'generator' in config.keys():
            optimizer_vae = optim.Adam(model_vae.parameters(), lr=1e-4)
            u.train_model(model_vae, optimizer_vae, train_loader, device, epochs)

        # Select the next chunk of classes
        if task * num_classes_per_task < total_classes:
            current_classes = class_order[(task * num_classes_per_task):((task + 1) * num_classes_per_task)]

        # Save the models
        if 'generator' in config.keys(): #TODO: reduce the number of if-checks
            model_path = os.path.join(dir_path, f'generator_class_incremental_with_replay_task{task}.pth')
            torch.save(model_vae.state_dict(), model_path)
        model_path = os.path.join(dir_path, f'classifier_class_incremental_with_replay_task{task}.pth')
        torch.save(model_classifier.state_dict(), model_path)

        # Evaluate the classifier on all seen classes
        seen_classes = class_order[:task * num_classes_per_task]
        test_subset = u.get_subset_of_classes(test_set, seen_classes)
        test_loader = data.DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

        accuracy = u.evaluate_classifier(model_classifier, test_loader, device)
        print(f"Task {task} - Accuracy on seen classes: {accuracy:.2f}%")

        # Store accuracy results
        with open(accuracy_file_path, 'a') as f:
            f.write(f"Task {task}, Accuracy: {accuracy:.2f}%\n")

    logger.info("Finished Incremental Training")


if __name__ == "__main__":
    main()
