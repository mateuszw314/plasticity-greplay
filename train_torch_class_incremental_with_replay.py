#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets, transforms

from vae_models import VAE
from feature_dataset import CustomNumpyDataset
import numpy as np


# In[2]:
# Set CUDA device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #todo: make it not global
print(device)

def get_subset_of_classes(dataset, class_indices):
    """
    Returns a subset of the dataset containing only the specified classes.
    
    Parameters:
    - dataset: The original dataset
    - class_indices: List of class indices to include in the subset
    
    Returns:
    - Subset of the dataset containing only the specified classes
    """
    targets = dataset[:][1]#torch.tensor(dataset.labels)
    mask = torch.isin(targets, torch.tensor(class_indices))
    subset_indices = mask.nonzero(as_tuple=False).squeeze().tolist()
    return torch.utils.data.Subset(dataset, subset_indices)


# In[3]:






def generate_samples(generator, class_to_generate, examples_to_generate):
    with torch.no_grad():
        prior_means = generator.prior_means[class_to_generate]  # Shape (100,)
        prior_logvars = generator.prior_logvars[class_to_generate]  # Shape (100,)
        std_devs = torch.sqrt(torch.exp(prior_logvars))
        means_extended = prior_means.unsqueeze(0).expand(examples_to_generate, -1)  # Shape (15, 100)
        std_devs_extended = std_devs.unsqueeze(0).expand(examples_to_generate, -1)
        random_latent_vectors = torch.normal(means_extended, std_devs_extended)
        generated_images = generator.decoder((random_latent_vectors))
    return generated_images





# In[9]:





def train_incremental(model, optimizer, dataloader, device, epochs=10):
    """
    Trains the model incrementally on the given dataloader.

    Parameters:
    - model: The neural network model.
    - criterion: The loss function.
    - optimizer: The optimizer.
    - dataloader: The DataLoader for the current batch of tasks.
    - device: Device to train on ('cuda' or 'cpu').
    - epochs: Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad()
            z_mean, z_log_var, reconstruction = model(images)
            total_loss, reconstruction_loss, kl_loss = model.loss_function(images, labels, z_mean, z_log_var, reconstruction)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f'Epoch {epoch+1} completed. Training loss: {running_loss / len(dataloader)}')

    print('Finished Training Task')


# In[12]:


def train_incremental_tasks(dir_path, cbp):
    
    data_path = 'data/core50_features/full.npy'
    labels_path = 'data/core50_features/full_labels.npy'
    dataset = CustomNumpyDataset(data_path, labels_path, one_hot_labels=False)
    train_set, test_set = torch.utils.data.random_split(dataset=dataset, lengths=[int(np.floor(0.75 * len(dataset))), int(np.ceil(0.25 * len(dataset)))], generator=torch.Generator().manual_seed(42))

    total_classes = 50
    num_classes_per_task = 2
    batch_size=1024
    epochs=100
    current_classes = list(range(0, num_classes_per_task))

    model = VAE(conditional=True, alpha=10., continual_backprop=cbp).to(device)

    for task in range(1, total_classes // num_classes_per_task + 1):
        print(f"\nTraining on Task {task}: Classes {current_classes}")

        train_subset = get_subset_of_classes(train_set, current_classes)

        
        if task > 1: # REPLAY
            generated_images = []
            generated_labels = []
            previous_classes = list(range(0, task*num_classes_per_task))
            for class_label in previous_classes:
                model.eval()
                samples = generate_samples(model, class_label, int(len(train_subset)/2))
                generated_images.append(samples)
                generated_labels.extend([class_label] * samples.size(0))

        
            generated_images = torch.cat(generated_images, dim=0)
            generated_labels = torch.tensor(generated_labels)

            replay_dataset = torch.utils.data.TensorDataset(generated_images, generated_labels)
            train_subset = torch.utils.data.ConcatDataset([train_subset, replay_dataset])
            
            
        
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        train_incremental(model, optimizer, train_loader, device, epochs=epochs)

        if task * num_classes_per_task < total_classes:
            current_classes = list(range(task * num_classes_per_task, (task + 1) * num_classes_per_task))
        torch.save(model.state_dict(), f'{dir_path}/generator_class_incremental_with_replay_task{task}.pth')
        
    print("Finished Incremental Training")
    return model


# In[13]:

for experiment in range(15):
    root_path = f'models/replay_linear_{experiment}'
    os.mkdir(root_path)
    model = train_incremental_tasks(root_path, cbp=False)

# In[ ]:


#torch.save(model.state_dict(), 'generator_test.pth')


# In[4]:


#! jupyter nbconvert --to script train_torch.ipynb --output train_torch_full


# In[ ]:




