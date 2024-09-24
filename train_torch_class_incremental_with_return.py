#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets, transforms

from cifarvae import CIFARVAE, CIFARVAE_continual


# In[2]:
# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #todo: make it not global


def get_subset_of_classes(dataset, class_indices):
    """
    Returns a subset of the dataset containing only the specified classes.
    
    Parameters:
    - dataset: The original dataset
    - class_indices: List of class indices to include in the subset
    
    Returns:
    - Subset of the dataset containing only the specified classes
    """
    targets = torch.tensor(dataset.targets)
    mask = torch.isin(targets, torch.tensor(class_indices))
    subset_indices = mask.nonzero(as_tuple=False).squeeze().tolist()
    return torch.utils.data.Subset(dataset, subset_indices)


# In[3]:









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
            if i % 100 == 99:
                running_loss = 0.0
        print(f'Epoch {epoch+1} completed. Training loss: {running_loss / len(dataloader)}')

    print('Finished Training Task')


# In[12]:


def train_incremental_tasks():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    
    num_classes_per_task = 10
    total_classes = 100
    current_classes = list(range(0, num_classes_per_task))

    model = CIFARVAE(conditional=True, alpha=1.).to(device)
    #model = CIFARVAE_continual(conditional=True, alpha=1.).to(device)
    
    for task in range(1, total_classes // num_classes_per_task + 1):
        print(f"\nTraining on Task {task}: Classes {current_classes}")

        train_subset = get_subset_of_classes(full_dataset, current_classes)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=1024, shuffle=True, num_workers=2)

        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        train_incremental(model, optimizer, train_loader, device, epochs=100)

        if task * num_classes_per_task < total_classes:
            current_classes.extend(range(task * num_classes_per_task, (task + 1) * num_classes_per_task))
    
    print("Finished Incremental Training")
    return model


# In[13]:


model = train_incremental_tasks()

torch.save(model.state_dict(), 'generator_class_incremental_linear_loss_corrected.pth')
# In[ ]:


#torch.save(model.state_dict(), 'generator_test.pth')


# In[4]:


#! jupyter nbconvert --to script train_torch.ipynb --output train_torch_full


# In[ ]:




