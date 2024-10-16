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

from vae_models import VAE
from feature_dataset import CustomNumpyDataset
import numpy as np

# In[2]:


# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(model, data, optimizer):
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    z_mean, z_log_var, reconstruction = model(images)
    total_loss, reconstruction_loss, kl_loss = model.loss_function(images, labels, z_mean, z_log_var, reconstruction)
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), reconstruction_loss.item(), kl_loss.item()

data_path = 'data/core50_features/full.npy'
labels_path = 'data/core50_features/full_labels.npy'
dataset = CustomNumpyDataset(data_path, labels_path, one_hot_labels=False)
(train_set, test_set) = torch.utils.data.random_split(dataset=dataset, lengths=[int(np.floor(0.75*len(dataset))), int(np.floor(0.25*len(dataset))+1)], generator=torch.Generator().manual_seed(42))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1024,
                                          shuffle=True, num_workers=2)

# Training setup
model = VAE(conditional=True, alpha=10.).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(1000):
    for i, data in enumerate(train_loader, 0):
        loss, reconstruction_loss, kl_loss = train_step(model, data, optimizer)
    print(f"[Epoch {epoch+1}] loss: {loss:.3f}, reconstruction loss: {reconstruction_loss:.3f}, kl loss: {kl_loss:.3f}")

# Save model weights
torch.save(model.state_dict(), 'generator_full_1000_alpha_10.pth')


# In[3]:




# In[ ]:




