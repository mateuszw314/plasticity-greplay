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

from cifarvae import CIFARVAE

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

# Data loading and transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

# Training setup
model = CIFARVAE(conditional=True, alpha=10.).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        loss, reconstruction_loss, kl_loss = train_step(model, data, optimizer)
    print(f"[Epoch {epoch+1}] loss: {loss:.3f}, reconstruction loss: {reconstruction_loss:.3f}, kl loss: {kl_loss:.3f}")

# Save model weights
torch.save(model.state_dict(), 'generator_full_1000_alpha_1_loss_corrected.pth')


# In[3]:




# In[ ]:




