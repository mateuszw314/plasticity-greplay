import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets, transforms

from cbp_linear import CBPLinear
from cbp_conv import CBPConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        

class VAE(nn.Module):
    def __init__(self, alpha=1., conditional=False, continual_backprop=False, num_classes = 50, latent_dim = 10, input_size=2048):
        super(VAE, self).__init__()
        self.alpha = alpha
        self.conditional = conditional
        self.encoder = self.get_encoder(latent_dim, input_size)
        self.sampling = Sampling()
        self.decoder = self.get_decoder(latent_dim, input_size)
        self.continual_backprop = continual_backprop


        if self.conditional:
            self.prior_means = nn.Parameter(torch.randn(num_classes, latent_dim))
            self.prior_logvars = nn.Parameter(torch.randn(num_classes, latent_dim))

    def get_encoder(self, latent_dim, input_size=2048):

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                self.dense1 = nn.Linear(input_size, 128)
                self.dense2 = nn.Linear(128,128)
                self.z_mean = nn.Linear(128, latent_dim)
                self.z_log_var = nn.Linear(128, latent_dim)

            def forward(self, x):
                x = torch.relu(self.dense1(x))
                x = torch.relu(self.dense2(x))
                z_mean = self.z_mean(x)
                z_log_var = self.z_log_var(x)
                return z_mean, z_log_var
        
        return Encoder().to(device)

    def get_decoder(self, latent_dim=100, output_size=2048):
        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()
                self.fc1 = nn.Linear(latent_dim, 128)
                self.bn1 = nn.BatchNorm1d(128)
                self.fc2 = nn.Linear(128, 128)
                self.bn2 = nn.BatchNorm1d(128)
                self.fc3 = nn.Linear(128, output_size)
                self.activation = nn.ReLU()
                

            def forward(self, x):
                x = self.activation(self.bn1(self.fc1(x)))
                x = self.activation(self.bn2(self.fc2(x)))
                x = self.fc3(x)
                return x

        return Decoder().to(device)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def loss_function(self, images, labels, z_mean, z_log_var, reconstruction):
        # Reconstruction loss
        reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, images, reduction='mean') #perhaps sum instead of mean 

        # KL Divergence loss
        if self.conditional:
            tmp_prior_means = self.prior_means[labels]
            tmp_prior_logvars = self.prior_logvars[labels]
            kl_loss = 1 + z_log_var - tmp_prior_logvars - (1 / torch.exp(tmp_prior_logvars)) * \
                      (torch.square(z_mean - tmp_prior_means) + torch.exp(z_log_var))
        else:
            kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)
        
        kl_loss = -0.5 * torch.mean(torch.sum(kl_loss, dim=1))
        total_loss = reconstruction_loss * self.alpha + kl_loss
        return total_loss, reconstruction_loss, kl_loss



