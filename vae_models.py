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

default_encoder_config={'input_size':2048, 'hidden1': 128, 'hidden2': 128 }
default_decoder_config={'output_size':2048, 'hidden1': 128, 'hidden2': 128 } 

        

class VAE(nn.Module):
    def __init__(self,   alpha=1., conditional=False, continual_backprop=False, num_classes = 50, latent_dim = 10, encoder_config = default_encoder_config, decoder_config = default_decoder_config):
        super(VAE, self).__init__()
        self.alpha = alpha
        self.conditional = conditional
        self.encoder = self.get_encoder(latent_dim, encoder_config, continual_backprop)
        self.sampling = Sampling()
        self.decoder = self.get_decoder(latent_dim, decoder_config, continual_backprop)


        if self.conditional:
            self.prior_means = nn.Parameter(torch.randn(num_classes, latent_dim))
            self.prior_logvars = nn.Parameter(torch.randn(num_classes, latent_dim))

    def get_encoder(self, latent_dim, config, continual_backprop):

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                self.dense1 = nn.Linear(config['input_size'], config['hidden1'])
                self.dense2 = nn.Linear(config['hidden1'],config['hidden2'])
                self.z_mean = nn.Linear(config['hidden2'], latent_dim)
                self.z_log_var = nn.Linear(config['hidden2'], latent_dim)
                self.activation = nn.ReLU()

                if continual_backprop:
                    self.extractor = nn.Sequential(
                        self.dense1,
                        self.activation,
                        CBPLinear(in_layer=self.dense1, out_layer=self.dense2),
                        self.dense2,
                        self.activation,
                    )
                else:
                    self.extractor = nn.Sequential(
                        self.dense1,
                        self.activation,
                        self.dense2,
                        self.activation,
                    )

            def forward(self, x):
                x = self.extractor(x)
                z_mean = self.z_mean(x)
                z_log_var = self.z_log_var(x)
                return z_mean, z_log_var
        
        return Encoder().to(device)

    def get_decoder(self, latent_dim, config, continual_backprop):
        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()
                self.fc1 = nn.Linear(latent_dim, config['hidden1'])
                self.bn1 = nn.BatchNorm1d(config['hidden1'])
                self.fc2 = nn.Linear(config['hidden1'], config['hidden2'])
                self.bn2 = nn.BatchNorm1d(config['hidden2'])
                self.fc3 = nn.Linear(config['hidden2'], config['output_size'])
                self.activation = nn.ReLU()

                if continual_backprop:
                    self.model = nn.Sequential(
                        self.fc1,
                        self.bn1,
                        self.activation,
                        CBPLinear(in_layer=self.fc1, out_layer=self.fc2),
                        self.fc2,
                        self.bn2,
                        self.activation,
                        CBPLinear(in_layer=self.fc2, out_layer=self.fc3),
                        self.fc3
                        
                    )
                else:
                    self.model = nn.Sequential(
                        self.fc1,
                        self.bn1,
                        self.activation,
                        self.fc2,
                        self.bn2,
                        self.activation,
                        self.fc3
                        
                    )
                

            def forward(self, x):
                return self.model(x)

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



