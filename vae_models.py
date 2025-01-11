import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from cbp_conv import CBPConv
from cbp_linear import CBPLinear
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sampling(nn.Module):
    """
        Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.
    """

    def forward(self, z_mean, z_log_var):
        """
                Sample a latent vector z from the normal distribution parameterized by (z_mean, z_log_var).

                Args:
                    z_mean (torch.Tensor): Mean of the latent distribution.
                    z_log_var (torch.Tensor): Log variance of the latent distribution.

                Returns:
                    torch.Tensor: Sampled latent vector.
                """
        batch, dim = z_mean.size()
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon


# Default configuration dictionaries
default_encoder_config = {'input_size': 2048, 'hidden1': 128, 'hidden2': 128}
default_decoder_config = {'output_size': 2048, 'hidden1': 128, 'hidden2': 128}
default_cbp_config = {'replacement_rate': 10e-4, 'maturity_threshold': 50, 'decay_rate': 0}


class LinearEncoder(nn.Module):
    def __init__(self, latent_dim, config, continual_backprop, cbp_config):
        super(LinearEncoder, self).__init__()
        self.dense1 = nn.Linear(config['input_size'], config['hidden1'])
        self.dense2 = nn.Linear(config['hidden1'], config['hidden2'])
        self.z_mean = nn.Linear(config['hidden2'], latent_dim)
        self.z_log_var = nn.Linear(config['hidden2'], latent_dim)
        self.activation = nn.ReLU()

        if continual_backprop:
            self.extractor = nn.Sequential(self.dense1, self.activation,
                                           CBPLinear(in_layer=self.dense1, out_layer=self.dense2,
                                                     replacement_rate=cbp_config['replacement_rate'],
                                                     maturity_threshold=cbp_config['maturity_threshold'],
                                                     decay_rate=cbp_config['decay_rate']), self.dense2,
                                           self.activation, )
        else:
            self.extractor = nn.Sequential(self.dense1, self.activation, self.dense2, self.activation, )

    def forward(self, x):
        x = self.extractor(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim, config, continual_backprop, cbp_config):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)

        # Calculate flattened size
        self.flatten_size = 512 * (64 // 16) * (64 // 16)

        # Latent space
        self.z_mean = nn.Linear(self.flatten_size, latent_dim)
        self.z_log_var = nn.Linear(self.flatten_size, latent_dim)

        if continual_backprop:
            self.extractor = nn.Sequential(
                self.conv1, nn.ReLU(),
                CBPConv(in_layer=self.conv1, out_layer=self.conv2, replacement_rate=cbp_config['replacement_rate'],
                        maturity_threshold=cbp_config['maturity_threshold'], decay_rate=cbp_config['decay_rate']),
                self.conv2, nn.ReLU(),
                CBPConv(in_layer=self.conv2, out_layer=self.conv3, replacement_rate=cbp_config['replacement_rate'],
                        maturity_threshold=cbp_config['maturity_threshold'], decay_rate=cbp_config['decay_rate']),
                self.conv3, nn.ReLU(),
                CBPConv(in_layer=self.conv3, out_layer=self.conv4, replacement_rate=cbp_config['replacement_rate'],
                        maturity_threshold=cbp_config['maturity_threshold'], decay_rate=cbp_config['decay_rate']),
                self.conv4, nn.ReLU()
            )
        else:
            self.extractor = nn.Sequential(
                self.conv1, nn.ReLU(),
                self.conv2, nn.ReLU(),
                self.conv3, nn.ReLU(),
                self.conv4, nn.ReLU()
            )

    def forward(self, x):
        x = self.extractor(x)
        x = nn.Flatten()(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var


class LinearDecoder(nn.Module):
    def __init__(self, latent_dim, config, continual_backprop, cbp_config):
        super(LinearDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, config['hidden1'])
        self.bn1 = nn.BatchNorm1d(config['hidden1'])
        self.fc2 = nn.Linear(config['hidden1'], config['hidden2'])
        self.bn2 = nn.BatchNorm1d(config['hidden2'])
        self.fc3 = nn.Linear(config['hidden2'], config['output_size'])
        self.activation = nn.ReLU()

        if continual_backprop:
            self.model = nn.Sequential(self.fc1, self.bn1, self.activation,
                CBPLinear(in_layer=self.fc1, out_layer=self.fc2,
                          replacement_rate=cbp_config['replacement_rate'],
                          maturity_threshold=cbp_config['maturity_threshold'],
                          decay_rate=cbp_config['decay_rate']), self.fc2, self.bn2, self.activation,
                CBPLinear(in_layer=self.fc2, out_layer=self.fc3,
                          replacement_rate=cbp_config['replacement_rate'],
                          maturity_threshold=cbp_config['maturity_threshold'],
                          decay_rate=cbp_config['decay_rate']), self.fc3

            )
        else:
            self.model = nn.Sequential(self.fc1, self.bn1, self.activation, self.fc2, self.bn2, self.activation,
                self.fc3

            )

    def forward(self, x):
        return self.model(x)


class ConvDecoder(nn.Module): #for now without cbp
    def __init__(self, latent_dim, config, continual_backprop, cbp_config):
        super(ConvDecoder, self).__init__()

        self.flatten_size = 512 * (64 // 16) * (64 // 16)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.decoder_input(x)
        x = x.view(-1, 512, 4, 4)
        return self.decoder(x)


class VAE(nn.Module):
    """
        Variational Autoencoder (VAE) class with optional conditional and continual backpropagation (CBP) support.
    """

    def __init__(self, alpha=1., conditional=False, continual_backprop=False, num_classes=50, latent_dim=10,
                 encoder_config=default_encoder_config, decoder_config=default_decoder_config,
                 cbp_config=default_cbp_config, dataset=None):
        """
                Initialize the VAE model.

                Args:
                    alpha (float): Weighting factor for the reconstruction loss.
                    conditional (bool): If True, use conditional VAE with class-specific priors.
                    continual_backprop (bool): If True, use CBP layers.
                    num_classes (int): Number of possible classes to generate.
                    latent_dim (int): Dimensionality of the latent space.
                    encoder_config (dict): Configuration dictionary for the encoder.
                    decoder_config (dict): Configuration dictionary for the decoder.
                    cbp_config (dict): Configuration dictionary for CBP layers.
                """
        super(VAE, self).__init__()
        self.alpha = alpha
        self.conditional = conditional
        self.encoder = self.get_encoder(latent_dim, encoder_config, continual_backprop, cbp_config, dataset)
        self.sampling = Sampling()
        self.decoder = self.get_decoder(latent_dim, decoder_config, continual_backprop, cbp_config, dataset)

        if self.conditional:
            self.prior_means = nn.Parameter(torch.randn(num_classes, latent_dim))
            self.prior_logvars = nn.Parameter(torch.randn(num_classes, latent_dim))

    def get_encoder(self, latent_dim, config, continual_backprop, cbp_config, dataset):
        """
                Build the encoder model.

                Args:
                    latent_dim (int): Dimensionality of the latent space.
                    config (dict): Configuration dictionary for the encoder.
                    continual_backprop (bool): If True, use CBP layers.
                    cbp_config (dict): Configuration dictionary for CBP layers.

                Returns:
                    nn.Module: Encoder model.
        """

        if dataset == 'imagenet':
            return ConvEncoder(latent_dim, config, continual_backprop, cbp_config).to(device)
        else:
            return LinearEncoder(latent_dim, config, continual_backprop, cbp_config).to(device)


    def get_decoder(self, latent_dim, config, continual_backprop, cbp_config, dataset):
        """
                Build the decoder model.

                Args:
                    latent_dim (int): Dimensionality of the latent space.
                    config (dict): Configuration dictionary for the decoder.
                    continual_backprop (bool): If True, use CBP layers.
                    cbp_config (dict): Configuration dictionary for CBP layers.

                Returns:
                    nn.Module: Decoder model.
                """

        if dataset == 'imagenet':
            return ConvDecoder(latent_dim, config, continual_backprop, cbp_config).to(device)
        else:
            return LinearDecoder(latent_dim, config, continual_backprop, cbp_config).to(device)

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, reconstruction

    def loss_function(self, images, labels, z_mean, z_log_var, reconstruction):
        """
                Calculate the loss function for the VAE.

                Args:
                    images (torch.Tensor): Input images.
                    labels (torch.Tensor): Ground-truth labels for conditional VAE.
                    z_mean (torch.Tensor): Mean of the latent distribution.
                    z_log_var (torch.Tensor): Log variance of the latent distribution.
                    reconstruction (torch.Tensor): Reconstructed images.

                Returns:
                    tuple: Total loss, reconstruction loss, and KL divergence loss.
                """
        # Reconstruction loss
        reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, images, reduction='mean')

        # KL Divergence loss
        if self.conditional:
            tmp_prior_means = self.prior_means[labels]
            tmp_prior_logvars = self.prior_logvars[labels]
            kl_loss = 1 + z_log_var - tmp_prior_logvars - (1 / torch.exp(tmp_prior_logvars)) * (
                        torch.square(z_mean - tmp_prior_means) + torch.exp(z_log_var))
        else:
            kl_loss = 1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)

        kl_loss = -0.5 * torch.mean(torch.sum(kl_loss, dim=1))
        total_loss = reconstruction_loss * self.alpha + kl_loss
        return total_loss, reconstruction_loss, kl_loss
