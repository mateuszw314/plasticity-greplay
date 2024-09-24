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

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            self._create_conv_block(3, 16, 3, stride=1),
            self._create_conv_block(16, 32, 3, stride=2),
            self._create_conv_block(32, 64, 3, stride=2),
            self._create_conv_block(64, 128, 3, stride=2),
            self._create_conv_block(128, 256, 3, stride=2)
        )

    def _create_conv_block(self, in_channels, out_channels, kernel_size, stride):
        padding = self._calculate_padding(kernel_size, stride)
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return block

    def _calculate_padding(self, kernel_size, stride):
        if stride > 1:
            padding = (kernel_size - 1) // 2
        else:
            padding = kernel_size // 2
        return padding

    def forward(self, x):
        return self.model(x)


class Sampling(nn.Module):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    
    def forward(self, z_mean, z_log_var):
        batch, dim = z_mean.size()
        epsilon = torch.randn(batch, dim).to(z_mean.device)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon
        

class CIFARVAE(nn.Module):
    def __init__(self, alpha=10, conditional=False, continual_backprop=False):
        super(CIFARVAE, self).__init__()
        self.alpha = alpha
        self.conditional = conditional
        self.encoder = self.get_encoder()
        self.sampling = Sampling()
        self.decoder = self.get_decoder()
        self.continual_backprop = continual_backprop


        if self.conditional:
            self.prior_means = nn.Parameter(torch.randn(100, 100))
            self.prior_logvars = nn.Parameter(torch.randn(100, 100))

    def get_encoder(self, latent_dim=100):

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                self.feature_extractor = FeatureExtractor()#.to(device)
                self.feature_extractor.load_state_dict(torch.load('cifar_pretrained_feature_extractor.pth'))
                #self.feature_extractor.eval()
                #for param in self.feature_extractor.parameters():
                #    param.requires_grad=False
                self.flatten = nn.Flatten()
                self.dense1 = nn.Linear(256 * 2 * 2, 2000)
                self.dense2 = nn.Linear(2000, 2000)
                self.z_mean = nn.Linear(2000, latent_dim)
                self.z_log_var = nn.Linear(2000, latent_dim)

            def forward(self, x):
                x = self.feature_extractor(x)
                x = self.flatten(x)
                x = torch.relu(self.dense1(x))
                x = torch.relu(self.dense2(x))
                z_mean = self.z_mean(x)
                z_log_var = self.z_log_var(x)
                return z_mean, z_log_var
        
        return Encoder().to(device)

    def get_decoder(self, latent_dim=100):
        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()
                self.fc1 = nn.Linear(latent_dim, 2000)
                self.bn1 = nn.BatchNorm1d(2000)
                self.fc2 = nn.Linear(2000, 4 * 4 * 256)
                self.bn2 = nn.BatchNorm1d(4 * 4 * 256)
                self.fc3 = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)
                )

            def forward(self, x):
                x = torch.relu(self.bn1(self.fc1(x)))
                x = torch.relu(self.bn2(self.fc2(x)))
                x = x.view(-1, 256, 4, 4)
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


class CIFARVAE_continual(nn.Module):
    def __init__(self, alpha=10, conditional=False):
        super(CIFARVAE_continual, self).__init__()
        self.alpha = alpha
        self.conditional = conditional
        self.encoder = self.get_encoder()
        self.sampling = Sampling()
        self.decoder = self.get_decoder()


        if self.conditional:
            self.prior_means = nn.Parameter(torch.randn(100, 100))
            self.prior_logvars = nn.Parameter(torch.randn(100, 100))

    def get_encoder(self, latent_dim=100):

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                self.feature_extractor = FeatureExtractor()#.to(device)
                self.feature_extractor.load_state_dict(torch.load('cifar_pretrained_feature_extractor.pth'))
                #self.feature_extractor.eval()
                #for param in self.feature_extractor.parameters():
                #    param.requires_grad=False
                self.flatten = nn.Flatten()
                self.dense1 = nn.Linear(256 * 2 * 2, 2000)
                self.dense2 = nn.Linear(2000, 2000)
                self.z_mean = nn.Linear(2000, latent_dim)
                self.z_log_var = nn.Linear(2000, latent_dim)

                self.lcbp1 = CBPLinear(in_layer=self.dense1, out_layer=self.dense2)
                self.lcbp_mean = nn.Identity()#CBPLinear(in_layer=self.dense2, out_layer=self.z_mean, act_type='linear')
                self.lcbp_var = nn.Identity()#CBPLinear(in_layer=self.dense2, out_layer=self.z_log_var, act_type='linear')

            

            def forward(self, x):
                x = self.feature_extractor(x)
                x = self.flatten(x)
                x = self.lcbp1(torch.relu(self.dense1(x)))
                x = self.lcbp_mean(torch.relu(self.dense2(x))) #TODO: mozliwe, ze to nie bedzie pasowac. Problem jest taki, ze w CBPlinear nie mozna chyba ustawic dwoch warstw output. Teraz troche pomijamy z_log_var.
                z_mean = self.z_mean(x)
                z_log_var = self.lcbp_var(self.z_log_var(x))
                return z_mean, z_log_var
        
        return Encoder().to(device)

    def get_decoder(self, latent_dim=100):
        class Decoder(nn.Module):
            def __init__(self):
                super(Decoder, self).__init__()
                self.fc1 = nn.Linear(latent_dim, 2000)
                self.bn1 = nn.BatchNorm1d(2000)
                self.fc2 = nn.Linear(2000, 4 * 4 * 256)
                self.bn2 = nn.BatchNorm1d(4 * 4 * 256)
                self.fc3 = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)
                )
                
                self.lcbp1 = CBPLinear(in_layer=self.fc1, out_layer=self.fc2, bn_layer = self.bn1)
                #self.lcbp2 = CBPLinear(in_layer=self.fc2, out_layer=self.fc3[0], bn_layer = self.bn2) #to moze sie wywalic
            

            def forward(self, x):
                x = self.lcbp1(torch.relu(self.bn1(self.fc1(x))))
                x = torch.relu(self.bn2(self.fc2(x)))
                x = x.view(-1, 256, 4, 4)
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
        reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, images)

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
