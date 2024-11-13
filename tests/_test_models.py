import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import patch, MagicMock
from vae_models import VAE
from feature_classifier import FeatureClassifier
from utils import train_model, evaluate_classifier
from main import generate_samples


@patch.object(VAE, 'decoder', return_value=torch.rand(10, 1, 28, 28))
@patch.object(VAE, 'prior_means', return_value=torch.ones(10, 20))
@patch.object(VAE, 'prior_logvars', return_value=torch.ones(10, 20))
@patch.object(FeatureClassifier, 'forward', return_value=(torch.ones(10, 10),))
def test_generate_samples(mock_classifier, mock_prior_means, mock_prior_logvars, mock_decoder):
    vae = VAE(conditional=True, alpha=0.1, continual_backprop=True, num_classes=10, latent_dim=20, encoder_config={},
              decoder_config={}, cbp_config={})
    classifier = FeatureClassifier(input_size=100, hidden1=50, hidden2=50, num_classes=10, continual_backprop=True,
                                   cbp_config={})
    device = torch.device('cpu')

    samples = generate_samples(vae, classifier, 0, 5, device)
    assert len(samples) == 5


@patch('torch.optim.Adam')
def test_train_model(mock_Adam):
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters())
    dataloader = [(torch.rand(10, 10), torch.randint(0, 2, (10,))) for _ in range(2)]
    device = torch.device('cpu')

    train_model(model, optimizer, dataloader, device, epochs=1, is_classifier=True)


@patch.object(torch.nn.Module, 'eval')
def test_evaluate_model(mock_eval):
    model = nn.Linear(10, 1)
    dataloader = [(torch.rand(10, 10), torch.randint(0, 2, (10,))) for _ in range(2)]
    device = torch.device('cpu')

    accuracy = evaluate_model(model, dataloader, device)
    assert isinstance(accuracy, float)