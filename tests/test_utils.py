import torch.nn as nn
from utils import count_parameters

def test_count_parameters():
    model = nn.Linear(10, 1)
    assert count_parameters(model) == 11