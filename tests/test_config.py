import pytest
import yaml
from utils import load_config

def test_load_config(tmp_path):
    config_content = {'key': 'value'}
    config_file = tmp_path / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.safe_dump(config_content, f)

    config = load_config(config_file)
    assert config == config_content