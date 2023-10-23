import os
import pandas as pd
import pdb
import yaml

def load_config(path : str) -> dict:
    """
    Loads and parses a YAML configuration file
    """
    with open(path, 'r', encoding='utf-8') as yamlfile:
        cfg = yaml.safe_load(yamlfile)

    return cfg