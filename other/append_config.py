import json
import os
from pathlib import Path

def deep_update(d, u):
    """Recursively update dict d with dict u"""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def append_config(folder, new_data, filename="config.json"):
    """
    Appends data to the config file in the derivatives folder
    

    Args:
        derivatives_base: Path to derivatives_base
        data: dictionary with configuration data
    """
    config_file = Path(folder) / filename

    if config_file.exists():
        with open(config_file, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # merge deeply instead of replacing
    deep_update(existing_data, new_data)

    with open(config_file, "w") as f:
        json.dump(existing_data, f, indent=4)

def deep_update(d, u):
    """Recursively update dict d with dict u"""
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def append_config(folder, new_data, filename="config.json"):
    """
    Appends data to the config file in the derivatives folder
    

    Args:
        derivatives_base: Path to derivatives_base
        data: dictionary with configuration data
    """
    config_file = Path(folder) / filename

    if config_file.exists():
        with open(config_file, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # merge deeply instead of replacing
    deep_update(existing_data, new_data)

    with open(config_file, "w") as f:
        json.dump(existing_data, f, indent=4)
