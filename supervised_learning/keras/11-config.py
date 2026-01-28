#!/usr/bin/env python3
"""
Functions for saving and loading Keras model configuration
"""
import tensorflow.keras as K
import json


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format
    """
    config = network.get_config()

    with open(filename, 'w') as f:
        json.dump(config, f)


def load_config(filename):
    """
    Loads a model with a specific configuration
    """
    with open(filename, 'r') as f:
        config = json.load(f)

    model = K.models.model_from_config(config)

    return model
