#!/usr/bin/env python3
"""
Make predictions using a neural network
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network
    """
    predictions = network.predict(
        x=data,
        verbose=verbose
    )

    return predictions
