#!/usr/bin/env python3
"""
Test a neural network
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network
    """
    results = network.evaluate(
        x=data,
        y=labels,
        verbose=verbose
    )

    return results
