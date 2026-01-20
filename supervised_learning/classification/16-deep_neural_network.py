#!/usr/bin/env python3
"""
Module that defines a deep neural network performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class that defines a deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        Initialize the neural network
        nx: number of input features
        layers: a list representing the number of nodes in each layer
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for layer_size in layers:
            if not isinstance(layer_size, int) or layer_size < 1:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for layer_idx in range(1, self.L + 1):
            layer_size = layers[layer_idx-1]

            if layer_idx == 1:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[layer_idx-2]
            self.weights['W' + str(layer_idx)] = np.random.randn(
                current_layer_size, prev_layer_size
            ) * np.sqrt(2 / prev_layer_size)
            self.weights['b' + str(layer_idx)] = np.zeros((layer_size, 1))
