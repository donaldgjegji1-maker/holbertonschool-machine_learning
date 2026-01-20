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

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        for layer_size in layers:
            if not isinstance(layer_size, int) or layer_size < 1:
                raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx in range(1, self.__L + 1):
            current_layer_size = layers[layer_idx - 1]

            if layer_idx == 1:
                prev_layer_size = nx
            else:
                prev_layer_size = layers[layer_idx - 2]

            self.__weights['W' + str(layer_idx)] = np.random.randn(
                current_layer_size, prev_layer_size
            ) * np.sqrt(2 / prev_layer_size)

            self.__weights['b' + str(layer_idx)] = np.zeros(
                (current_layer_size, 1)
            )

    @property
    def L(self):
        """Getter for L (number of layers)"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache dictionary"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the neural network
        """
        self.__cache['A0'] = X
        for layer_idx in range(1, self.__L + 1):
            W = self.__weights['W' + str(layer_idx)]
            b = self.__weights['b' + str(layer_idx)]

            A_prev = self.__cache['A' + str(layer_idx - 1)]

            Z = np.dot(W, A_prev) + b

            A = 1 / (1 + np.exp(-Z))

            self.__cache['A' + str(layer_idx)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression
        """
        m = Y.shape[1]

        cost = -(1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions
        """
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)

        return predictions, cost
