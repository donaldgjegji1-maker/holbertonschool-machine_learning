#!/usr/bin/env python3
"""
Module that that defines a neural network with one hidden layer
performing binary classification
"""

import numpy as np


class NeuralNetwork:
    """
    A class that defines a neural network with one hidden layer
    performing binary classification
    """

    def __init__(self, nx, nodes):
        """
        Initialize the neural network

        nx: number of input features
        nodes: number of nodes in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))

        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for hidden layer weights"""
        return self.__W1

    @property
    def b1(self):
        """Getter for hidden layer bias"""
        return self.__b1

    @property
    def A1(self):
        """Getter for hidden layer activated output"""
        return self.__A1

    @property
    def W2(self):
        """Getter for output layer weights"""
        return self.__W2

    @property
    def b2(self):
        """Getter for output layer bias"""
        return self.__b2

    @property
    def A2(self):
        """Getter for output layer activated output"""
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """

        m = Y.shape[1]
        log_loss = -1/m*np.sum(Y*np.log(A) + (1-Y)*(np.log(1.0000001-A)))
        return log_loss

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions
        """
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        result = np.where(A2 >= 0.5, 1, 0)
        return result, cost
