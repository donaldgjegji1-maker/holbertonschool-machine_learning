#!/usr/bin/env python3
"""
Module that defines a deep neural network performing binary classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for layer_idx in range(1, self.__L + 1):
            current_layer_size = layers[layer_idx - 1]

            if not isinstance(current_layer_size, int) \
               or current_layer_size < 1:
                raise TypeError("layers must be a list of positive integers")

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

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        A_L = cache[f'A{self.__L}']
        dZ = A_L - Y

        for i in range(self.__L, 0, -1):
            A_prev = cache[f'A{i-1}']

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights[f'W{i}'] = weights_copy[f'W{i}'] - alpha * dW
            self.__weights[f'b{i}'] = weights_copy[f'b{i}'] - alpha * db

            if i > 1:
                W = weights_copy[f'W{i}']
                A_prev = cache[f'A{i-1}']
                dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the deep neural network
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        steps = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            if i == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)

                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

            if i > 0 and i % step == 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)

                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

            if i == iterations and i % step != 0:
                cost = self.cost(Y, A)
                costs.append(cost)
                steps.append(i)

                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            plt.plot(steps, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the instance object to a file in pickle format
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a pickled DeepNeuralNetwork object
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, IOError, pickle.UnpicklingError):
            return None
