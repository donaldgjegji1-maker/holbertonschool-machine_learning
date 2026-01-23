#!/usr/bin/env python3
"""
Module that defines a deep neural network performing multiclass classification
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Class that defines a dnn performing multiclass classification
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize the neural network
        nx: number of input features
        layers: a list representing the number of nodes in each layer
        activation: activation function for hidden layers ('sig' or 'tanh')
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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

    @property
    def activation(self):
        """Getter for activation function"""
        return self.__activation

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
            if layer_idx == self.__L:
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            else:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)

            self.__cache['A' + str(layer_idx)] = A
            self.__cache['Z' + str(layer_idx)] = Z

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost of the model using cross-entropy for multiclass
        """
        m = Y.shape[1]
        A_clipped = np.clip(A, 1e-10, 1 - 1e-10)
        cost = -(1 / m) * np.sum(Y * np.log(A_clipped))

        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions for multiclass classification
        """
        A, _ = self.forward_prop(X)

        predictions = np.argmax(A, axis=0)
        cost = self.cost(Y, A)

        predictions_one_hot = np.zeros_like(A)
        predictions_one_hot[predictions, np.arange(predictions.shape[0])] = 1

        return predictions_one_hot, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Perform one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        dZ = {}
        dW = {}
        db = {}

        for layer_idx in range(self.__L, 0, -1):
            A_current = cache['A' + str(layer_idx)]
            A_prev = cache['A' + str(layer_idx - 1)]
            Z_current = cache.get('Z' + str(layer_idx), None)

            if layer_idx == self.__L:
                dZ[str(layer_idx)] = A_current - Y
            else:
                W_next = self.__weights['W' + str(layer_idx + 1)]
                dZ_next = dZ[str(layer_idx + 1)]

                if self.__activation == 'sig':
                    derivative = A_current * (1 - A_current)
                else:
                    derivative = 1 - np.square(A_current)

                dZ[str(layer_idx)] = np.dot(W_next.T, dZ_next) * derivative

            dW[str(layer_idx)] = (1 / m) * np.dot(
                dZ[str(layer_idx)], A_prev.T
            )
            db[str(layer_idx)] = (1 / m) * np.sum(
                dZ[str(layer_idx)], axis=1, keepdims=True
            )
            self.__weights['W' + str(layer_idx)] -= alpha * dW[str(layer_idx)]
            self.__weights['b' + str(layer_idx)] -= alpha * db[str(layer_idx)]

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
