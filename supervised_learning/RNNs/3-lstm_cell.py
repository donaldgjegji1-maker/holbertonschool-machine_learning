#!/usr/bin/env python3
"""
Recurrent Neural Networks Module
"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Initialize LSTMCell

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def _sigmoid(self, x):
        """Applies sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """Applies numerically stable softmax activation function"""
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
            c_prev: numpy.ndarray of shape (m, h) containing the previous
                    cell state
            x_t: numpy.ndarray of shape (m, i) containing the data input

        Returns:
            h_next: the next hidden state
            c_next: the next cell state
            y: the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = self._sigmoid(concat @ self.Wf + self.bf)
        u = self._sigmoid(concat @ self.Wu + self.bu)
        c_tilde = np.tanh(concat @ self.Wc + self.bc)
        o = self._sigmoid(concat @ self.Wo + self.bo)

        c_next = f * c_prev + u * c_tilde
        h_next = o * np.tanh(c_next)

        y = self._softmax(h_next @ self.Wy + self.by)

        return h_next, c_next, y
