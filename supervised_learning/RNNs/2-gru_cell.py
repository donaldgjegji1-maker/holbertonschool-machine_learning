#!/usr/bin/env python3
"""
Recurrent Neural Networks Module
"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit cell"""

    def __init__(self, i, h, o):
        """
        Initialize GRUCell

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def _sigmoid(self, x):
        """Applies sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """Applies numerically stable softmax activation function"""
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
                    hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data input

        Returns:
            h_next: the next hidden state
            y: the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = self._sigmoid(concat @ self.Wz + self.bz)
        r = self._sigmoid(concat @ self.Wr + self.br)

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(concat_r @ self.Wh + self.bh)

        h_next = (1 - z) * h_prev + z * h_tilde

        y = self._softmax(h_next @ self.Wy + self.by)

        return h_next, y
