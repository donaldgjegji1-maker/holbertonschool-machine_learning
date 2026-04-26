#!/usr/bin/env python3
"""
Recurrent Neural Networks Module
"""

import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(concat @ self.Wh + self.bh)
        y_linear = h_next @ self.Wy + self.by
        exp = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp / exp.sum(axis=1, keepdims=True)
        return h_next, y
