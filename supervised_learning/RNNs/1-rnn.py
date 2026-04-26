#!/usr/bin/env python3
"""
Recurrent Neural Networks Module
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    A function that performs forward propagation for a simple RNN
    """
    t, m, _ = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    Y_list = []

    for step in range(t):
        H[step + 1], y = rnn_cell.forward(H[step], X[step])
        Y_list.append(y)

    Y = np.array(Y_list)
    return H, Y
