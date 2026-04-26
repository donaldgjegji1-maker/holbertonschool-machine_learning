#!/usr/bin/env python3
"""
Recurrent Neural Networks Module
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    A function that performs forward propagation for a deep RNN
    """
    t, m, _ = X.shape
    num_layers = len(rnn_cells)
    h = h_0.shape[2]

    H = np.zeros((t + 1, num_layers, m, h))
    H[0] = h_0

    Y_list = []

    for step in range(t):
        x = X[step]
        for layer, cell in enumerate(rnn_cells):
            h_next, y = cell.forward(H[step, layer], x)
            H[step + 1, layer] = h_next
            x = h_next
        Y_list.append(y)

    Y = np.array(Y_list)
    return H, Y
