#!/usr/bin/env python3
"""
One-hot encoding module
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix
    """
    if not isinstance(Y, np.ndarray):
        return None

    if len(Y.shape) != 1:
        return None

    if not isinstance(classes, int) or classes <= 0:
        return None

    if np.max(Y) >= classes or np.min(Y) < 0:
        return None

    m = Y.shape[0]

    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1

    return one_hot
