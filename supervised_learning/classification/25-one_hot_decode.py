#!/usr/bin/env python3
"""
One-hot decoding module
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    Convert a one-hot matrix into a vector of labels
    """
    if not isinstance(one_hot, np.ndarray):
        return None

    if len(one_hot.shape) != 2:
        return None

    classes, m = one_hot.shape

    column_sums = np.sum(one_hot, axis=0)
    if not np.allclose(column_sums, 1):
        return None

    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    labels = np.argmax(one_hot, axis=0)

    return labels
