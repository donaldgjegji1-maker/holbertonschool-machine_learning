#!/usr/bin/env python3
"""
Contains the function `initialize` to set up initial
centroids for K-means clustering using a multivariate uniform distribution.
"""

import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0):
        return None

    n, d = X.shape
    if k > n:
        return None

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    return np.random.uniform(mins, maxs, size=(k, d))
