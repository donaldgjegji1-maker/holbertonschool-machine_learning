#!/usr/bin/env python3
"""
Contains the function `variance` to calculate the total
intra-cluster variance for a dataset X given centroids C.
"""

import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a dataset.

    Parameters:
    ----------
    X : numpy.ndarray
        Shape (n, d), the dataset.
    C : numpy.ndarray
        Shape (k, d), the centroid means for each cluster.

    Returns:
    -------
    float or None
        Total variance or None on failure.
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if X.ndim != 2 or C.ndim != 2:
        return None

    # Compute squared distances of each point to each centroid
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]  # shape (n, k, d)
    sq_dist = np.sum(diff ** 2, axis=2)               # shape (n, k)

    # Assign each point to the closest centroid
    closest = np.min(sq_dist, axis=1)                 # shape (n,)

    # Total intra-cluster variance
    total_var = np.sum(closest)
    return total_var
