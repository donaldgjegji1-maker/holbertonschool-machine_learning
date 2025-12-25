#!/usr/bin/env python3
"""
Performs K-means on a dataset
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer, number of clusters
        iterations: positive integer, maximum number of iterations

    Returns:
        C, clss or None, None on failure
        C: numpy.ndarray of shape (k, d) containing centroid means
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    # Input validation
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape
    
    if k > n:
        return None, None

    # Get data bounds
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    # FIRST use of np.random.uniform: initialize centroids
    C = np.random.uniform(low=min_vals, high=max_vals, size=(k, d))

    # Main algorithm (LOOP 1)
    for _ in range(iterations):
        C_prev = np.copy(C)

        # Assign points to centroids (vectorized - no loop)
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        clss = np.argmin(distances, axis=1)

        # Update centroids (LOOP 2)
        for j in range(k):
            cluster_points = X[clss == j]

            if len(cluster_points) > 0:
                C[j] = cluster_points.mean(axis=0)
            else:
                # SECOND use of np.random.uniform: reinitialize empty cluster
                C[j] = np.random.uniform(low=min_vals, high=max_vals, size=d)

        # Check convergence
        if np.allclose(C_prev, C, atol=1e-8):
            break

    # Final assignment (vectorized - no loop)
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))
    clss = np.argmin(distances, axis=1)

    return C, clss
