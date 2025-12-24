#!/usr/bin/env python3
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering
    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    if k > n:
        return None, None

    # Initialize centroids (1st use of np.random.uniform)
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    C = np.random.uniform(mins, maxs, size=(k, d))

    for _ in range(iterations):  # loop 1
        C_prev = C.copy()

        # Assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(distances, axis=1)

        # Update centroids
        for i in range(k):  # loop 2
            points = X[clss == i]
            if points.size == 0:
                # Reinitialize centroid (2nd use of np.random.uniform)
                C[i] = np.random.uniform(mins, maxs, size=(1, d))
            else:
                C[i] = points.mean(axis=0)

        # Stop if no change
        if np.allclose(C, C_prev):
            break

    return C, clss
