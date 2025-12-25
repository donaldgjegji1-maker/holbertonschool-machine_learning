#!/usr/bin/env python3
"""
Gaussian Mixture Model initialization module
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize variables for a Gaussian Mixture Model.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    Returns:
        tuple: (pi, m, S) where:
            pi: numpy.ndarray of shape (k,) with priors for each cluster
            m: numpy.ndarray of shape (k, d) with centroid means
            S: numpy.ndarray of shape (k, d, d) with covariance matrices
        or (None, None, None) on failure
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None

    if not isinstance(k, int) or k <= 0:
        return None, None, None

    n, d = X.shape

    if k > n:
        return None, None, None

    try:
        pi = np.full(k, 1.0 / k)

        # Get centroids from K-means
        m, _ = kmeans(X, k)

        if m is None:
            return None, None, None

        # Initialize covariance matrices as identity matrices
        # Shape: (k, d, d)
        S = np.tile(np.eye(d), (k, 1, 1))

        return pi, m, S

    except Exception:
        return None, None, None
