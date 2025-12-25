#!/usr/bin/env python3
"""
Maximization step in EM algorithm GMM
"""

import numpy as np


def maximization(X, g):
    """
    Calculate the maximization step in EM algorithm.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        g: numpy.ndarray of shape (k, n) containing posterior probabilities

    Returns:
        tuple: (pi, m, S) where:
            pi: numpy.ndarray of shape (k,) with updated priors
            m: numpy.ndarray of shape (k, d) with updated means
            S: numpy.ndarray of shape (k, d, d) with updated covariances
        or (None, None, None) on failure
    """
    # Input validation
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n_g != n:
        return None, None, None

    # Check if g contains valid probabilities
    if not np.allclose(np.sum(g, axis=0), 1):
        return None, None, None

    # Calculate effective number of points per cluster
    N_k = np.sum(g, axis=1)  # shape (k,)

    # Update priors
    pi = N_k / n

    # Update means
    m = (g @ X) / N_k[:, np.newaxis]

    # Initialize covariance matrices
    S = np.zeros((k, d, d))

    # Update covariance matrices
    for i in range(k):
        # Centered data
        diff = X - m[i]  # shape (n, d)

        # Weighted outer products
        weighted = g[i, :, np.newaxis] * diff  # shape (n, d)

        # Covariance matrix
        S[i] = (weighted.T @ diff) / N_k[i]

    return pi, m, S
