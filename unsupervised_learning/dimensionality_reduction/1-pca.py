#!/usr/bin/env python3
"""
PCA v2 - performs PCA on a dataset
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) where:
           n is the number of data points
           d is the number of dimensions in each point
        ndim: new dimensionality of the transformed X

    Returns:
        T: numpy.ndarray of shape (n, ndim) containing the
           transformed version of X
    """
    # Center the data by subtracting the mean
    X_mean = X - np.mean(X, axis=0)

    # Perform SVD
    _, _, vt = np.linalg.svd(X_mean, full_matrices=False)

    # Get the weight matrix (first ndim principal components)
    W = vt[:ndim].T

    # Transform X
    T = np.matmul(X_mean, W)

    return T
