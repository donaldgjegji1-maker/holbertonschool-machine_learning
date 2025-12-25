#!/usr/bin/env python3
"""
PCA on a dataset
"""

import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           where n is the number of data points and d is the number
           of dimensions. All dimensions have a mean of 0.
        var: fraction of the variance that the PCA transformation
             should maintain

    Returns:
        W: numpy.ndarray of shape (d, nd) containing the weights matrix
           that maintains var fraction of X's original variance, where
           nd is the new dimensionality
    """
    # Compute the SVD of X
    # X = U * S * Vt
    # where U is (n, n), S is (min(n,d),), Vt is (d, d)
    u, s, vt = np.linalg.svd(X)

    # Calculate the variance explained by each component
    # Variance is proportional to the square of singular values
    variance = s ** 2

    # Calculate cumulative variance ratio
    total_variance = np.sum(variance)
    cumulative_variance_ratio = np.cumsum(variance) / total_variance

    # Find the number of components needed to maintain var fraction
    # We want the first nd components where cumulative variance >= var
    nd = np.where(cumulative_variance_ratio >= var)[0]

    if len(nd) == 0:
        # If no single component reaches the threshold, use all components
        nd = len(s)
    else:
        # Take the first index where we meet or exceed the threshold
        nd = nd[0] + 1

    # The weight matrix W consists of the first nd principal components
    # Principal components are the rows of Vt (or columns of V)
    # W should have shape (d, nd)
    W = vt.T[:, :nd]

    return W
