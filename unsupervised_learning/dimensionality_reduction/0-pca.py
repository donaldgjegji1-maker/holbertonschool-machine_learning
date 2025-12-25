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
    # Compute SVD
    _, s, vt = np.linalg.svd(X)

    # Calculate variance ratios
    var_ratios = (s ** 2) / np.sum(s ** 2)

    # Calculate cumulative variance
    cumulative_var = np.cumsum(var_ratios)

    # Find number of components needed
    nd = np.argmax(cumulative_var >= var) + 1

    # Check if we should include one more component
    # This handles edge cases where variance is very close to threshold
    if nd < len(s) and var_ratios[nd] > 1e-10:
        nd += 1

    # Return weight matrix
    return vt[:nd].T
