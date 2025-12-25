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
    u, s, vt = np.linalg.svd(X)
    
    # Calculate the variance explained by each component
    variance = s ** 2
    
    # Calculate cumulative variance ratio
    total_variance = np.sum(variance)
    cumulative_variance_ratio = np.cumsum(variance) / total_variance
    
    # Find the number of components needed to maintain var fraction
    # Get the index of the first component where cumulative variance >= var
    for i in range(len(cumulative_variance_ratio)):
        if cumulative_variance_ratio[i] >= var:
            nd = i + 1
            break
    else:
        # If we never reach var, use all components
        nd = len(s)
    
    # The weight matrix W consists of the first nd principal components
    # W should have shape (d, nd)
    W = vt.T[:, :nd]
    
    return W